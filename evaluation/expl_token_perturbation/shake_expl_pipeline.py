import os
import re
import torch
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("SHAKE_MODEL", "meta-llama/Llama-2-7b-chat-hf")

START_CLAIM, END_CLAIM = "<<CLAIM>>", "<<ENDCLAIM>>"
START_RAT,   END_RAT   = "<<RAT>>",   "<<ENDRAT>>"


class ShakeExplPipeline:
    """
    Pipeline for explanation-perturbing SHAKE variant; builds relevant prompts for explanation generation, label prediction
    keyword/rationale generation; extracts keywords, and maps keyword token indices to correct explanation indices

    @params: 
        claims/explanations, model environment details 
    
    @returns: 
        prompts, explanations, keywords, indices
    """
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=os.environ.get("HF_TOKEN", None),
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.device = next(self.model.parameters()).device

    def build_prompt_explain_only(self, claim: str) -> str:
        """
        Builds prompt to elicit a brief CoT explanation only

        @params: 
            input claim 
        
        @returns: 
            required prompt string
        """
        messages = [
            {"role": "system",
             "content": "Explain briefly in 1–3 sentences. Do NOT give a final label (TRUE/FALSE)."},
            {"role": "user",
             "content": (f'Claim: "{START_CLAIM}{claim}{END_CLAIM}"\n'
                         f'Let us think step-by-step. Respond as:\n'
                         f'Explanation: ')}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def build_prompt_classify_with_expl(self, claim: str, explanation: str) -> str:
        """
        Builds prompt with claim + explanation that asks for a TRUE/FALSE label only

        @params: 
            claim + explanation
            
        @returns: 
            required prompt string
        """
        messages = [
            {"role": "system", "content": "You are a factual claim evaluator."},
            {"role": "user", "content": (
                f'Claim: "{START_CLAIM}{claim}{END_CLAIM}"\n'
                f'Explanation: "{START_RAT}{explanation}{END_RAT}"\n'
                f'Output only TRUE or FALSE.'
            )}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def build_prompt_keywords_from_expl(self, explanation: str, max_k: int = 8) -> str:
        """
        Builds prompt to elicit upto K keywords from the explanation

        @params: 
            explanation text, number of keywords to extract 
            
        @returns: 
            required prompt string
        """
        messages = [
            {"role": "system", "content":
                f"You are a keyword extractor. Output ONLY up to {max_k} important words from the EXPLANATION. "
                "Do not output sentences, punctuation, or filler. Output must be space-separated words only."},
            {"role": "user", "content":
                f'EXPLANATION: "{explanation}"\n\n'
                f'Output format example: word1 word2 word3\n\n'
                f'Now extract up to {max_k} words:'}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate_text(self, prompt: str, max_new_tokens: int = 128) -> str:
        """
        Greedy decoding without sampling

        @params: 
            prompt string, max number of tokens 
        
        @returs: 
            generated text string
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        for k in inputs:
            if inputs[k].dtype == torch.float:
                inputs[k] = inputs[k].to(dtype=torch.float16)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        gen = out[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def explain_only(self, claim: str) -> str:
        """
        Generates an explanation-only response

        @params: 
            input claim 
        
        @returns: 
            CoT explanation only
        """
        prompt = self.build_prompt_explain_only(claim)
        text = self._generate_text(prompt, max_new_tokens=128)
        m = re.search(r'Explanation:\s*(.*)', text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            expl = m.group(1).strip()
        else:
            expl = text.strip()
        expl = re.sub(r'\b(TRUE|FALSE)\b\.?', '', expl, flags=re.IGNORECASE).strip()
        print(f"[EXPLAIN] {expl}")
        return expl

    def classify_with_expl(self, claim: str, explanation: str) -> Tuple[str, float, str]:
        """
        Returns a label prediction prompt that includes claim + CoT explanation

        @params: claim + explanation 
        
        @returns: prompt string
        """
        prompt = self.build_prompt_classify_with_expl(claim, explanation)
        return prompt 

    def keywords_from_expl(self, explanation: str, max_k: int = 8) -> List[str]:
        """
        Extracts upto K keywords from the explanation

        @params: explanation text, max number of keywords 
        
        @returns: list of unique lowercase words (keywords)
        """
        prompt = self.build_prompt_keywords_from_expl(explanation, max_k=max_k)
        text = self._generate_text(prompt, max_new_tokens=24)
        bad_tokens = {"sure","here","are","the","key","words","extracted","from","explanation", "important", "now", "up", "to", "claim","most"}
        raw = re.findall(r"[A-Za-z]{2,}", text)
        raw = [w.lower() for w in raw if w.lower() not in bad_tokens]
        seen, out = set(), []
        for w in raw:
            lw = w.lower()
            if lw not in seen:
                seen.add(lw)
                out.append(lw)
            if len(out) >= max_k:
                break
        print(f"[KEYWORDS] {out}")
        return out

    def indices_for_keywords_in_expl_span(self, full_prompt: str, keywords: List[str]) -> List[int]:
        """
        Maps keywords to token indices inside the <<RAT>>…<<ENDRAT>> span of a given prompt

        @params: full prompt string, list of keywords

        @returns: list of token indices
        """
        txt = full_prompt
        try:
            s = txt.index(START_RAT) + len(START_RAT)
            e = txt.index(END_RAT)
        except ValueError:
            print("!! Could not find RAT span in prompt.")
            return []

        enc = self.tokenizer(txt, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=2048)
        offsets = enc["offset_mapping"]
        txt_lower = txt.lower()

        indices = []
        for w in keywords:
            w = w.lower().strip()
            if not w:
                continue
            start = txt_lower.find(w, s, e)
            if start == -1:
                continue
            end = start + len(w)
            for i, (a, b) in enumerate(offsets):
                if a < end and b > start:
                    indices.append(i)

        seen, out = set(), []
        for i in indices:
            if i not in seen:
                seen.add(i)
                out.append(i)
        print(f"[IDX] Expl-target token indices in prompt: {out}")
        return out
