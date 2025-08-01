# evaluation/attention_perturbation_llama.py

import torch
import re
from transformers.models.llama.modeling_llama import LlamaAttention

def normalize_token(token):
    return re.sub(r'\W+', '', token.replace("▁", "").lower())

def get_token_indices(tokenizer, text, target_words):
    # Get tokenization with character offset mapping
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']
    
    print(f"Tokens: {tokens}")
    print(f"Target words: {target_words}")
    
    indices = []
    text_lower = text.lower()
    
    for target_word in target_words:
        target_word = target_word.lower().strip()
        if not target_word:
            continue
        
        # Find all occurrences of the target word in the text
        start_pos = 0
        while True:
            word_start = text_lower.find(target_word, start_pos)
            if word_start == -1:
                break
                
            word_end = word_start + len(target_word)
            
            # Check if this is a complete word (not part of another word)
            if word_start > 0 and text_lower[word_start-1].isalnum():
                start_pos = word_start + 1
                continue
            if word_end < len(text_lower) and text_lower[word_end].isalnum():
                start_pos = word_start + 1
                continue
            
            # Find tokens that overlap with this word
            word_indices = []
            for i, (char_start, char_end) in enumerate(offsets):
                if char_start < word_end and char_end > word_start:
                    word_indices.append(i)
            
            indices.extend(word_indices)
            print(f"Word match: '{target_word}' at chars {word_start}-{word_end} -> tokens {word_indices} ({[tokens[j] for j in word_indices]})")
            
            start_pos = word_end
            break  # Only match first occurrence of each word
    
    # Remove duplicates while preserving order
    indices = list(dict.fromkeys(indices))
    print(f"Final matched token indices: {indices}")
    return indices

class CustomLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx, target_indices=None):
        super().__init__(config, layer_idx)
        self.target_indices = target_indices or []

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        output = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,
            **kwargs
        )
        if self.target_indices and output[1] is not None:
            attention_probs = output[1].to(hidden_states.device)
            print(f"🔧 Modifying attention weights for indices: {self.target_indices}")
            for idx in self.target_indices:
                attention_probs[:, :, :, idx] = 0.0
            attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)
            return output[0], attention_probs
        return output

def replace_llama_attention(model, target_indices, last_n_layers=1):
    device = next(model.parameters()).device
    total_layers = len(model.model.layers)
    for i in range(total_layers):
        if i >= total_layers - last_n_layers:
            custom_attn = CustomLlamaAttention(model.config, i, target_indices)
            custom_attn.load_state_dict(model.model.layers[i].self_attn.state_dict(), strict=False)
            custom_attn = custom_attn.to(device).half()
            model.model.layers[i].self_attn = custom_attn
    print("Replaced final attention layer(s) with CustomLlamaAttention")

def run_with_attention_perturbation(model, tokenizer, text, target_tokens):
    model.eval()
    model = model.half()

    # prompt = (
    #     f"Q: Is the following claim true or false?\n"
    #     f"Claim: \"{text}\"\n\n"
    #     "Return your answer only as 'TRUE' or 'FALSE' and also provide your confidence from 0 to 100.\nA:"
    # )
    # inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    messages = [
            {"role": "system", "content": "You are a factual claim evaluator. Respond with TRUE or FALSE."},
            {"role": "user", "content": f'Claim: "{text}"\n\nAnswer: TRUE or FALSE'}
        ]
        
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    for key in inputs:
        if inputs[key].dtype == torch.float:
            inputs[key] = inputs[key].to(dtype=torch.float16)

    token_ids = inputs["input_ids"][0]
    print(f"input_ids device: {inputs['input_ids'].device}")
    print(f"model device: {next(model.parameters()).device}")

    target_indices = get_token_indices(tokenizer, text, target_tokens)

    if not target_indices:
        print("No matching token indices found for rationale tokens.")
        return "UNKNOWN", 0.5

    replace_llama_attention(model, target_indices, last_n_layers=1)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"Perturbed output: {output_text}")

    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    print(f"Again, Generated tokens only: {output_text}")

    label = "TRUE" if "TRUE" in output_text.upper() else "FALSE" if "FALSE" in output_text.upper() else "UNKNOWN"
    match = re.search(r"[Cc]onfidence\s*[:=]?\s*(\d{1,3})", output_text)
    confidence = float(match.group(1)) / 100 if match else 0.5

    print(f"Perturbed label: {label} | confidence: {confidence:.2f}")
    return label, confidence
