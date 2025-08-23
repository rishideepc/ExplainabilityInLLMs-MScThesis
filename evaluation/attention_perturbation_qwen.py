# evaluation/attention_perturbation_qwen.py
import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention  # Qwen2 / Qwen2.5

def normalize_token(token):
    return re.sub(r'\W+', '', token.replace("▁", "").lower())

def get_token_indices(tokenizer, text, target_words):
    # Same logic as other backends: char-token overlap on the raw claim text
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

        start_pos = 0
        while True:
            word_start = text_lower.find(target_word, start_pos)
            if word_start == -1:
                break
            word_end = word_start + len(target_word)

            # word-boundary checks
            if word_start > 0 and text_lower[word_start - 1].isalnum():
                start_pos = word_start + 1
                continue
            if word_end < len(text_lower) and text_lower[word_end].isalnum():
                start_pos = word_start + 1
                continue

            word_indices = []
            for i, (char_start, char_end) in enumerate(offsets):
                if char_start < word_end and char_end > word_start:
                    word_indices.append(i)

            indices.extend(word_indices)
            print(f"Word match: '{target_word}' at chars {word_start}-{word_end} -> tokens {word_indices} ({[tokens[j] for j in word_indices]})")
            start_pos = word_end
            break  # first occurrence only

    indices = list(dict.fromkeys(indices))
    print(f"Final matched token indices: {indices}")
    return indices

class CustomQwenAttention(Qwen2Attention):
    def __init__(self, config, layer_idx, target_indices=None):
        super().__init__(config, layer_idx)
        self.target_indices = target_indices or []
        self.original_attention = None
        self.perturbed_attention = None
        self.forward_call_count = 0

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        self.forward_call_count += 1
        print(f"🚀 CustomQwenAttention.forward() called #{self.forward_call_count}")
        print(f"   - hidden_states shape: {hidden_states.shape}")
        print(f"   - [OVERRIDE] output_attentions: {output_attentions}")
        print(f"   - target_indices: {self.target_indices}")

        # Force attention output so we can capture probs
        output_attentions = True

        output = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )

        # Qwen2Attention returns (attn_output, attn_weights, past_key_value) when output_attentions=True
        if isinstance(output, tuple):
            print(f"   - Attention shape from super: {output[1].shape if output[1] is not None else 'None'}")

        if self.target_indices and len(output) > 1 and output[1] is not None:
            attention_probs = output[1].to(hidden_states.device)  # [b, heads, q_len, k_len]
            print(f"   - Attention probs shape: {attention_probs.shape}")

            if self.original_attention is None:
                self.original_attention = attention_probs.clone().detach()
                print(f"✅ Stored original attention weights: {self.original_attention.shape}")

            print(f"🔧 Modifying attention weights for indices: {self.target_indices}")

            modified = False
            for idx in self.target_indices:
                if idx < attention_probs.shape[-1]:
                    attention_probs[:, :, :, idx] = 0.0
                    modified = True
                    print(f"   - Zeroed attention for token {idx}")
                else:
                    print(f"   - WARNING: Index {idx} out of bounds for shape {attention_probs.shape}")

            if modified:
                den = attention_probs.sum(dim=-1, keepdim=True)
                attention_probs = attention_probs / (den + 1e-8)
                zero_mask = (den == 0)
                if zero_mask.any():
                    attention_probs[zero_mask.expand_as(attention_probs)] = 1.0 / attention_probs.size(-1)
                print(f"   - Renormalized attention weights")

            self.perturbed_attention = attention_probs.clone().detach()
            print(f"✅ Stored perturbed attention weights: {self.perturbed_attention.shape}")

            return output[0], attention_probs if output_attentions else output[0]
        else:
            print(f"   - Skipping perturbation: target_indices={bool(self.target_indices)}, has_attention={len(output) > 1 and output[1] is not None}")

        return output

def visualize_attention_weights(model, tokenizer, text, target_indices, max_tokens_to_show=20, save_path=None):
    print("🔍 Looking for custom attention layers...")

    custom_layer = None
    for i, layer in enumerate(model.model.layers):
        print(f"Layer {i}: {type(layer.self_attn).__name__}")
        if hasattr(layer.self_attn, 'original_attention'):
            print(f"  - Has original_attention: {layer.self_attn.original_attention is not None}")
            if layer.self_attn.original_attention is not None:
                print(f"  - Original attention shape: {layer.self_attn.original_attention.shape}")
                custom_layer = layer.self_attn
                break
        if hasattr(layer.self_attn, 'target_indices'):
            print(f"  - Has target_indices: {layer.self_attn.target_indices}")

    if custom_layer is None:
        print("❌ No CustomQwenAttention layers found!")
        return

    if custom_layer.original_attention is None:
        print("❌ Found CustomQwenAttention but original_attention is None!")
        print(f"   Target indices were: {custom_layer.target_indices}")
        return

    print(f"✅ Found attention weights! Shape: {custom_layer.original_attention.shape}")

    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens_to_show:
        tokens = tokens[:max_tokens_to_show]

    orig_attn = custom_layer.original_attention[0, 0].cpu().numpy()
    pert_attn = custom_layer.perturbed_attention[0, 0].cpu().numpy()

    orig_attn = orig_attn[:max_tokens_to_show, :max_tokens_to_show]
    pert_attn = pert_attn[:max_tokens_to_show, :max_tokens_to_show]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.heatmap(orig_attn, annot=False, cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Original Attention Weights')
    axes[0,0].set_xlabel('Key Tokens')
    axes[0,0].set_ylabel('Query Tokens')

    sns.heatmap(pert_attn, annot=False, cmap='Reds', ax=axes[0,1])
    axes[0,1].set_title('Perturbed Attention Weights')
    axes[0,1].set_xlabel('Key Tokens')
    axes[0,1].set_ylabel('Query Tokens')

    orig_target_attn = [orig_attn[:, i].mean() if i < orig_attn.shape[1] else 0 for i in range(len(tokens))]
    pert_target_attn = [pert_attn[:, i].mean() if i < pert_attn.shape[1] else 0 for i in range(len(tokens))]

    axes[1,0].bar(range(len(tokens)), orig_target_attn, alpha=0.7, color='blue', label='Original')
    axes[1,0].bar(range(len(tokens)), pert_target_attn, alpha=0.7, color='red', label='Perturbed')
    axes[1,0].set_title('Average Attention to Each Token')
    axes[1,0].set_xlabel('Token Position')
    axes[1,0].set_ylabel('Average Attention Weight')
    axes[1,0].legend()

    for idx in target_indices:
        if idx < len(tokens):
            axes[1,0].axvline(x=idx, color='red', linestyle='--', alpha=0.5)

    token_labels = [f"{i}: {token.replace('▁', '')}" for i, token in enumerate(tokens)]
    axes[1,1].axis('off')
    axes[1,1].text(0.1, 0.9, "Tokens:", fontsize=12, fontweight='bold', transform=axes[1,1].transAxes)

    for i, label in enumerate(token_labels):
        color = 'red' if i in target_indices else 'black'
        weight = 'bold' if i in target_indices else 'normal'
        axes[1,1].text(0.1, 0.85 - i*0.04, label, fontsize=10, color=color, weight=weight, transform=axes[1,1].transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to: {save_path}")
    else:
        plt.savefig('attention_visualization.png', dpi=300, bbox_inches='tight')
        print("Attention visualization saved to: attention_visualization.png")

    plt.show()

    print("\n" + "="*60)
    print("ATTENTION WEIGHT ANALYSIS")
    print("="*60)

    for i, token in enumerate(tokens):
        if i >= orig_attn.shape[1]:
            break
        orig_avg = orig_attn[:, i].mean()
        pert_avg = pert_attn[:, i].mean()
        status = "🎯 PERTURBED" if i in target_indices else ""
        print(f"Token {i:2d}: {token:>12} | Orig: {orig_avg:.4f} | Pert: {pert_avg:.4f} | Δ: {pert_avg-orig_avg:+.4f} {status}")

def replace_qwen_attention(model, target_indices, last_n_layers=1):
    device = next(model.parameters()).device
    total_layers = len(model.model.layers)

    print(f"🔄 Replacing attention in last {last_n_layers} layer(s) out of {total_layers}")
    print(f"🎯 Target indices: {target_indices}")

    for i in range(total_layers):
        if i >= total_layers - last_n_layers:
            print(f"   Replacing layer {i}...")
            original_layer = model.model.layers[i].self_attn
            print(f"   Original layer type: {type(original_layer).__name__}")

            custom_attn = CustomQwenAttention(model.config, i, target_indices)
            try:
                custom_attn.load_state_dict(original_layer.state_dict(), strict=False)
                print(f"   ✅ Loaded state dict successfully")
            except Exception as e:
                print(f"   ❌ Error loading state dict: {e}")

            custom_attn = custom_attn.to(device).half()
            model.model.layers[i].self_attn = custom_attn

            replaced_layer = model.model.layers[i].self_attn
            print(f"   New layer type: {type(replaced_layer).__name__}")
            print(f"   Target indices set: {replaced_layer.target_indices}")

    print("✅ Attention layer replacement complete")

def run_with_attention_perturbation(model, tokenizer, text, target_tokens, visualize=False, save_path=None):
    model.eval()
    model = model.half()

    # Respect Accelerate's device placement
    device = next(model.parameters()).device
    print(f"Resolved model device: {device}")

    messages = [
        {"role": "system", "content": "You are a factual claim evaluator. Respond with TRUE or FALSE."},
        {"role": "user", "content": f'Claim: "{text}"\n\nAnswer: TRUE or FALSE'}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    for k in inputs:
        if inputs[k].dtype == torch.float:
            inputs[k] = inputs[k].to(dtype=torch.float16)

    print(f"input_ids device: {inputs['input_ids'].device}")
    print(f"model param device: {device}")

    target_indices = get_token_indices(tokenizer, text, target_tokens)
    if not target_indices:
        print("No matching token indices found for rationale tokens.")
        return "UNKNOWN", 0.5

    replace_qwen_attention(model, target_indices, last_n_layers=1)

    with torch.no_grad():
        print("⚙️ Calling model with output_attentions=True")
        outputs = model(**inputs, output_attentions=True)

    logits = outputs.logits  # (1, seq_len, vocab)
    last_logits = logits[0, -1]

    probs = torch.nn.functional.softmax(last_logits, dim=-1)
    true_id = tokenizer.convert_tokens_to_ids("true")
    false_id = tokenizer.convert_tokens_to_ids("false")

    true_conf = probs[true_id].item() if true_id in tokenizer.get_vocab().values() else 0.0
    false_conf = probs[false_id].item() if false_id in tokenizer.get_vocab().values() else 0.0

    label = "TRUE" if true_conf > false_conf else "FALSE"
    confidence = max(true_conf, false_conf)

    print(f"Manual prediction: {label} | TRUE_conf={true_conf:.4f}, FALSE_conf={false_conf:.4f}")

    if visualize:
        visualize_attention_weights(model, tokenizer, text, target_indices, save_path=save_path)

    return label, confidence

