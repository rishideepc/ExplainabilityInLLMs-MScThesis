# evaluation/attention_perturbation_llama.py
import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
        print(f"🚀 CustomLlamaAttention.forward() called #{self.forward_call_count}")
        print(f"   - hidden_states shape: {hidden_states.shape}")
        print(f"   - [OVERRIDE] output_attentions: {output_attentions}")
        print(f"   - target_indices: {self.target_indices}")

        # ✅ Force attention output regardless of outer layers
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

        # ✅ Confirm attention presence
        if isinstance(output, tuple):
            print(f"   - Attention shape from super: {output[1].shape if output[1] is not None else 'None'}")

        if self.target_indices and len(output) > 1 and output[1] is not None:
            attention_probs = output[1].to(hidden_states.device)
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
                attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-8)
                print(f"   - Renormalized attention weights")

            self.perturbed_attention = attention_probs.clone().detach()
            print(f"✅ Stored perturbed attention weights: {self.perturbed_attention.shape}")

            return output[0], attention_probs if output_attentions else output[0]
        else:
            print(f"   - Skipping perturbation: target_indices={bool(self.target_indices)}, has_attention={len(output) > 1 and output[1] is not None}")

        return output


def visualize_attention_weights(model, tokenizer, text, target_indices, max_tokens_to_show=20, save_path=None):
    """
    Visualize attention weights before and after perturbation
    """
    print("🔍 Looking for custom attention layers...")
    
    # Get the custom attention layer
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
        print("❌ No CustomLlamaAttention layers found!")
        return
        
    if custom_layer.original_attention is None:
        print("❌ Found CustomLlamaAttention but original_attention is None!")
        print(f"   Target indices were: {custom_layer.target_indices}")
        return
    
    print(f"✅ Found attention weights! Shape: {custom_layer.original_attention.shape}")
    
    # Rest of your existing visualization code...
    # Get tokens for visualization
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens_to_show:
        tokens = tokens[:max_tokens_to_show]
    
    # Get attention weights (batch=0, head=0 for simplicity)
    orig_attn = custom_layer.original_attention[0, 0].cpu().numpy()
    pert_attn = custom_layer.perturbed_attention[0, 0].cpu().numpy()
    
    # Limit to max_tokens_to_show
    orig_attn = orig_attn[:max_tokens_to_show, :max_tokens_to_show]
    pert_attn = pert_attn[:max_tokens_to_show, :max_tokens_to_show]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original attention heatmap
    sns.heatmap(orig_attn, annot=False, cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Original Attention Weights')
    axes[0,0].set_xlabel('Key Tokens')
    axes[0,0].set_ylabel('Query Tokens')
    
    # Perturbed attention heatmap  
    sns.heatmap(pert_attn, annot=False, cmap='Reds', ax=axes[0,1])
    axes[0,1].set_title('Perturbed Attention Weights')
    axes[0,1].set_xlabel('Key Tokens')
    axes[0,1].set_ylabel('Query Tokens')
    
    # Attention to target tokens (column-wise average)
    orig_target_attn = [orig_attn[:, i].mean() if i < orig_attn.shape[1] else 0 for i in range(len(tokens))]
    pert_target_attn = [pert_attn[:, i].mean() if i < pert_attn.shape[1] else 0 for i in range(len(tokens))]
    
    axes[1,0].bar(range(len(tokens)), orig_target_attn, alpha=0.7, color='blue', label='Original')
    axes[1,0].bar(range(len(tokens)), pert_target_attn, alpha=0.7, color='red', label='Perturbed')
    axes[1,0].set_title('Average Attention to Each Token')
    axes[1,0].set_xlabel('Token Position')
    axes[1,0].set_ylabel('Average Attention Weight')
    axes[1,0].legend()
    
    # Highlight perturbed tokens
    for idx in target_indices:
        if idx < len(tokens):
            axes[1,0].axvline(x=idx, color='red', linestyle='--', alpha=0.5)
    
    # Token labels
    token_labels = [f"{i}: {token.replace('▁', '')}" for i, token in enumerate(tokens)]
    axes[1,1].axis('off')
    axes[1,1].text(0.1, 0.9, "Tokens:", fontsize=12, fontweight='bold', transform=axes[1,1].transAxes)
    
    for i, label in enumerate(token_labels):
        color = 'red' if i in target_indices else 'black'
        weight = 'bold' if i in target_indices else 'normal'
        axes[1,1].text(0.1, 0.85 - i*0.04, label, fontsize=10, color=color, 
                      weight=weight, transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to: {save_path}")
    else:
        plt.savefig('attention_visualization.png', dpi=300, bbox_inches='tight')
        print("Attention visualization saved to: attention_visualization.png")
    
    plt.show()
    
    # Print numerical comparison
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

def replace_llama_attention(model, target_indices, last_n_layers=1):
    device = next(model.parameters()).device
    total_layers = len(model.model.layers)
    
    print(f"🔄 Replacing attention in last {last_n_layers} layer(s) out of {total_layers}")
    print(f"🎯 Target indices: {target_indices}")
    
    for i in range(total_layers):
        if i >= total_layers - last_n_layers:
            print(f"   Replacing layer {i}...")
            
            # Store the original layer info
            original_layer = model.model.layers[i].self_attn
            print(f"   Original layer type: {type(original_layer).__name__}")
            
            # Create custom attention
            custom_attn = CustomLlamaAttention(model.config, i, target_indices)
            
            # Load state dict
            try:
                custom_attn.load_state_dict(original_layer.state_dict(), strict=False)
                print(f"   ✅ Loaded state dict successfully")
            except Exception as e:
                print(f"   ❌ Error loading state dict: {e}")
            
            # Move to device and set dtype
            custom_attn = custom_attn.to(device).half()
            
            # Replace the layer
            model.model.layers[i].self_attn = custom_attn
            
            # Verify replacement
            replaced_layer = model.model.layers[i].self_attn
            print(f"   New layer type: {type(replaced_layer).__name__}")
            print(f"   Target indices set: {replaced_layer.target_indices}")
            
    print("✅ Attention layer replacement complete")

# def run_with_attention_perturbation(model, tokenizer, text, target_tokens, visualize=False, save_path=None):
#     model.eval()
#     model = model.half()
    
#     messages = [
#         {"role": "system", "content": "You are a factual claim evaluator. Respond with TRUE or FALSE."},
#         {"role": "user", "content": f'Claim: "{text}"\n\nAnswer: TRUE or FALSE'}
#     ]
       
#     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     for key in inputs:
#         if inputs[key].dtype == torch.float:
#             inputs[key] = inputs[key].to(dtype=torch.float16)
    
#     token_ids = inputs["input_ids"][0]
#     print(f"input_ids device: {inputs['input_ids'].device}")
#     print(f"model device: {next(model.parameters()).device}")
    
#     target_indices = get_token_indices(tokenizer, text, target_tokens)
    
#     if not target_indices:
#         print("No matching token indices found for rationale tokens.")
#         return "UNKNOWN", 0.5
    
#     replace_llama_attention(model, target_indices, last_n_layers=1)
    
#     with torch.no_grad():
#         # FORCE output_attentions=True during generation
#         outputs = model.generate(
#             **inputs, 
#             max_new_tokens=10,
#             output_attentions=True,  # ← ADD THIS
#             return_dict_in_generate=True  # ← ADD THIS TO GET ATTENTION OUTPUTS
#         )
    
#     # Handle the different output format when return_dict_in_generate=True
#     if hasattr(outputs, 'sequences'):
#         generated_sequences = outputs.sequences
#     else:
#         generated_sequences = outputs
    
#     output_text = tokenizer.decode(generated_sequences[0], skip_special_tokens=True).strip()
#     print(f"Perturbed output: {output_text}")
    
#     generated_tokens = generated_sequences[0][inputs['input_ids'].shape[1]:]
#     output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
#     print(f"Again, Generated tokens only: {output_text}")
    
#     # Visualize attention if requested
#     if visualize:
#         visualize_attention_weights(model, tokenizer, text, target_indices, save_path=save_path)
    
#     label = "TRUE" if "TRUE" in output_text.upper() else "FALSE" if "FALSE" in output_text.upper() else "UNKNOWN"
#     match = re.search(r"[Cc]onfidence\s*[:=]?\s*(\d{1,3})", output_text)
#     confidence = float(match.group(1)) / 100 if match else 0.5
    
#     print(f"Perturbed label: {label} | confidence: {confidence:.2f}")
#     return label, confidence


def run_with_attention_perturbation(model, tokenizer, text, target_tokens, visualize=False, save_path=None):
    model.eval()
    model = model.half()

    # Prepare prompt
    messages = [
        {"role": "system", "content": "You are a factual claim evaluator. Respond with TRUE or FALSE."},
        {"role": "user", "content": f'Claim: "{text}"\n\nAnswer: TRUE or FALSE'}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    # Ensure half precision
    for k in inputs:
        if inputs[k].dtype == torch.float:
            inputs[k] = inputs[k].to(dtype=torch.float16)
    
    print(f"input_ids device: {inputs['input_ids'].device}")
    print(f"model device: {next(model.parameters()).device}")
    
    # Get token indices of rationale
    target_indices = get_token_indices(tokenizer, text, target_tokens)
    if not target_indices:
        print("No matching token indices found for rationale tokens.")
        return "UNKNOWN", 0.5
    
    # Inject perturbed attention
    replace_llama_attention(model, target_indices, last_n_layers=1)

    # Manual forward pass (no generate())
    with torch.no_grad():
        print("⚙️ Calling model with output_attentions=True")
        outputs = model(**inputs, output_attentions=True)

    logits = outputs.logits  # Shape: (1, seq_len, vocab_size)
    last_logits = logits[0, -1]  # Last token logits

    # Decode prediction
    probs = torch.nn.functional.softmax(last_logits, dim=-1)
    true_id = tokenizer.convert_tokens_to_ids("true")
    false_id = tokenizer.convert_tokens_to_ids("false")

    true_conf = probs[true_id].item() if true_id in tokenizer.get_vocab().values() else 0.0
    false_conf = probs[false_id].item() if false_id in tokenizer.get_vocab().values() else 0.0

    label = "TRUE" if true_conf > false_conf else "FALSE"
    confidence = max(true_conf, false_conf)

    print(f"Manual prediction: {label} | TRUE_conf={true_conf:.4f}, FALSE_conf={false_conf:.4f}")

    # Optional attention visualization
    if visualize:
        visualize_attention_weights(model, tokenizer, text, target_indices, save_path=save_path)

    return label, confidence