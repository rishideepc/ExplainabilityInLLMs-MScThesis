# rationale_token_alignment_tabular.py
# Two-column layout for the "Rationale Token Selection & Alignment" diagram:
# Left column = row labels; Right column = illustration.
# Saves PNG + SVG.

import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# --- Example data ---
claim_text = 'The moon is made of cheese.'
subword_tokens = ['The', 'moon', 'is', 'made', 'of', 'che', 'ese']  # didactic tokenization
rationales = ['moon', 'cheese']

# --- Helpers ---
def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def align_rationale_to_tokens(rationale, tokens):
    """Find windows of consecutive subword tokens that concatenate to the rationale."""
    rnorm = normalize(rationale)
    tnorm = [normalize(t) for t in tokens]
    matches = []
    for i in range(len(tnorm)):
        concat = ''
        for j in range(i, len(tnorm)):
            concat += tnorm[j]
            if concat == rnorm:
                matches.append((i, j))
                break
            if len(concat) > len(rnorm):
                break
    return matches

# Map rationales → token indices
rationale_to_token_indices = {}
for r in rationales:
    windows = align_rationale_to_tokens(r, subword_tokens)
    if not windows:
        idxs = [i for i, t in enumerate(subword_tokens) if normalize(t) == normalize(r)]
    else:
        idxs = []
        for (i, j) in windows:
            idxs.extend(range(i, j + 1))
    rationale_to_token_indices[r] = sorted(set(idxs))

# --- Figure & layout ---
fig = plt.figure(figsize=(15, 6.5))  # wider canvas
gs = GridSpec(1, 2, width_ratios=[1.0, 5.0], wspace=0.06, figure=fig)
axL = fig.add_subplot(gs[0, 0])  # labels column
axR = fig.add_subplot(gs[0, 1])  # illustration column

for ax in (axL, axR):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

# Global title on the whole figure
# fig.suptitle("Rationale Token Selection & Alignment", fontsize=18, fontweight='bold', y=0.97)
fig.text(0.5, 0.92, f'Claim: "{claim_text}"', ha='center', va='top', fontsize=18, fontweight='bold')

# Shared row y-positions
y_rationales = 0.72
y_tokens     = 0.42
y_indices    = 0.305

# ----- Left column: fixed row labels (no overlap possible) -----
axL.text(0.02, y_rationales, "Model-declared\nrationales →", ha='left', va='center',
         fontsize=17, fontstyle='italic')
axL.text(0.02, y_tokens, "Subword tokens →", ha='left', va='center',
         fontsize=17, fontstyle='italic')
axL.text(0.02, y_indices, "Indices →", ha='left', va='center',
         fontsize=16, color='dimgray')

# ----- Right column: illustration -----
# Layout for tokens inside the right axis
n_tokens = len(subword_tokens)
left_margin  = 0.06
right_margin = 0.03
box_h = 0.09
box_w = 0.085
avail_w = 1 - left_margin - right_margin
gap = (avail_w - n_tokens * box_w) / (n_tokens - 1)

# Compute token box positions
token_boxes = []
x = left_margin
for i, tok in enumerate(subword_tokens):
    token_boxes.append({'x0': x, 'y0': y_tokens - box_h / 2, 'w': box_w, 'h': box_h, 'text': tok, 'index': i})
    x += box_w + gap

# Draw token boxes and indices
for tb in token_boxes:
    axR.add_patch(Rectangle((tb['x0'], tb['y0']), tb['w'], tb['h'],
                             linewidth=1.5, edgecolor='black', facecolor='white'))
    axR.text(tb['x0'] + tb['w']/2, tb['y0'] + tb['h']/2, tb['text'],
             ha='center', va='center', fontsize=17)
    axR.text(tb['x0'] + tb['w']/2, y_indices, f"{tb['index']}",
             ha='center', va='center', fontsize=16, color='dimgray')

# Rationale boxes across the top
r_box_w = 0.16
r_box_h = 0.09
r_gap   = 0.06
total_r_width = len(rationales) * r_box_w + (len(rationales) - 1) * r_gap
r_start = 0.5 - total_r_width / 2

rationale_boxes = []
for i, r in enumerate(rationales):
    x0 = r_start + i * (r_box_w + r_gap)
    axR.add_patch(Rectangle((x0, y_rationales - r_box_h / 2), r_box_w, r_box_h,
                             linewidth=2, edgecolor='black', facecolor='#e6f2ff'))
    axR.text(x0 + r_box_w/2, y_rationales, r, ha='center', va='center',
             fontsize=17, fontweight='bold')
    rationale_boxes.append({
        'x_center': x0 + r_box_w/2,
        'y0': y_rationales - r_box_h / 2,
        'text': r
    })

# Highlights & arrows
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
rationale_colors = {r: color_cycle[i % len(color_cycle)] for i, r in enumerate(rationales)}

# Highlight aligned tokens
for r in rationales:
    for idx in rationale_to_token_indices[r]:
        tb = token_boxes[idx]
        axR.add_patch(Rectangle((tb['x0'], tb['y0']), tb['w'], tb['h'],
                                 linewidth=2, edgecolor=rationale_colors[r], facecolor='none'))

# Arrows
for rb in rationale_boxes:
    r = rb['text']
    for idx in rationale_to_token_indices[r]:
        tb = token_boxes[idx]
        axR.annotate('',
                     xy=(tb['x0'] + tb['w']/2, tb['y0'] + tb['h']),
                     xytext=(rb['x_center'], rb['y0']),
                     arrowprops=dict(arrowstyle='->', lw=2,
                                     color=rationale_colors[r],
                                     shrinkA=6, shrinkB=6,
                                     connectionstyle="arc3,rad=0.15"))

# Optional explainer box (kept inside right column)
# explain = ("Mapping example:\n"
#            "  • 'moon'   → token[1] ('moon')\n"
#            "  • 'cheese' → tokens[5:7] ('che','ese')\n"
#            "We align model-declared rationale words to subword indices.")
# axR.text(0.5, 0.10, explain, ha='center', va='center', fontsize=11, family='monospace',
#          bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7', edgecolor='lightgray'))

# Final layout & save
plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.06)
plt.savefig('rationale_token_alignment_tabular.png', dpi=220)
plt.savefig('rationale_token_alignment_tabular.svg')
print("Saved: rationale_token_alignment_tabular.png and .svg")
