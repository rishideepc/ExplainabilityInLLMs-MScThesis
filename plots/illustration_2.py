# dataflow_overview.py
# High-level dataflow diagram:
# Datasets -> (CoT | argLLM) Generators -> (Deductive | Argumentative) Metrics
# plus SHAKE (attention perturbation) -> Aggregation -> Reports
#
# Saves:
#   - dataflow_overview.png
#   - dataflow_overview.svg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Rectangle
import matplotlib as mpl
# ----------------------------
# Style
# ----------------------------
mpl.rcParams['font.size'] = 10
mpl.rcParams['figure.dpi'] = 200
COL_DATA      = "#e8f1ff"  # datasets
COL_GEN_COT   = "#e9fbe7"  # CoT generator
COL_GEN_ARG   = "#fff4e6"  # argLLM generator
COL_MET_COT   = "#dff7e0"  # CoT metrics
COL_MET_ARG   = "#ffedd6"  # argLLM metrics
COL_SHAKE     = "#f3e8ff"  # SHAKE block
COL_AGG       = "#eef2f7"  # aggregator
COL_OUT       = "#e6f7ff"  # reports
COL_TEXT_DARK = "#222222"
ARROW_COLOR_MAIN   = "#6b7280"  # neutral gray
ARROW_COLOR_SHAKE  = "#8b5cf6"  # purple-ish
ARROW_WIDTH        = 1.5
# ----------------------------
# Helpers
# ----------------------------
def round_box(ax, xy, w, h, text, fc="#ffffff", ec="#111111", fontsize=10,
              lw=1.6, radius=0.04, align="center", weight="normal"):
    """Draw a rounded rectangle with centered text (in axis fraction coords)."""
    (x, y) = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1.0
    )
    ax.add_patch(box)
    if align == "center":
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=COL_TEXT_DARK, fontweight=weight)
    elif align == "left":
        ax.text(x + 0.01, y + h/2, text, ha="left", va="center",
                fontsize=fontsize, color=COL_TEXT_DARK, fontweight=weight)
    return box
def arrow(ax, xy_from, xy_to, color=ARROW_COLOR_MAIN, lw=ARROW_WIDTH, curve=0.0):
    """Draw an arrow between two points (axis fraction coords)."""
    ax.annotate(
        "", xy=xy_to, xytext=xy_from,
        arrowprops=dict(arrowstyle="->", lw=lw, color=color,
                        connectionstyle=f"arc3,rad={curve}")
    )
def label(ax, x, y, txt, size=10, weight="normal", ha="center", va="center", color=COL_TEXT_DARK):
    ax.text(x, y, txt, fontsize=size, fontweight=weight, ha=ha, va=va, color=color)
# ----------------------------
# Figure & Axes
# ----------------------------
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
# Title & subtitle
label(ax, 0.5, 0.95, "Implementation Dataflow Overview", size=16, weight="bold")
label(ax, 0.5, 0.91,
      "Datasets → Explanation Generators → Paradigm-Specific Evaluations (+ SHAKE) → Aggregation → Reports",
      size=11)
# ----------------------------
# Layout coordinates (axis fraction) - Fixed spacing
# ----------------------------
# Column x-positions
x_data   = 0.05
x_gen    = 0.25
x_eval   = 0.50
x_shake  = 0.50  # SHAKE sits vertically with eval, separate lane
x_agg    = 0.75
x_out    = 0.88
# Row y-positions - Better vertical spacing
y_top    = 0.65
y_mid    = 0.40
y_low    = 0.15
# Common sizes - Better proportions
W_data   = 0.15; H_data   = 0.55
W_block  = 0.20; H_block  = 0.15
W_eval   = 0.18; H_eval   = 0.15
W_shake  = 0.18; H_shake  = 0.15
W_agg    = 0.10; H_agg    = 0.30
W_out    = 0.10; H_out    = 0.18
# ----------------------------
# Left column: Dataset stack
# ----------------------------
datasets_box = round_box(
    ax, (x_data, y_low), W_data, H_data,
    "",  # We'll add text manually for better control
    fc=COL_DATA, ec="#1f4acc", weight="bold"
)
# Dataset header and items
header_y = y_low + H_data - 0.08
label(ax, x_data + W_data/2, header_y, "Datasets", size=12, weight="bold", color="#1f4acc")
label(ax, x_data + W_data/2, header_y - 0.06, "Claim/QA corpora", size=10, color="#1f4acc")
ds_items = [
    "TruthfulClaim/TruthfulQA",
    "StrategyClaim/StrategyQA", 
    "MedClaim/MedQA",
    "CommonsenseClaim/\nCommonsenseQA",
]
for i, name in enumerate(ds_items):
    label(ax, x_data + 0.01, header_y - 0.12 - i*0.08, name, ha="left", size=9)
# ----------------------------
# Middle-left: Generators (two lanes)
# ----------------------------
cot_gen = round_box(
    ax, (x_gen, y_top), W_block, H_block,
    "CoT Generator\n(few-shot prompting, decoding)",
    fc=COL_GEN_COT, ec="#15803d", weight="bold", fontsize=10
)
arg_gen = round_box(
    ax, (x_gen, y_low), W_block, H_block,
    "argLLM Generator\n(argument mining, graphs)",
    fc=COL_GEN_ARG, ec="#b45309", weight="bold", fontsize=10
)
# Arrows from Datasets to Generators
arrow(ax, (x_data + W_data, y_low + H_data * 0.7), (x_gen, y_top + H_block/2), curve=0.1)
arrow(ax, (x_data + W_data, y_low + H_data * 0.3), (x_gen, y_low + H_block/2), curve=-0.1)
# ----------------------------
# Middle-right: Evaluators (two lanes)
# ----------------------------
cot_eval = round_box(
    ax, (x_eval, y_top), W_eval, H_eval,
    "Deductive Metrics:\nRedundancy, Weak/Strong\nRelevance",
    fc=COL_MET_COT, ec="#15803d", fontsize=9
)
arg_eval = round_box(
    ax, (x_eval, y_low), W_eval, H_eval,
    "Argumentative Metrics:\nCircularity, Acceptability,\nDialectical Faithfulness",
    fc=COL_MET_ARG, ec="#b45309", fontsize=9
)
# Arrows from Generators to their Evaluators
arrow(ax, (x_gen + W_block, y_top + H_block/2), (x_eval, y_top + H_eval/2))
arrow(ax, (x_gen + W_block, y_low + H_block/2), (x_eval, y_low + H_eval/2))
# ----------------------------
# SHAKE lane (faithfulness via attention perturbation)
# ----------------------------
shake = round_box(
    ax, (x_shake, y_mid - H_shake/2), W_shake, H_shake,
    "SHAKE: Attention Perturbation\n• Rationale alignment\n• Zero-out + renormalize\n• Label flips / Δconfidence",
    fc=COL_SHAKE, ec="#7c3aed", weight="bold", fontsize=9
)
# Arrow from Datasets to SHAKE
arrow(ax, (x_data + W_data, y_low + H_data/2), (x_shake, y_mid), color=ARROW_COLOR_SHAKE, curve=0.0)
# ----------------------------
# Aggregation & Reports
# ----------------------------
agg = round_box(
    ax, (x_agg, y_mid - H_agg/2), W_agg, H_agg,
    "Aggregation\n&\nAnalytics",
    fc=COL_AGG, ec="#334155", weight="bold", fontsize=10
)
reports = round_box(
    ax, (x_out, y_mid - H_out/2), W_out, H_out,
    "Reports\n(CSV, Plots,\nTables)",
    fc=COL_OUT, ec="#0369a1", weight="bold", fontsize=9
)
# Arrows from all evaluators + SHAKE -> Aggregation
arrow(ax, (x_eval + W_eval, y_top + H_eval/2), (x_agg, y_mid + H_agg/4))
arrow(ax, (x_eval + W_eval, y_low + H_eval/2), (x_agg, y_mid - H_agg/4))
arrow(ax, (x_shake + W_shake, y_mid), (x_agg, y_mid), color=ARROW_COLOR_SHAKE)
# Arrow Aggregation -> Reports
arrow(ax, (x_agg + W_agg, y_mid), (x_out, y_mid))
# # ----------------------------
# # Row labels (left margin)
# # ----------------------------
# label(ax, 0.01, y_top + H_block/2, "CoT branch", size=10, ha="left", color="#166534", weight="bold")
# label(ax, 0.01, y_low + H_block/2, "argLLM branch", size=10, ha="left", color="#92400e", weight="bold")
# label(ax, 0.01, y_mid, "Faithfulness (SHAKE)", size=10, ha="left", color="#6d28d9", weight="bold")
# Caption area
# label(ax, 0.5, 0.05,
#       "Overview: data from claim/QA corpora flows to generators, then to paradigm-specific evaluation suites "
#       "and the SHAKE faithfulness test. Metrics are aggregated into final analytical reports.",
#       size=9, ha="center", color="#4b5563")
plt.tight_layout()
plt.savefig("dataflow_overview.png", bbox_inches='tight', dpi=200)
plt.savefig("dataflow_overview.svg", bbox_inches='tight')
print("Saved: dataflow_overview.png and dataflow_overview.svg")