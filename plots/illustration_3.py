# render_prompt_templates.py
# Renders the CoT prompt templates as neat, academic cards (PNG + SVG).
# No external deps beyond matplotlib. Produces:
#  - prompt_template_cv.png / .svg
#  - prompt_template_qa.png / .svg
#  - prompt_templates_combined.png / .svg

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import textwrap
import re

# ----------------------------
# Templates (as given; unchanged)
# ----------------------------
template_cv = (
    "Claim: {claim}\n"
    "Q: Is the above claim true? Let's think step-by-step to ensure each part of our reasoning connects clearly to the final answer. \n"
    "Generate your explanation slightly elaborately! Conclude with a single-sentence verdict beginning with 'Conclusion:'.\n"
    "A:"
)

template_qa = (
    "Q: Why does ice float on water?\n"
    "A:\n"
    "1. Water molecules form hydrogen bonds.\n"
    "2. As water freezes, these molecules arrange into a crystalline structure.\n"
    "3. This structure takes up more space and reduces density.\n"
    "4. Less dense substances float on denser liquids.\n"
    "Conclusion: Ice floats because its crystal structure makes it less dense than liquid water.\n"
    "\n"
    "Q: What causes thunder?\n"
    "A:\n"
    "1. Lightning rapidly heats the surrounding air.\n"
    "2. The sudden heat causes the air to expand explosively.\n"
    "3. This rapid expansion creates a shockwave.\n"
    "4. That shockwave is what we hear as thunder.\n"
    "Conclusion: Thunder is the sound of air expanding rapidly due to lightning.\n"
    "\n"
    "Q: {question}\n"
    "Let's think step-by-step to ensure each part of our reasoning connects clearly to the final answer.\n"
    "Generate your answer slightly elaborately!\n"
    "A:"
)

# ----------------------------
# Utilities
# ----------------------------
def wrap_code_block(src: str, width: int = 88) -> str:
    """
    Soft-wrap each line to a target width while preserving indentation.
    Intended for monospaced display. Avoids merging separate lines.
    """
    wrapped_lines = []
    for line in src.splitlines():
        if not line.strip():
            wrapped_lines.append("")  # keep blank lines
            continue
        # Preserve leading indentation (spaces only).
        m = re.match(r"^(\s*)", line)
        indent = m.group(1) if m else ""
        body = line[len(indent):]
        # Use textwrap with preserved indent.
        wrapped = textwrap.fill(
            body,
            width=max(10, width - len(indent)),
            initial_indent=indent,
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_lines.append(wrapped)
    return "\n".join(wrapped_lines)

def draw_card(ax, title: str, content: str,
              face="#f8fafc", edge="#cbd5e1",
              head_face="#0f172a", head_text="#ffffff",
              mono_font="DejaVu Sans Mono"):
    """
    Draw a rounded 'card' with a title header and monospaced content.
    Coordinates in axis fraction [0,1].
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Outer rounded panel
    outer = FancyBboxPatch(
        (0.03, 0.03), 0.94, 0.94,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2, edgecolor=edge, facecolor=face
    )
    ax.add_patch(outer)

    # Header bar
    header = FancyBboxPatch(
        (0.03, 0.88), 0.94, 0.09,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0, edgecolor="none", facecolor=head_face
    )
    ax.add_patch(header)
    ax.text(0.5, 0.925, title, ha="center", va="center",
            fontsize=14, color=head_text, fontweight="bold")

    # Content area (monospaced)
    ax.text(0.05, 0.84, content,
            ha="left", va="top", fontsize=10.5,
            family=mono_font)

def save_fig(fig, basename: str):
    fig.tight_layout()
    fig.savefig(f"{basename}.png", dpi=300)
    fig.savefig(f"{basename}.svg")

# ----------------------------
# Render individual templates
# ----------------------------
def render_cv():
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    content = wrap_code_block(template_cv, width=92)
    draw_card(ax, "Claim Verification Prompt Template", content)
    save_fig(fig, "prompt_template_cv")
    plt.close(fig)

def render_qa():
    # QA template is longer; give more vertical space
    fig, ax = plt.subplots(figsize=(8.5, 12.0))
    content = wrap_code_block(template_qa, width=92)
    draw_card(ax, "Question Answering Prompt Template (Few-shot)", content)
    save_fig(fig, "prompt_template_qa")
    plt.close(fig)

# ----------------------------
# Render combined (stacked)
# ----------------------------
def render_combined():
    fig = plt.figure(figsize=(9.0, 16.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.8], hspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    cv_content = wrap_code_block(template_cv, width=92)
    draw_card(ax1, "Claim Verification Prompt Template", cv_content)

    ax2 = fig.add_subplot(gs[1, 0])
    qa_content = wrap_code_block(template_qa, width=92)
    draw_card(ax2, "Question Answering Prompt Template (Few-shot)", qa_content)

    save_fig(fig, "prompt_templates_combined")
    plt.close(fig)

if __name__ == "__main__":
    render_cv()
    render_qa()
    render_combined()
    print("Saved: prompt_template_cv.(png|svg), prompt_template_qa.(png|svg), prompt_templates_combined.(png|svg)")
