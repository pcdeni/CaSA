"""
Generate publication-quality figures for the PIM-LLM paper.
Outputs: fig1_architecture.png, fig2_protocol.png, fig3_bottleneck.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

DARK_BLUE = '#1a365d'
MED_BLUE = '#2b6cb0'
LIGHT_BLUE = '#bee3f8'
DARK_GREEN = '#22543d'
MED_GREEN = '#38a169'
LIGHT_GREEN = '#c6f6d5'
DARK_RED = '#742a2a'
MED_RED = '#e53e3e'
LIGHT_RED = '#fed7d7'
DARK_ORANGE = '#7b341e'
MED_ORANGE = '#dd6b20'
LIGHT_ORANGE = '#feebc8'
GRAY = '#718096'
LIGHT_GRAY = '#edf2f7'
DARK_GRAY = '#2d3748'
PURPLE = '#6b46c1'
LIGHT_PURPLE = '#e9d8fd'


# =====================================================================
# FIGURE 1: System Architecture Block Diagram
# =====================================================================
def draw_figure1():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')

    def box(x, y, w, h, color, edgecolor, label, fontsize=9, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor=edgecolor, linewidth=1.5)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=DARK_GRAY,
                linespacing=1.4)

    def arrow(x1, y1, x2, y2, color=GRAY, style='->', lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw))

    # --- Host PC ---
    box(0.3, 5.0, 2.0, 1.2, LIGHT_GRAY, GRAY, 'Host PC\n(CPU + System RAM)', fontsize=9, bold=True)

    # --- PCIe arrow ---
    arrow(2.3, 5.6, 3.2, 5.6, color=GRAY)
    ax.text(2.75, 5.8, 'PCIe x16', ha='center', va='bottom', fontsize=7, color=GRAY)

    # --- FPGA Controller (big box) ---
    fpga_rect = FancyBboxPatch((3.2, 3.8), 3.6, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#ebf8ff', edgecolor=MED_BLUE, linewidth=2.0)
    ax.add_patch(fpga_rect)
    ax.text(5.0, 6.15, 'FPGA Controller (Alveo U200)', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=DARK_BLUE)

    # Sub-blocks inside FPGA
    box(3.4, 5.3, 1.5, 0.7, LIGHT_BLUE, MED_BLUE, 'DRAM Bender\nCommand Engine', fontsize=7)
    box(5.1, 5.3, 1.5, 0.7, LIGHT_BLUE, MED_BLUE, 'Popcount\nAccumulator', fontsize=7)
    box(3.4, 4.2, 1.5, 0.7, LIGHT_GREEN, MED_GREEN, 'Non-linear Ops\nRMSNorm · SiLU · Softmax', fontsize=6.5)
    box(5.1, 4.2, 1.5, 0.7, LIGHT_PURPLE, PURPLE, 'KV-Cache\n(URAM, 65 MB)', fontsize=7)

    # Activation quantizer
    box(3.65, 3.9, 2.7, 0.2, LIGHT_ORANGE, MED_ORANGE, 'Activation Quantizer (AbsMean → 8-bit)', fontsize=6.5)

    # --- DDR4 Bus arrow (wide, labeled) ---
    # Draw a thick double-headed arrow
    bus_y = 3.3
    ax.annotate('', xy=(5.0, 2.8), xytext=(5.0, 3.8),
                arrowprops=dict(arrowstyle='<->', color=MED_RED, lw=2.5))
    # Bus label with box
    bus_box = FancyBboxPatch((3.5, 3.1), 3.0, 0.45, boxstyle="round,pad=0.05",
                              facecolor=LIGHT_RED, edgecolor=MED_RED, linewidth=1.2,
                              linestyle='--')
    ax.add_patch(bus_box)
    ax.text(5.0, 3.32, 'DDR4 Bus  ·  19.2 GB/s  ·  64-bit', ha='center', va='center',
            fontsize=8, fontweight='bold', color=DARK_RED)

    # Left/right labels on arrows
    ax.text(4.2, 3.55, 'activations ↓', ha='center', va='center', fontsize=6, color=MED_RED)
    ax.text(5.8, 3.55, '↑ AND results', ha='center', va='center', fontsize=6, color=MED_RED)

    # --- DRAM DIMM (big box) ---
    dimm_rect = FancyBboxPatch((2.5, 0.3), 5.0, 2.4, boxstyle="round,pad=0.1",
                                facecolor='#fffff0', edgecolor=MED_ORANGE, linewidth=2.0)
    ax.add_patch(dimm_rect)
    ax.text(5.0, 2.55, 'DDR4 DIMM  (SK Hynix HMA81GU6, 8 GB, C-die)', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color=DARK_ORANGE)

    # Sub-blocks inside DIMM
    box(2.7, 1.6, 2.0, 0.7, '#fefcbf', MED_ORANGE, 'Ternary Weights\n(W_pos, W_neg) row pairs\n131,640 rows', fontsize=6.5)
    box(5.0, 1.6, 2.2, 0.7, '#fefcbf', MED_ORANGE, 'Scratch Rows\nActivation bit-planes\n1,680 rows', fontsize=6.5)

    # Charge-sharing AND box (highlighted)
    and_box = FancyBboxPatch((3.2, 0.45), 3.6, 0.8, boxstyle="round,pad=0.08",
                              facecolor='#fef3c7', edgecolor=MED_RED, linewidth=2.0)
    ax.add_patch(and_box)
    ax.text(5.0, 0.85, '** Charge-Sharing AND **', ha='center', va='center',
            fontsize=9, fontweight='bold', color=DARK_RED)
    ax.text(5.0, 0.58, '65,536-bit parallel  ·  62 ns  ·  zero energy cost',
            ha='center', va='center', fontsize=7, color=DARK_RED)

    # --- Right side: data flow summary ---
    ax.text(8.2, 5.8, 'Data Flow (per bit-plane):', ha='left', va='center',
            fontsize=8, fontweight='bold', color=DARK_GRAY)
    steps = [
        '1. FPGA writes activation\n   bit-plane to scratch row',
        '2. DRAM: doubleACT\n   (weight AND scratch)',
        '3. FPGA reads AND result\n   (8,000 bytes)',
        '4. FPGA: popcount +\n   shift-accumulate',
    ]
    for i, step in enumerate(steps):
        y = 5.2 - i * 0.85
        ax.text(8.2, y, step, ha='left', va='center', fontsize=6.5, color=DARK_GRAY,
                linespacing=1.3, family='monospace')

    ax.text(8.2, 1.8, 'x16 per layer\n(8 bit-planes x 2 halves)\nx30 layers per token',
            ha='left', va='center', fontsize=7, color=MED_RED, fontweight='bold',
            linespacing=1.3)

    ax.text(5.0, -0.05, 'Figure 1: PIM-LLM System Architecture',
            ha='center', va='top', fontsize=10, fontweight='bold', color=DARK_GRAY)

    fig.tight_layout()
    fig.savefig('C:/Users/Udja/Documents/Deni/PIM/figures/fig1_architecture.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved fig1_architecture.png")


# =====================================================================
# FIGURE 2: Activation-Sacrificial Protocol Comparison
# =====================================================================
def draw_figure2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 5.5)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')

    def step_box(ax, x, y, w, h, color, edgecolor, label, fontsize=8, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                              facecolor=color, edgecolor=edgecolor, linewidth=1.5)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=DARK_GRAY,
                linespacing=1.3)

    def darrow(ax, x, y1, y2, color=GRAY):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # ---- Panel (a): Standard RowCopy-based PIM ----
    ax1.text(2.75, 7.7, '(a) Standard RowCopy-Based PIM', ha='center', va='center',
             fontsize=11, fontweight='bold', color=DARK_GRAY)

    # Step 1: ACT weight row
    step_box(ax1, 0.75, 6.5, 4.0, 0.8, LIGHT_BLUE, MED_BLUE,
             'ACT weight row\n(load weight into row buffer)', fontsize=8)

    darrow(ax1, 2.75, 6.5, 6.0)

    # Step 2: RowCopy (THE PROBLEM)
    step_box(ax1, 0.75, 4.8, 4.0, 1.0, LIGHT_RED, MED_RED,
             'RowCopy activation → scratch row\nBER = 16.3%  [FAIL]', fontsize=9, bold=True)
    # X mark
    ax1.text(5.0, 5.3, 'X', ha='center', va='center', fontsize=24, color=MED_RED, fontweight='bold')

    darrow(ax1, 2.75, 4.8, 4.3)

    # Step 3: AND
    step_box(ax1, 0.75, 3.2, 4.0, 0.8, LIGHT_GREEN, MED_GREEN,
             'doubleACT (weight AND scratch)\nCharge-sharing AND', fontsize=8)

    darrow(ax1, 2.75, 3.2, 2.7)

    # Step 4: READ
    step_box(ax1, 0.75, 1.7, 4.0, 0.8, LIGHT_BLUE, MED_BLUE,
             'READ result through DDR4 bus\n(8,000 bytes, 459 ns)', fontsize=8)

    # Error callout
    ax1.text(2.75, 0.8, 'Effective BER: 16.3%\n(dominated by RowCopy)',
             ha='center', va='center', fontsize=9, fontweight='bold', color=MED_RED,
             bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT_RED, edgecolor=MED_RED, linewidth=1.5))

    # Timeline arrow on left
    ax1.annotate('', xy=(0.3, 1.7), xytext=(0.3, 7.3),
                 arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.0))
    ax1.text(0.3, 4.5, 'time', ha='center', va='center', fontsize=7, color=GRAY, rotation=90)

    # ---- Panel (b): Our Activation-Sacrificial Protocol ----
    ax2.text(2.75, 7.7, '(b) Activation-Sacrificial Protocol (Ours)', ha='center', va='center',
             fontsize=11, fontweight='bold', color=DARK_GREEN)

    # Step 1: WRITE activation
    step_box(ax2, 0.75, 6.5, 4.0, 0.8, LIGHT_GREEN, MED_GREEN,
             'WRITE activation bit-plane → scratch\nStandard DDR4 write (460 ns)', fontsize=8)

    darrow(ax2, 2.75, 6.5, 6.0)

    # Step 2: doubleACT (AND)
    step_box(ax2, 0.75, 4.8, 4.0, 1.0, '#c6f6d5', DARK_GREEN,
             'doubleACT (weight AND scratch)\nCharge-sharing AND (62 ns)\nBER < 3.8e-8  [PASS]', fontsize=8.5, bold=True)
    # Checkmark
    ax2.text(5.0, 5.25, 'OK', ha='center', va='center', fontsize=18, color=MED_GREEN, fontweight='bold')

    darrow(ax2, 2.75, 4.8, 4.3)

    # Step 3: READ
    step_box(ax2, 0.75, 3.2, 4.0, 0.8, LIGHT_GREEN, MED_GREEN,
             'READ result through DDR4 bus\n(8,000 bytes, 459 ns)', fontsize=8)

    # Note: activation destroyed
    ax2.text(2.75, 2.5, 'Scratch row now contains AND result\n(activation is sacrificed — acceptable\nbecause activations are ephemeral)',
             ha='center', va='center', fontsize=7.5, color=DARK_GREEN, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0fff4', edgecolor=MED_GREEN,
                       linewidth=0.8, linestyle='--'))

    # Success callout
    ax2.text(2.75, 0.8, 'Effective BER: < 3.8e-8\n(4 orders of magnitude better)',
             ha='center', va='center', fontsize=9, fontweight='bold', color=DARK_GREEN,
             bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT_GREEN, edgecolor=DARK_GREEN, linewidth=1.5))

    # Timeline arrow on left
    ax2.annotate('', xy=(0.3, 3.2), xytext=(0.3, 7.3),
                 arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.0))
    ax2.text(0.3, 5.2, 'time', ha='center', va='center', fontsize=7, color=GRAY, rotation=90)

    # Key insight at bottom spanning both panels
    fig.text(0.5, 0.02,
             'Key insight: RowCopy is eliminated because activations are written via the standard DDR4 bus.\n'
             'Weights are permanent (stored in DRAM); activations are ephemeral (re-sent each bit-plane).',
             ha='center', va='bottom', fontsize=9, color=DARK_GRAY, style='italic')

    fig.suptitle('Figure 2: Activation-Sacrificial Protocol Comparison',
                 fontsize=12, fontweight='bold', color=DARK_GRAY, y=0.98)

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig('C:/Users/Udja/Documents/Deni/PIM/figures/fig2_protocol.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved fig2_protocol.png")


# =====================================================================
# FIGURE 3: Inference Time Breakdown (Stacked Bar Chart)
# =====================================================================
def draw_figure3():
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1]})

    def draw_bar(ax, categories, times_ms, colors, title, insight):
        total = sum(times_ms)
        percentages = [t/total*100 for t in times_ms]

        left = 0
        for i, (t, pct, color) in enumerate(zip(times_ms, percentages, colors)):
            if t > 0:
                bar = ax.barh(0, t, left=left, height=0.6, color=color, edgecolor='white', linewidth=2)
                cx = left + t/2
                if t > total * 0.08:
                    ax.text(cx, 0, f'{t:.0f} ms\n({pct:.0f}%)', ha='center', va='center',
                            fontsize=9, fontweight='bold', color='white')
                left += t

        # Category labels below
        left = 0
        for i, (t, cat) in enumerate(zip(times_ms, categories)):
            if t > 0:
                cx = left + t/2
                ax.text(cx, -0.55, cat, ha='center', va='top', fontsize=7, color=DARK_GRAY)
            left += t

        toks = 1000.0 / total if total > 0 else 0
        ax.text(total + 5, 0, f'Total: {total:.0f} ms\n= {toks:.1f} tok/s', ha='left', va='center',
                fontsize=9, fontweight='bold', color=DARK_GRAY)

        ax.text(total/2, -1.2, insight, ha='center', va='top', fontsize=7.5,
                color=DARK_GRAY, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT_GRAY, edgecolor=GRAY, linewidth=0.8))

        ax.set_xlim(-10, total + 150)
        ax.set_ylim(-1.8, 0.8)
        ax.set_yticks([])
        ax.set_xlabel('Time (ms)', fontsize=9, color=DARK_GRAY)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_title(title, fontsize=10, fontweight='bold', color=DARK_GRAY, pad=8)

    # Panel A: No pipelining (DRAM-bound) — 1 DIMM, ternary
    cats_a = ['Bus write\n(activation)', 'MAJ3 AND\n(DRAM-internal)', 'RowCopy\n(weight reload)', 'FPGA']
    times_a = [0.1, 206, 411, 6.5]  # From simulator: ternary, 1 DIMM, no pipeline
    colors_a = [MED_ORANGE, MED_GREEN, PURPLE, GRAY]
    draw_bar(axes[0], cats_a, times_a, colors_a,
             '(a) Without pipelining: DRAM-bound (ternary, 1 DIMM)',
             'RowCopy (67%) dominates — weight must be reloaded after each MAJ3.\n'
             'Bus is idle. DRAM internal ops are the bottleneck.')

    # Panel B: With 4-bank pipelining (bus-bound) — 1 DIMM, ternary
    cats_b = ['Bus write\n(activation)', 'Bus read\n(AND result)', 'FPGA']
    times_b = [0.1, 289, 6.5]  # From simulator: ternary, 1 DIMM, 4-bank pipeline
    colors_b = [MED_ORANGE, MED_BLUE, GRAY]
    draw_bar(axes[1], cats_b, times_b, colors_b,
             '(b) With 4-bank pipelining: bus-bound (ternary, 1 DIMM)',
             'RowCopy + MAJ3 run in parallel with bus transfers → hidden.\n'
             'Bus read (98%) is the sole bottleneck. In-DRAM popcount eliminates it.')

    fig.suptitle('Figure 3: Per-Token Inference Time Breakdown (Corrected — MAJ3 + RowCopy Protocol)',
                 fontsize=11, fontweight='bold', color=DARK_GRAY, y=1.02)
    fig.tight_layout()
    fig.savefig('C:/Users/Udja/Documents/Deni/PIM/figures/fig3_bottleneck.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved fig3_bottleneck.png")


# =====================================================================
# FIGURE 4 (bonus): Throughput Scaling Roadmap
# =====================================================================
def draw_figure4():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    configs = [
        'Baseline\n(1D, DDR4, 8-bit)',
        '+ 4-bit\nactivations',
        '+ 4 DIMMs',
        '+ In-DRAM\npopcount',
        '+ DDR5-4800',
    ]
    tok_s = [1.8, 3.5, 14, 31, 60]
    colors_bar = [GRAY, MED_ORANGE, MED_BLUE, MED_GREEN, PURPLE]
    changes = ['None', 'Software\nonly', '+ 3 DIMMs\n(~$60)', '~2K gates/bank\n(~$0.10/DIMM)', 'Next-gen\nmemory']

    x = np.arange(len(configs))
    bars = ax.bar(x, tok_s, width=0.6, color=colors_bar, edgecolor='white', linewidth=2)

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, tok_s)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                f'{val} tok/s', ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=colors_bar[i])

    # Changes required labels below
    for i, change in enumerate(changes):
        ax.text(i, -6, change, ha='center', va='top', fontsize=7.5, color=DARK_GRAY)

    ax.text(-0.7, -5.5, 'Changes\nrequired:', ha='right', va='top', fontsize=7.5,
            fontweight='bold', color=DARK_GRAY)

    # Reference lines
    ax.axhline(y=5.9, color=MED_RED, linewidth=1.5, linestyle='--', alpha=0.7)
    ax.text(4.7, 6.5, 'BitNet.cpp (CPU): 5.9 tok/s', ha='right', va='bottom',
            fontsize=8, color=MED_RED, fontweight='bold')

    ax.axhline(y=25, color=PURPLE, linewidth=1.0, linestyle=':', alpha=0.5)
    ax.text(4.7, 25.8, 'TeLLMe v2 (FPGA): 25 tok/s', ha='right', va='bottom',
            fontsize=7, color=PURPLE)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8)
    ax.set_ylabel('Throughput (tokens/second)', fontsize=10, color=DARK_GRAY)
    ax.set_ylim(0, 70)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

    ax.set_title('Figure 4: Cumulative Throughput Scaling Roadmap (Table 6.6)',
                 fontsize=11, fontweight='bold', color=DARK_GRAY, pad=15)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig('C:/Users/Udja/Documents/Deni/PIM/figures/fig4_scaling.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved fig4_scaling.png")


if __name__ == '__main__':
    draw_figure1()
    draw_figure2()
    draw_figure3()
    draw_figure4()
    print("\nAll figures generated successfully.")
