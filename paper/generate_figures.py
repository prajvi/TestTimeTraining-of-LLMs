"""
Generate publication-quality figures for NeurIPS 2026 E&D paper.
Run: python generate_figures.py
Outputs: fig1_gap.pdf, fig2_diagnostic.pdf, fig3_heatmap.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── NeurIPS style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette (colorblind-friendly)
C_EXPLICIT = '#4C72B0'   # blue
C_APPLIED  = '#DD8452'   # orange
C_FROZEN   = '#A8A8A8'   # grey
C_NTL      = '#C44E52'   # red
C_AR       = '#55A868'   # green
C_COMPOSED = '#8172B3'   # purple
C_BSTTT    = '#CCB974'   # gold


# ═══════════════════════════════════════════════════════════
# FIGURE 1: The Explicit-to-Applied Gap (Hero Figure)
# Shows the gap across models on SimpleToM (cleanest signal)
# ═══════════════════════════════════════════════════════════
def fig1_gap():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)

    models = ['Qwen-2.5-7B', 'Llama-3.1-8B', 'GPT-oss-20B']

    # SimpleToM per-question-type: [mental_state, behavior, judgment]
    # Frozen (before BSTTT)
    frozen = {
        'Qwen-2.5-7B':   [54.4, 0.0, 76.0],
        'Llama-3.1-8B':  [100.0, 26.7, 40.0],
        'GPT-oss-20B':   [100.0, 100.0, 0.0],
    }
    # After BSTTT-AR
    after = {
        'Qwen-2.5-7B':   [54.4, 54.4, 76.0],
        'Llama-3.1-8B':  [100.0, 43.3, 60.0],
        'GPT-oss-20B':   [100.0, 100.0, 0.0],
    }

    q_types = ['Mental-State\n(explicit)', 'Behavior\n(applied)', 'Judgment\n(applied)']
    x = np.arange(len(q_types))
    w = 0.32

    for i, (ax, model) in enumerate(zip(axes, models)):
        bars_f = ax.bar(x - w/2, frozen[model], w, color=C_FROZEN, 
                        edgecolor='white', linewidth=0.5, label='Frozen', zorder=3)
        bars_a = ax.bar(x + w/2, after[model], w, color=C_AR, 
                        edgecolor='white', linewidth=0.5, label='+ BSTTT-AR', zorder=3)

        # Annotate the dramatic changes
        for j in range(3):
            delta = after[model][j] - frozen[model][j]
            if abs(delta) > 1:
                y_pos = max(frozen[model][j], after[model][j]) + 3
                ax.annotate(f'+{delta:.0f}', xy=(x[j], y_pos),
                           fontsize=7, ha='center', fontweight='bold',
                           color='#2d6a2d')

        # Highlight the gap region for Qwen behavior
        if model == 'Qwen-2.5-7B':
            ax.annotate('', xy=(1 - w/2, 0), xytext=(1 - w/2, 54.4),
                        arrowprops=dict(arrowstyle='<->', color='#CC0000', lw=1.5))
            ax.text(1 - w/2 - 0.18, 27, 'GAP\n54 pp', fontsize=6.5, color='#CC0000',
                    fontweight='bold', ha='center')

        ax.set_xticks(x)
        ax.set_xticklabels(q_types)
        ax.set_title(model, fontweight='bold', pad=8)
        ax.set_ylim(0, 115)
        ax.yaxis.set_major_locator(plt.MultipleLocator(25))
        ax.axhline(y=50, color='#CCCCCC', linestyle=':', linewidth=0.5, zorder=1)
        ax.grid(axis='y', alpha=0.2, zorder=0)

        if i == 0:
            ax.set_ylabel('Accuracy (%)')

    axes[0].legend(loc='upper left', framealpha=0.9, edgecolor='#CCCCCC')

    fig.suptitle('The Explicit-to-Applied Gap on SimpleToM', 
                 fontweight='bold', fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig('fig1_gap.pdf', format='pdf')
    fig.savefig('fig1_gap.png', format='png', dpi=300)
    print('✓ fig1_gap.pdf')


# ═══════════════════════════════════════════════════════════
# FIGURE 2: AR vs NTL Diagnostic Contrast (Key Result)
# Paired lollipop chart showing AR improves, NTL doesn't
# ═══════════════════════════════════════════════════════════
def fig2_diagnostic():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    # Data: (model, benchmark, frozen, ntl, ar)
    data = [
        ('Llama',   'SimpleToM', 55.6, 55.6, 61.1),
        ('Llama',   'Hi-ToM',    41.4, 41.4, 54.4),
        ('Llama',   'OpenToM',   42.5, 42.7, 43.1),
        ('Qwen',    'SimpleToM', 41.7, 45.3, 61.6),
        ('Qwen',    'Hi-ToM',    40.7, 40.7, 40.7),
        ('Qwen',    'OpenToM',   40.5, 40.5, 40.5),
        ('GPT-oss', 'SimpleToM', 66.7, 66.7, 66.7),
        ('GPT-oss', 'Hi-ToM',    32.6, 31.6, 47.3),
        ('GPT-oss', 'OpenToM',   35.6, 35.1, 36.0),
    ]

    y_positions = np.arange(len(data))[::-1]
    labels = [f'{d[0]} · {d[1]}' for d in data]

    for i, (model, bench, frz, ntl, ar) in enumerate(data):
        y = y_positions[i]

        # Frozen baseline (grey dot)
        ax.plot(frz, y, 'o', color=C_FROZEN, markersize=6, zorder=5)

        # NTL arrow (red)
        delta_ntl = ntl - frz
        ax.annotate('', xy=(ntl, y + 0.15), xytext=(frz, y + 0.15),
                    arrowprops=dict(arrowstyle='->', color=C_NTL, lw=1.5))
        ax.plot(ntl, y + 0.15, 'o', color=C_NTL, markersize=4, zorder=5)

        # AR arrow (green)
        delta_ar = ar - frz
        ax.annotate('', xy=(ar, y - 0.15), xytext=(frz, y - 0.15),
                    arrowprops=dict(arrowstyle='->', color=C_AR, lw=2.0))
        ax.plot(ar, y - 0.15, 'D', color=C_AR, markersize=5, zorder=5)

        # Delta label for AR if meaningful
        if abs(delta_ar) > 2:
            ax.text(ar + 1.0, y - 0.15, f'+{delta_ar:.1f}',
                   fontsize=6.5, va='center', color='#2d6a2d', fontweight='bold')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('Accuracy (%)')
    ax.set_xlim(25, 75)
    ax.grid(axis='x', alpha=0.2)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_FROZEN,
                   markersize=7, label='Frozen'),
        mpatches.FancyArrowPatch((0,0), (1,0), color=C_NTL, 
                                  arrowstyle='->', mutation_scale=10),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_NTL,
                   markersize=5, label='NTL (generic)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=C_AR,
                   markersize=6, label='AR (ToM-aligned)'),
    ]
    # Manual legend
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_FROZEN, markersize=7, label='Frozen baseline'),
        Line2D([0], [0], marker='o', color=C_NTL, markerfacecolor=C_NTL, markersize=5, 
               linestyle='-', linewidth=1.5, label='+ NTL (generic)'),
        Line2D([0], [0], marker='D', color=C_AR, markerfacecolor=C_AR, markersize=6, 
               linestyle='-', linewidth=2, label='+ AR (ToM-aligned)'),
    ]
    ax.legend(handles=leg, loc='lower right', framealpha=0.95, edgecolor='#CCCCCC', fontsize=7.5)

    ax.set_title('Diagnostic Contrast: Only ToM-Aligned Adaptation Closes the Gap',
                fontweight='bold', fontsize=9.5, pad=10)

    plt.tight_layout()
    fig.savefig('fig2_diagnostic.pdf', format='pdf')
    fig.savefig('fig2_diagnostic.png', format='png', dpi=300)
    print('✓ fig2_diagnostic.pdf')


# ═══════════════════════════════════════════════════════════
# FIGURE 3: Heatmap of gains (Δ best vs frozen)
# Shows where BSTTT helps and where it doesn't
# ═══════════════════════════════════════════════════════════
def fig3_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5),
                              gridspec_kw={'width_ratios': [3, 3]})

    # (a) BSTTT-AR alone Δ vs Frozen
    models = ['Llama-8B', 'Qwen-7B', 'GPT-oss-20B']
    benchmarks = ['SimpleToM', 'Hi-ToM', 'OpenToM', 'DynToM']

    delta_ar = np.array([
        [5.5,  13.0,  0.6,  -0.8],   # Llama BSTTT-AR - Frozen
        [19.9, 0.0,   0.0,  np.nan],  # Qwen
        [0.0,  14.7,  0.4,  np.nan],  # GPT-oss
    ])

    delta_best = np.array([
        [27.7, 13.9, 7.5, 3.8],    # Llama best overall - Frozen
        [31.6, 13.0, 6.2, 2.5],    # Qwen
        [0.0,  14.7, 0.7, 0.5],    # GPT-oss
    ])

    for idx, (data, title) in enumerate([
        (delta_ar, '(a) BSTTT-AR alone (Δ vs. Frozen)'),
        (delta_best, '(b) Best overall (Δ vs. Frozen)')
    ]):
        ax = axes[idx]

        # Mask NaN
        masked = np.ma.masked_invalid(data)

        im = ax.imshow(masked, cmap='RdYlGn', aspect='auto',
                       vmin=-5, vmax=35, interpolation='nearest')

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(benchmarks)):
                val = data[i, j]
                if np.isnan(val):
                    ax.text(j, i, '—', ha='center', va='center',
                           fontsize=8, color='#888888')
                else:
                    color = 'white' if val > 20 or val < 0 else 'black'
                    sign = '+' if val > 0 else ''
                    ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                           fontsize=7.5, fontweight='bold', color=color)

        ax.set_xticks(range(len(benchmarks)))
        ax.set_xticklabels(benchmarks, rotation=30, ha='right', fontsize=7.5)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=7.5)
        ax.set_title(title, fontsize=9, fontweight='bold', pad=8)

        # Grid
        for edge in range(len(benchmarks) + 1):
            ax.axvline(edge - 0.5, color='white', linewidth=1.5)
        for edge in range(len(models) + 1):
            ax.axhline(edge - 0.5, color='white', linewidth=1.5)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, aspect=20)
    cbar.set_label('Δ accuracy (pp)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig('fig3_heatmap.pdf', format='pdf')
    fig.savefig('fig3_heatmap.png', format='png', dpi=300)
    print('✓ fig3_heatmap.pdf')


# ═══════════════════════════════════════════════════════════
# FIGURE 4 (BONUS): Composability diagram
# Shows Frozen → BSTTT → BSTTT+Prompt trajectory
# ═══════════════════════════════════════════════════════════
def fig4_composability():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # Data: (label, frozen, bsttt, bsttt+prompt)
    series = [
        ('Llama · SimpleToM',  55.6, 61.1, 83.3),
        ('Llama · Hi-ToM',     41.4, 54.4, 55.3),
        ('Llama · OpenToM',    42.5, 43.1, 50.0),
        ('Qwen · SimpleToM',   41.7, 61.6, 73.3),
        ('Qwen · Hi-ToM',      40.7, 40.7, 53.7),
        ('Qwen · OpenToM',     40.5, 40.5, 46.7),
        ('GPT · Hi-ToM',       32.6, 47.3, 46.5),
    ]

    x_stages = [0, 1, 2]
    stage_labels = ['Frozen', '+ BSTTT-AR', '+ BSTTT + Prompt']

    colors = ['#4C72B0', '#6B8FC2', '#8BADD4',   # Llama shades
              '#DD8452', '#E8A77A', '#F2CBA2',   # Qwen shades
              '#55A868']                          # GPT

    for i, (label, frz, bsttt, comp) in enumerate(series):
        vals = [frz, bsttt, comp]
        alpha = 0.85
        lw = 2.0 if 'SimpleToM' in label else 1.2
        ls = '-' if comp > bsttt else '--'

        ax.plot(x_stages, vals, 'o-', color=colors[i], linewidth=lw,
                markersize=5, alpha=alpha, label=label, linestyle=ls)

    ax.set_xticks(x_stages)
    ax.set_xticklabels(stage_labels, fontsize=8.5)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(-0.3, 2.5)
    ax.set_ylim(25, 90)
    ax.grid(axis='y', alpha=0.2)

    ax.legend(loc='upper left', fontsize=6.5, framealpha=0.95,
              edgecolor='#CCCCCC', ncol=2)

    ax.set_title('Composability: Weight + Input Interventions',
                fontweight='bold', fontsize=9.5, pad=10)

    plt.tight_layout()
    fig.savefig('fig4_composability.pdf', format='pdf')
    fig.savefig('fig4_composability.png', format='png', dpi=300)
    print('✓ fig4_composability.pdf')


if __name__ == '__main__':
    fig1_gap()
    fig2_diagnostic()
    fig3_heatmap()
    fig4_composability()
    print('\nAll figures generated. Use PDF versions in LaTeX.')
