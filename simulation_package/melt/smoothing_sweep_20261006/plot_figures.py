from pathlib import Path
import csv, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"reactive_ah_comparison_20261029/code"))
from plot_style import setup
setup()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent / 'results'
_font = plt.rcParams['font.family'][0]
rows = list(csv.DictReader((ROOT / 'all_grid_summary.csv').open()))
def series(kappa, beta):
    return sorted([r for r in rows if float(r['smooth_kappa']) == kappa
                   and float(r['smooth_beta']) == beta], key=lambda r: float(r['epsilon']))

plt.rcParams.update({
    'font.family': _font, 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.linewidth': 0.7, 'lines.linewidth': 0.9,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'mathtext.fontset': 'custom', 'mathtext.rm': _font,
    'mathtext.it': _font+':italic', 'mathtext.bf': _font+':bold',
    'mathtext.fallback': 'stixsans', 'svg.fonttype': 'path',
    'svg.hashsalt': 'reactivelj-smoothing-sweep-full-grid-20261011',
})
COLORS = {0.5: '#0072b2', 1: '#e69f00', 2: '#882255'}
KAPPAS = [0.025, 0.05, 0.1]
BETAS = [0.5, 1, 2]
EPS = [0, 3, 6, 9, 12, 15, 18]
TAU_R = 4041.0

def save(fig, stem):
    svg = ROOT / (stem + '.svg')
    fig.savefig(svg, transparent=True)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    fig.savefig(ROOT / (stem + '.pdf'), transparent=True)
    fig.savefig(ROOT / (stem + '.png'), dpi=240, facecolor='white')

def format_ax(ax, label):
    ax.set_xlim(-0.8, 18.8)
    ax.set_xticks(EPS)
    ax.tick_params(direction='in', length=3)
    ax.text(0.0, 1.04, label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=12, fontweight='bold')

fig, axes = plt.subplots(2, 3, figsize=(7, 4.4), sharex=True, sharey='row')
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.19,
                    top=0.91, wspace=0.18, hspace=0.18)
for col, kappa in enumerate(KAPPAS):
    top, bottom = axes[:, col]
    top.set_title(rf'$\kappa_{{\mathrm{{smooth}}}}={kappa:g}$', pad=10)
    for beta in BETAS:
        cell = series(kappa, beta)
        eps = np.array([float(r['epsilon']) for r in cell])
        color = COLORS[beta]
        top.errorbar(eps, [float(r['gelation_mean']) for r in cell],
                     yerr=[float(r['gelation_stderr']) for r in cell],
                     color=color, marker='o', markersize=2.8, capsize=2,
                     elinewidth=0.7, capthick=0.7)
        for kind, marker, linestyle, filled in [
                ('assoc', 'o', '-', True), ('dissoc', 's', '--', False)]:
            bottom.errorbar(eps,
                [float(r[f'rate_{kind}_mean']) * TAU_R for r in cell],
                yerr=[float(r[f'rate_{kind}_stderr']) * TAU_R for r in cell],
                color=color, marker=marker, markersize=3,
                linestyle=linestyle, markerfacecolor=color if filled else 'white',
                markeredgewidth=0.7, capsize=2, elinewidth=0.7, capthick=0.7)
    top.set_ylim(0, 2.05)
    top.set_yticks([0, 0.5, 1, 1.5, 2])
    bottom.set_yscale('log')
    bottom.set_ylim(0.12, 30)
    bottom.set_xlabel(r'$\varepsilon_{\mathrm{RLJ}}/\varepsilon_0$')
    format_ax(top, f'({chr(97 + col)})')
    format_ax(bottom, f'({chr(100 + col)})')
axes[0, 0].set_ylabel(r'Degree of gelation, $\mathcal{E}$')
axes[1, 0].set_ylabel(r'Apparent turnover, $\nu_\alpha\tau_R^{(0)}$')
legend = [Line2D([], [], color=COLORS[b], lw=1.2,
                 label=rf'$\beta_{{\mathrm{{smooth}}}}={b:g}$') for b in BETAS]
legend.extend([
    Line2D([], [], color='#111111', marker='o', markersize=3, lw=0.9,
           label='Associative-type'),
    Line2D([], [], color='#111111', marker='s', markersize=3, lw=0.9,
           linestyle='--', markerfacecolor='white', label='Dissociative-type')])
fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=5, frameon=False, fontsize=10, columnspacing=1,
           handletextpad=0.4, handlelength=1.5)
save(fig, 'full_grid_gelation_turnover')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(7, 3.0), sharex=True, sharey=True)
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.36,
                    top=0.84, wspace=0.18)
for col, kappa in enumerate(KAPPAS):
    ax = axes[col]
    ax.set_title(rf'$\kappa_{{\mathrm{{smooth}}}}={kappa:g}$', pad=10)
    for beta in BETAS:
        cell = series(kappa, beta)
        eps = np.array([float(r['epsilon']) for r in cell])
        fitted = np.array([float(r['tau_s_fit_mean_correlation']) for r in cell])
        error = np.array([float(r['tau_s_fit_bootstrap_stderr']) for r in cell])
        dt = np.array([float(r['frame_dt_tau_LJ']) for r in cell])
        span = np.array([float(r['observed_span_tau_LJ']) for r in cell])
        color = COLORS[beta]
        ax.plot(eps, fitted / TAU_R, color=color)
        for mask, marker, filled in [
                (fitted < dt, 'v', False),
                ((fitted >= dt) & (fitted <= span), 'o', True),
                (fitted > span, '^', False)]:
            ax.errorbar(eps[mask], fitted[mask] / TAU_R,
                        yerr=error[mask] / TAU_R, linestyle='none',
                        color=color, marker=marker, markersize=3.3,
                        markerfacecolor=color if filled else 'white',
                        markeredgewidth=0.7, capsize=2, elinewidth=0.7, capthick=0.7)
    ax.axhline(500 / TAU_R, color='#666666', linestyle=':', linewidth=0.7)
    ax.axhline(49500 / TAU_R, color='#666666', linestyle=':', linewidth=0.7)
    ax.set_yscale('log')
    ax.set_ylim(0.02, 250)
    ax.set_xlabel(r'$\varepsilon_{\mathrm{RLJ}}/\varepsilon_0$')
    format_ax(ax, f'({chr(97 + col)})')
axes[0].set_ylabel(r'Fitted persistence, $\tau_s/\tau_R^{(0)}$')
fig.legend(handles=legend[:3], loc='lower center', bbox_to_anchor=(0.5, 0.125),
           ncol=3, frameon=False, fontsize=10)
fig.legend(handles=[
    Line2D([], [], color='#111111', marker='v', markerfacecolor='white',
           linestyle='none', label='Below frame interval'),
    Line2D([], [], color='#111111', marker='^', markerfacecolor='white',
           linestyle='none', label='Beyond trajectory span')],
    loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=2,
    frameon=False, fontsize=10)
save(fig, 'full_grid_bond_persistence')
plt.close(fig)

print(ROOT / 'full_grid_gelation_turnover.svg')
print(ROOT / 'full_grid_bond_persistence.svg')
