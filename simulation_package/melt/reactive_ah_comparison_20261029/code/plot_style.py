"""Matplotlib settings shared with the manuscript's revision figures."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', f'/tmp/reactive_ah_mpl_{os.getuid()}')
os.environ.setdefault('XDG_CACHE_HOME', f'/tmp/reactive_ah_cache_{os.getuid()}')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

COLORS = {'ReactiveLJ': '#0072b2', 'ReactiveAH': '#e77500'}
MARKERS = {'ReactiveLJ': 'o', 'ReactiveAH': 's'}


def setup():
    font_dir = Path(os.environ.get('REACTIVELJ_FONT_DIR', str(Path(__file__).resolve().parents[1]/'fonts')))
    for path in font_dir.glob('*.ttf'):
        font_manager.fontManager.addfont(str(path))
    family = os.environ.get('REACTIVELJ_FONT', 'Arial')
    try:
        font_manager.findfont(family, fallback_to_default=False)
    except ValueError:
        family = 'DejaVu Sans'
    plt.rcParams.update({
        'font.family': family, 'font.size': 10, 'axes.labelsize': 10,
        'axes.titlesize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'axes.linewidth': .7, 'lines.linewidth': .9,
        'xtick.major.width': .7, 'ytick.major.width': .7,
        'mathtext.fontset': 'custom', 'mathtext.rm': family,
        'mathtext.it': family+':italic', 'mathtext.bf': family+':bold',
        'mathtext.fallback': 'stixsans', 'svg.fonttype': 'path',
        'svg.hashsalt': 'reactive-ah-comparison-20261029',
    })


def format_axes(ax, panel=None):
    ax.tick_params(which='both', direction='in', top=True, right=True)
    if panel is not None:
        ax.text(0, 1.055, f'({panel})', transform=ax.transAxes,
                ha='left', va='bottom', fontsize=11, fontweight='bold')


def save(fig, path):
    fig.savefig(path.with_suffix('.svg'), transparent=True)
    fig.savefig(path.with_suffix('.png'), dpi=220, facecolor='white')
    plt.close(fig)
