"""Static pair curves and total three-sticker energies for the two definitions."""
from pathlib import Path
import argparse
import numpy as np
from common import ROOT
from potentials import RM, pair_energy, weight, weakening
from plot_style import plt, setup, format_axes, save, COLORS


def main(output):
    setup()
    output.mkdir(parents=True, exist_ok=True)
    r = np.linspace(.98, 1.51, 700)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9))
    fig.subplots_adjust(left=.10, right=.98, bottom=.21, top=.82, wspace=.32)
    cache = {'r': r}
    for ax, model, panel in zip(axes, COLORS, ['a', 'b']):
        for W, color, label in [(float(weakening(0.)), '#111111', 'Isolated pair'),
                                (.1, '#0072b2', r'$W=0.1$'),
                                (.01, '#e77500', r'$W=0.01$')]:
            u = pair_energy(r, W, model)
            cache[f'{model}_{W:g}'] = u
            ax.plot(r, u, color=color, label=label)
        ax.axhline(0, color='.65', linewidth=.5)
        ax.set(xlim=(.98, 1.51), ylim=(-.75, .45), xlabel=r'Separation, $r/\sigma$',
               title=model)
        format_axes(ax, panel)
    axes[0].set_ylabel(r'Pair energy, $U_{ij}/\varepsilon$')
    axes[1].legend(frameon=False, loc='lower right', handlelength=1.3)
    save(fig, output/'pair_potentials')
    np.savez_compressed(output/'pair_potentials.npz', **cache)

    # A and B are fixed at their isolated minimum. C is moved in the xy plane.
    # Sum all three coordination-dependent pair energies; subtract the isolated
    # AB energy so the far-separated third sticker has zero interaction energy.
    x = np.linspace(-2.2, 2.2, 321)
    y = np.linspace(0., 2.5, 201)
    X, Y = np.meshgrid(x, y)
    a = np.sqrt((X+RM/2)**2+Y**2)
    b = np.sqrt((X-RM/2)**2+Y**2)
    safe_a, safe_b = np.maximum(a, 1.e-6), np.maximum(b, 1.e-6)
    wa, wb = weight(a), weight(b)
    energies = {}
    for model in COLORS:
        ab = pair_energy(RM, weakening(wa+wb), model)
        ac = pair_energy(safe_a, weakening(1+wb), model)
        bc = pair_energy(safe_b, weakening(1+wa), model)
        isolated = pair_energy(RM, weakening(0.), model)
        energies[model] = ab+ac+bc-isolated
    mask = (a < .98) | (b < .98)
    fig, axes = plt.subplots(1, 3, figsize=(7, 3.0))
    fig.subplots_adjust(left=.075, right=.985, bottom=.31, top=.83, wspace=.24)
    for i, (ax, title, z) in enumerate(zip(axes,
        ['ReactiveLJ', 'ReactiveAH', 'AH − LJ'],
        [energies['ReactiveLJ'], energies['ReactiveAH'], energies['ReactiveAH']-energies['ReactiveLJ']])):
        limits = (-.1, 1.) if i < 2 else (-.5, .5)
        im = ax.contourf(X, Y, np.ma.array(z, mask=mask),
                         levels=np.linspace(*limits, 45), cmap='coolwarm', extend='both')
        ax.scatter([-RM/2, RM/2], [0, 0], color='#e77500', edgecolor='black', s=26, clip_on=False)
        ax.set(xlabel=r'$x_C/\sigma$', title=title, aspect='equal', xlim=(-2.2, 2.2), ylim=(0, 2.5))
        format_axes(ax, chr(97+i))
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.10, pad=.32,
                          extend='both')
        cb.set_label(r'$\Delta U/\varepsilon$' if i < 2 else r'$\Delta U_{\mathrm{AH}}/\varepsilon-\Delta U_{\mathrm{LJ}}/\varepsilon$')
    axes[0].set_ylabel(r'$y_C/\sigma$')
    save(fig, output/'three_sticker_energy')
    np.savez_compressed(output/'three_sticker_energy.npz', x=x, y=y, mask=mask, **energies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'results/figures')
    main(parser.parse_args().output)
