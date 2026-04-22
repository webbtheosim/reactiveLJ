#!/usr/bin/env python3
"""Build a one-slide PPTX explaining ReactiveLJ bond swapping."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SVG_PATH = ROOT / "reactiveLJ_swap_barrier_slide.svg"
PNG_PATH = ROOT / "reactiveLJ_swap_barrier_slide.png"
MD_PATH = ROOT / "reactiveLJ_swap_barrier_slide.md"
PPTX_PATH = ROOT / "reactiveLJ_swap_barrier_slide.pptx"

SIGMA = 1.0
R_CUT = 1.5
R_IN = 1.3
R_OUT = 1.5
P = 4.0
SMOOTH_KAPPA = 0.05
SMOOTH_BETA = 1.0


def shifted_lj_energy(r: np.ndarray | float, epsilon: float = 1.0) -> np.ndarray:
    r_arr = np.asarray(r, dtype=np.float64)
    r_safe = np.maximum(r_arr, 1e-12)
    sr = SIGMA / r_safe
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    sigma_over_rcut = SIGMA / R_CUT
    sigma_over_rcut_6 = sigma_over_rcut**6
    energy_shift = 4.0 * epsilon * (sigma_over_rcut_6 * sigma_over_rcut_6 - sigma_over_rcut_6)
    return 4.0 * epsilon * (sr12 - sr6) - energy_shift


def shifted_lj_force_magnitude(r: np.ndarray | float, epsilon: float = 1.0) -> np.ndarray:
    r_arr = np.asarray(r, dtype=np.float64)
    r_safe = np.maximum(r_arr, 1e-7 * SIGMA)
    inv_r = 1.0 / r_safe
    sr = SIGMA * inv_r
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    return 24.0 * epsilon * inv_r * (2.0 * sr12 - sr6)


def coordination_weight(distance: float) -> float:
    if distance >= R_OUT:
        return 0.0
    if distance <= R_IN:
        return 1.0
    fraction = (distance - R_IN) / (R_OUT - R_IN)
    return 0.5 * (1.0 + math.cos(math.pi * fraction))


def c_exc(raw: float) -> float:
    eps = 1e-6
    return 0.5 * (raw + math.sqrt(raw * raw + eps * eps))


def weakening(distance: float) -> float:
    raw = coordination_weight(distance)
    return (1.0 + c_exc(raw)) ** (-P)


def reactive_pair_curve(r: np.ndarray, third_bead_distance: float, epsilon: float = 1.0) -> np.ndarray:
    base_energy = shifted_lj_energy(r=r, epsilon=epsilon)
    pair_energy = np.maximum(base_energy, 0.0) + weakening(third_bead_distance) * np.minimum(base_energy, 0.0)

    sigma_over_rcut = SIGMA / R_CUT
    sigma_over_rcut_6 = sigma_over_rcut**6
    quadratic_rhs = sigma_over_rcut_6 * sigma_over_rcut_6 - sigma_over_rcut_6
    discriminant = 1.0 + 4.0 * quadratic_rhs
    sr6_at_zero = 0.5 * (1.0 + math.sqrt(discriminant))
    sr_root = sr6_at_zero ** (1.0 / 6.0)
    r_elbow = SIGMA / sr_root

    delta = SMOOTH_KAPPA * SIGMA * (max(0.0, 1.0 - weakening(third_bead_distance)) ** SMOOTH_BETA)
    r1 = r_elbow - delta
    r2 = r_elbow + delta
    width = r2 - r1
    smooth_delta_tol = 1e-6 * SIGMA
    if delta <= smooth_delta_tol or width <= smooth_delta_tol:
        return pair_energy

    mask = (r > r1) & (r < r2)
    if not np.any(mask):
        return pair_energy

    u1 = float(shifted_lj_energy(r1, epsilon=epsilon))
    du1 = -float(shifted_lj_force_magnitude(r1, epsilon=epsilon))
    u2 = weakening(third_bead_distance) * float(shifted_lj_energy(r2, epsilon=epsilon))
    du2 = -weakening(third_bead_distance) * float(shifted_lj_force_magnitude(r2, epsilon=epsilon))

    t = (r[mask] - r1) / width
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    pair_energy[mask] = h00 * u1 + h10 * width * du1 + h01 * u2 + h11 * width * du2
    return pair_energy


def build_curve_svg() -> str:
    plot_x = 1030
    plot_y = 170
    plot_w = 760
    plot_h = 470
    xmin, xmax = 0.92, 1.50
    ymin, ymax = -0.82, 2.25

    def x_map(val: float) -> float:
        return plot_x + (val - xmin) / (xmax - xmin) * plot_w

    def y_map(val: float) -> float:
        return plot_y + (ymax - val) / (ymax - ymin) * plot_h

    r = np.linspace(xmin, xmax, 550)
    curves = [
        {
            "label": "No third bead nearby",
            "distance": 1.50,
            "color": "#1f2a44",
            "dash": "",
        },
        {
            "label": "Third bead approaching (r = 1.40σ)",
            "distance": 1.40,
            "color": "#d46a3d",
            "dash": "10 8",
        },
        {
            "label": "Third bead inside r_in (r ≤ 1.30σ)",
            "distance": 1.30,
            "color": "#1f8a8a",
            "dash": "4 7",
        },
    ]

    pieces: list[str] = []
    pieces.append(f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" rx="24" fill="#ffffff" stroke="#d9d2c7" stroke-width="2"/>')

    for y_tick in [-0.6, 0.0, 0.6, 1.2, 1.8]:
        y = y_map(y_tick)
        pieces.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#ebe5dc" stroke-width="2"/>')
        pieces.append(
            f'<text x="{plot_x - 18}" y="{y + 7:.1f}" text-anchor="end" font-size="24" fill="#6d655b" font-family="DejaVu Sans">{y_tick:.1f}</text>'
        )

    for x_tick in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        x = x_map(x_tick)
        pieces.append(f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_y + plot_h}" stroke="#f0ebe3" stroke-width="2"/>')
        pieces.append(
            f'<text x="{x:.1f}" y="{plot_y + plot_h + 40}" text-anchor="middle" font-size="24" fill="#6d655b" font-family="DejaVu Sans">{x_tick:.1f}</text>'
        )

    zero_y = y_map(0.0)
    pieces.append(f'<line x1="{plot_x}" y1="{zero_y:.1f}" x2="{plot_x + plot_w}" y2="{zero_y:.1f}" stroke="#8a8176" stroke-width="3"/>')
    pieces.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#8a8176" stroke-width="3"/>')
    pieces.append(
        f'<text x="{plot_x + plot_w / 2:.1f}" y="{plot_y + plot_h + 82}" text-anchor="middle" font-size="30" fill="#3a342d" font-family="DejaVu Sans">pair distance r / σ</text>'
    )
    pieces.append(
        f'<text x="{plot_x - 92}" y="{plot_y + plot_h / 2:.1f}" transform="rotate(-90 {plot_x - 92},{plot_y + plot_h / 2:.1f})" text-anchor="middle" font-size="30" fill="#3a342d" font-family="DejaVu Sans">U(r) / ε_reactiveLJ</text>'
    )

    for curve in curves:
        yvals = reactive_pair_curve(r, curve["distance"])
        pts = " ".join(f"{x_map(float(rx)):.2f},{y_map(float(uy)):.2f}" for rx, uy in zip(r, yvals))
        dash_attr = f' stroke-dasharray="{curve["dash"]}"' if curve["dash"] else ""
        pieces.append(
            f'<polyline fill="none" stroke="{curve["color"]}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"{dash_attr} points="{pts}"/>'
        )

    legend_x = plot_x + 22
    legend_y = plot_y + 26
    for idx, curve in enumerate(curves):
        y = legend_y + idx * 42
        dash_attr = f' stroke-dasharray="{curve["dash"]}"' if curve["dash"] else ""
        pieces.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 58}" y2="{y}" stroke="{curve["color"]}" stroke-width="6"{dash_attr}/>')
        pieces.append(
            f'<text x="{legend_x + 76}" y="{y + 9}" font-size="25" fill="#3a342d" font-family="DejaVu Sans">{curve["label"]}</text>'
        )

    highlight_x = x_map(1.30)
    pieces.append(f'<line x1="{highlight_x:.1f}" y1="{plot_y + 12}" x2="{highlight_x:.1f}" y2="{plot_y + plot_h - 12}" stroke="#bfb5a8" stroke-width="3" stroke-dasharray="10 10"/>')
    pieces.append(
        f'<text x="{highlight_x + 14:.1f}" y="{plot_y + 300}" font-size="24" fill="#6d655b" font-family="DejaVu Sans">full crowding starts</text>'
    )

    pieces.append(
        f'<text x="{plot_x + plot_w - 18}" y="{plot_y + plot_h - 52}" text-anchor="end" font-size="24" fill="#6d655b" font-family="DejaVu Sans">same LJ core, shallower well</text>'
    )
    pieces.append(
        f'<text x="{plot_x + plot_w - 18}" y="{plot_y + plot_h - 20}" text-anchor="end" font-size="24" fill="#6d655b" font-family="DejaVu Sans">no added radial hump</text>'
    )
    return "\n".join(pieces)


def generate_svg() -> str:
    rc6 = (SIGMA / R_CUT) ** 6
    u_min_ratio = -1.0 - 4.0 * (rc6 * rc6 - rc6)
    crowded_factor = 2 ** (-P)
    barrier_ratio = (1.0 - 2 ** (1.0 - P)) * abs(u_min_ratio)

    answer_lines = [
        "Answer:",
        "No explicit swap-barrier term is added to U_ij(r).",
        "But swapping is not barrierless:",
        "crowding weakens both attractive bonds in the 3-body intermediate,",
        "so associative exchange pays an implicit energy penalty.",
    ]

    left_lines = [
        ("Core definition (Eq. 8, without smoothing):", "#1f2a44", 34, "bold"),
        ("U_ij^(0)(r) = max(U_LJ, 0) + W_ij min(U_LJ, 0)", "#2d2925", 30, "normal"),
        ("W_ij = (1 + C_ij^exc)^(-p)", "#2d2925", 30, "normal"),
        ("", "#2d2925", 22, "normal"),
        ("What the math means:", "#1f2a44", 34, "bold"),
        ("1. The positive LJ core is unchanged.", "#2d2925", 29, "normal"),
        ("2. Crowding only shrinks the negative well depth.", "#2d2925", 29, "normal"),
        ("3. Eqs. 9-13 only smooth the elbow near U_LJ = 0.", "#2d2925", 29, "normal"),
        ("", "#2d2925", 22, "normal"),
        ("Associative swap implication:", "#1f2a44", 34, "bold"),
        ("If a third bead fully enters the crowding shell, C_ij^exc ≈ 1,", "#2d2925", 29, "normal"),
        ("so each affected attraction becomes W_ij = 2^(-p).", "#2d2925", 29, "normal"),
        ("For p = 4, each bond is 16x weaker in the crowded state.", "#d46a3d", 29, "bold"),
    ]

    y = 170
    text_blocks: list[str] = []
    for line, color, size, weight in left_lines:
        if not line:
            y += 24
            continue
        text_blocks.append(
            f'<text x="118" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" font-family="DejaVu Sans">{line}</text>'
        )
        y += 42

    answer_y = 738
    answer_block = [
        '<rect x="88" y="720" width="1738" height="248" rx="30" fill="#fff6ec" stroke="#e1b78e" stroke-width="3"/>',
        '<text x="118" y="775" font-size="38" font-weight="bold" fill="#8e3f18" font-family="DejaVu Sans">Bottom line</text>',
    ]
    for idx, line in enumerate(answer_lines):
        answer_block.append(
            f'<text x="118" y="{answer_y + 52 + idx * 36}" font-size="30" fill="#352f2a" font-family="DejaVu Sans">{line}</text>'
        )

    formula_block = [
        '<rect x="1095" y="678" width="650" height="208" rx="24" fill="#eef7f7" stroke="#9fcfcf" stroke-width="3"/>',
        '<text x="1128" y="726" font-size="30" font-weight="bold" fill="#1f2a44" font-family="DejaVu Sans">Simple 3-bead estimate from the equations</text>',
        '<text x="1128" y="772" font-size="28" fill="#2d2925" font-family="DejaVu Sans Mono">E_single = U_min</text>',
        '<text x="1128" y="810" font-size="28" fill="#2d2925" font-family="DejaVu Sans Mono">E_crowded ≈ 2^(1-p) U_min</text>',
        f'<text x="1128" y="848" font-size="28" fill="#2d2925" font-family="DejaVu Sans Mono">ΔE_assoc ≈ (1 - 2^(1-p)) |U_min| ≈ {barrier_ratio:.3f} ε</text>',
        '<text x="1128" y="886" font-size="24" fill="#6d655b" font-family="DejaVu Sans">using p = 4 and r_cut = 1.5σ from Methods §2.4.3</text>',
    ]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">
<rect width="1920" height="1080" fill="#f6f2ea"/>
<rect x="0" y="0" width="1920" height="104" fill="#1f2a44"/>
<text x="84" y="66" font-size="40" font-weight="bold" fill="#ffffff" font-family="DejaVu Sans">
ReactiveLJ bond swapping: implicit crowding penalty, no explicit swap barrier
</text>
<text x="84" y="120" font-size="28" fill="#6d655b" font-family="DejaVu Sans">
Methods §2.4 “ReactiveLJ Potential for Associative Bonds” and subsections, interpreted at the equation level
</text>
<rect x="82" y="150" width="820" height="520" rx="28" fill="#ffffff" stroke="#d9d2c7" stroke-width="2"/>
{chr(10).join(text_blocks)}
{build_curve_svg()}
{chr(10).join(answer_block)}
{chr(10).join(formula_block)}
<text x="1826" y="1036" text-anchor="end" font-size="22" fill="#8b8277" font-family="DejaVu Sans">
From Eqs. 2-13 and default parameters in Eq. 20 paragraph
</text>
</svg>
"""
    return svg


def write_markdown() -> None:
    md = f"""---
title: ""
margin-left: 0
margin-right: 0
margin-top: 0
margin-bottom: 0
---

![]({PNG_PATH.name}){{width=13.333in height=7.5in}}
"""
    MD_PATH.write_text(md, encoding="utf-8")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(generate_svg(), encoding="utf-8")
    run(["convert", str(SVG_PATH), str(PNG_PATH)])
    write_markdown()
    run(["pandoc", str(MD_PATH), "-o", str(PPTX_PATH)])

    rc6 = (SIGMA / R_CUT) ** 6
    u_min_ratio = -1.0 - 4.0 * (rc6 * rc6 - rc6)
    barrier_ratio = (1.0 - 2 ** (1.0 - P)) * abs(u_min_ratio)
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PPTX_PATH}")
    print(f"U_min / epsilon = {u_min_ratio:.6f}")
    print(f"Associative crowding penalty / epsilon = {barrier_ratio:.6f}")


if __name__ == "__main__":
    main()
