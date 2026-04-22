#!/usr/bin/env python3
"""Build a one-slide schematic for the associative-intermediate ΔE estimate."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SVG_PATH = ROOT / "reactiveLJ_assoc_geometry_slide.svg"
PNG_PATH = ROOT / "reactiveLJ_assoc_geometry_slide.png"
MD_PATH = ROOT / "reactiveLJ_assoc_geometry_slide.md"
PPTX_PATH = ROOT / "reactiveLJ_assoc_geometry_slide.pptx"

SIGMA = 1.0
R_CUT = 1.5
R_MIN = 2 ** (1 / 6)
R_IN = 1.3
P = 4


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: int = 5, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round"{dash_attr}/>'
    )


def circle(x: float, y: float, r: float, fill: str, stroke: str, width: int = 4, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def text(x: float, y: float, value: str, size: int = 28, color: str = "#2d2925", weight: str = "normal", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
        f'text-anchor="{anchor}" font-family="DejaVu Sans">{value}</text>'
    )


def panel_rect(x: float, y: float, w: float, h: float, fill: str = "#ffffff", stroke: str = "#d9d2c7") -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def build_svg() -> str:
    angle_deg = 100.0
    angle = math.radians(angle_deg)

    left_x, left_y, left_w, left_h = 82, 162, 845, 620
    right_x, right_y, right_w, right_h = 960, 162, 878, 620

    initial_A = (245, 430)
    initial_B = (425, 430)
    initial_C = (690, 300)

    assoc_A = (1310, 465)
    scale = 175 / R_MIN
    theta_B = math.pi / 2 + angle / 2
    theta_C = math.pi / 2 - angle / 2
    assoc_B = (
        assoc_A[0] + scale * R_MIN * math.cos(theta_B),
        assoc_A[1] - scale * R_MIN * math.sin(theta_B),
    )
    assoc_C = (
        assoc_A[0] + scale * R_MIN * math.cos(theta_C),
        assoc_A[1] - scale * R_MIN * math.sin(theta_C),
    )

    bc_dist = math.sqrt((assoc_B[0] - assoc_C[0]) ** 2 + (assoc_B[1] - assoc_C[1]) ** 2) / scale
    shift = 4.0 * ((SIGMA / R_CUT) ** 12 - (SIGMA / R_CUT) ** 6)
    u_min = -1.0 - shift
    dE_assoc = (1.0 - 2.0 ** (1 - P)) * abs(u_min)

    items: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">',
        '<rect width="1920" height="1080" fill="#f6f2ea"/>',
        '<rect x="0" y="0" width="1920" height="104" fill="#1f2a44"/>',
        text(84, 68, "Geometry behind ΔE_assoc: one deep bond becomes two crowded shallow bonds", 38, "#ffffff", "bold"),
        text(84, 122, "The estimate uses a hub-like 3-sticker intermediate consistent with Methods §2.4 and the default cutoffs", 27, "#6d655b"),
        panel_rect(left_x, left_y, left_w, left_h),
        panel_rect(right_x, right_y, right_w, right_h),
        panel_rect(82, 812, 1756, 208, "#fff6ec", "#e1b78e"),
        text(left_x + 30, left_y + 48, "1. Initial state used as reference", 32, "#1f2a44", "bold"),
        text(right_x + 30, right_y + 48, "2. Associative intermediate for the estimate", 32, "#1f2a44", "bold"),
    ]

    # Left panel: initial state
    items.extend(
        [
            circle(*initial_A, 42, "#f4c95d", "#8e6b00"),
            circle(*initial_B, 42, "#9ad1d4", "#1f8a8a"),
            circle(*initial_C, 42, "#e58f65", "#b04f2c"),
            line(*initial_A, *initial_B, "#2f3e63", 10),
            line(initial_B[0] + 52, initial_B[1] - 12, initial_C[0] - 58, initial_C[1] + 18, "#b7aea1", 4, "10 8"),
            text(initial_A[0], initial_A[1] + 10, "A", 34, "#2d2925", "bold", "middle"),
            text(initial_B[0], initial_B[1] + 10, "B", 34, "#2d2925", "bold", "middle"),
            text(initial_C[0], initial_C[1] + 10, "C", 34, "#2d2925", "bold", "middle"),
            text(335, 395, "A-B at r ≈ r_min", 27, "#2d2925", "normal", "middle"),
            text(560, 350, "C still free", 27, "#6d655b"),
            text(left_x + 30, 585, "Uncrowded A-B bond:", 29, "#1f2a44", "bold"),
            text(left_x + 30, 627, "C_AB^exc ≈ 0  →  W_AB = 1", 29, "#2d2925"),
            text(left_x + 30, 669, "E_initial ≈ U_min", 30, "#d46a3d", "bold"),
            text(left_x + 30, 724, "Interpretation: one full-strength attractive bond; no crowding penalty yet.", 27, "#2d2925"),
        ]
    )

    # Right panel: associative intermediate
    shell_r = scale * R_IN
    items.extend(
        [
            circle(*assoc_A, shell_r, "none", "#c8bfb2", 4, "10 8"),
            circle(*assoc_A, 44, "#f4c95d", "#8e6b00"),
            circle(*assoc_B, 42, "#9ad1d4", "#1f8a8a"),
            circle(*assoc_C, 42, "#e58f65", "#b04f2c"),
            line(*assoc_A, *assoc_B, "#2f3e63", 9),
            line(*assoc_A, *assoc_C, "#2f3e63", 9),
            line(*assoc_B, *assoc_C, "#b7aea1", 4, "10 8"),
            text(assoc_A[0], assoc_A[1] + 10, "A", 34, "#2d2925", "bold", "middle"),
            text(assoc_B[0], assoc_B[1] + 10, "B", 34, "#2d2925", "bold", "middle"),
            text(assoc_C[0], assoc_C[1] + 10, "C", 34, "#2d2925", "bold", "middle"),
            text(assoc_A[0] + 14, assoc_A[1] - shell_r - 18, "crowding shell r_in", 23, "#6d655b"),
            text((assoc_A[0] + assoc_B[0]) / 2 - 70, (assoc_A[1] + assoc_B[1]) / 2 - 12, "r_AB ≈ r_min", 26, "#2d2925"),
            text((assoc_A[0] + assoc_C[0]) / 2 + 18, (assoc_A[1] + assoc_C[1]) / 2 - 12, "r_AC ≈ r_min", 26, "#2d2925"),
            text((assoc_B[0] + assoc_C[0]) / 2 - 38, (assoc_B[1] + assoc_C[1]) / 2 - 20, f"r_BC ≈ {bc_dist:.2f}σ > r_cut", 26, "#6d655b"),
            text(right_x + 30, 562, "For pair A-B, sticker C crowds A but not B:", 24, "#2d2925"),
            text(right_x + 30, 594, "C_AB^exc ≈ 1  →  W_AB = 2^(-p)", 26, "#1f2a44", "bold"),
            text(right_x + 30, 634, "For pair A-C, sticker B does the analogous thing:", 24, "#2d2925"),
            text(right_x + 30, 666, "C_AC^exc ≈ 1  →  W_AC = 2^(-p)", 26, "#1f2a44", "bold"),
            text(right_x + 30, 706, "Two contacts exist simultaneously, but both are shallow.", 23, "#d46a3d", "bold"),
        ]
    )

    # Bottom explanation
    items.extend(
        [
            text(118, 860, "Qualitative explanation", 34, "#8e3f18", "bold"),
            text(118, 900, "No special radial hump is inserted into U(r).", 24),
            text(118, 934, "As C binds A, the old A-B well is weakened before the swap is complete.", 24),
            text(118, 968, "The hub intermediate has two contacts, but with p = 4 each is only 1/16 as deep.", 24),
            text(118, 1002, "That is why the associative route carries an energetic penalty even without an explicit barrier term.", 24),
            text(1148, 860, "Estimate used on the previous slide", 32, "#1f2a44", "bold"),
            text(1148, 900, "E_initial ≈ U_min", 24, "#2d2925"),
            text(1148, 934, "E_assoc ≈ 2 · 2^(-p) U_min", 24, "#2d2925"),
            text(1148, 968, "ΔE_assoc ≈ E_assoc - E_initial", 24, "#2d2925"),
            text(1148, 998, f"p = 4  →  ΔE_assoc ≈ {dE_assoc:.3f} eps", 23, "#d46a3d", "bold"),
            text(1830, 1036, "Chosen so A-B and A-C are near r_min while B-C stays beyond r_cut.", 22, "#8b8277", "normal", "end"),
        ]
    )

    items.append("</svg>")
    return "\n".join(items)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def write_markdown() -> None:
    MD_PATH.write_text(
        f"""---
title: ""
margin-left: 0
margin-right: 0
margin-top: 0
margin-bottom: 0
---

![]({PNG_PATH.name}){{width=13.333in height=7.5in}}
""",
        encoding="utf-8",
    )


def main() -> None:
    SVG_PATH.write_text(build_svg(), encoding="utf-8")
    run(["convert", str(SVG_PATH), str(PNG_PATH)])
    write_markdown()
    run(["pandoc", str(MD_PATH), "-o", str(PPTX_PATH)])
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PPTX_PATH}")


if __name__ == "__main__":
    main()
