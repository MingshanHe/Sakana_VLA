"""Figure for docs/wm_verifier_results.md (env: sakana_hyperdrone, CPU).

    python scripts/wm_results_figure.py --run runs/v1_stageA --out docs/assets/fig11_wm_verifier.png
Left: test-time scaling of the oracle verifier (SR and collision vs N, with and without primitive candidates).
Right: learned world-model verifiers against the baseline and the oracle at N=8.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET, MUTED, GRID, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984", "#e4e3df", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def load(run, tag):
    f = Path(run) / f"closed_loop_val_unseen_{tag}" / "results.json"
    return json.loads(f.read_text())["summary"] if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v1_stageA")
    ap.add_argument("--out", default="docs/assets/fig11_wm_verifier.png")
    args = ap.parse_args()
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.5]})

    # --- left: oracle scaling
    ax = axs[0]
    base = load(args.run, "bon_n1_first")
    for suffix, color, label in (("_oracle", BLUE, "oracle, policy samples only"), ("p_oracle", ORANGE, "oracle, + 4 primitive candidates")):
        pts = [(1, base)] if suffix == "_oracle" else []
        for n in (8, 16, 32):
            s = load(args.run, f"bon_n{n}{suffix}")
            if s:
                pts.append((n, s))
        if len(pts) < 2:
            continue
        ns = [p[0] for p in pts]
        ax.plot(ns, [p[1]["success"] for p in pts], "-o", color=color, lw=2, ms=6, label=f"{label}: success")
        ax.plot(ns, [p[1]["collision_rate"] for p in pts], "--s", color=color, lw=1.6, ms=5, alpha=0.8, label=f"{label}: collision")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 16, 32])
    ax.set_xticklabels(["1", "8", "16", "32"])
    ax.set_xlabel("candidates N (policy samples)")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=GRID)
    ax.set_title("Upper bound: oracle verifier (32 unseen episodes)", loc="left", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    # --- right: learned verifiers
    ax = axs[1]
    rows = [("policy alone (N=1)", "bon_n1_first", MUTED), ("random pick, N=8", "bon_n8_random", MUTED),
            ("oracle, N=8", "bon_n8_oracle", BLUE), ("oracle, N=8+4 prim.", "bon_n8p_oracle", BLUE),
            ("WM expert-only, N=8", "bon_n8_wmE", VIOLET), ("WM v1, N=8 (wm)", "bon_n8_wm1", ORANGE),
            ("WM v1, N=8 (wm_motion)", "bon_n8_wm1motion", ORANGE), ("WM v1, N=8+4 prim. (wm)", "bon_n8p_wm1", ORANGE),
            ("WM v1 shield, N=8", "bon_n8_wm1shield", ORANGE), ("WM v1, N=16 (wm)", "bon_n16_wm1", ORANGE),
            ("WM v1, N=16 (wm_motion)", "bon_n16_wm1motion", ORANGE), ("WM v2, N=8 (wm)", "bon_n8_wm2", AQUA),
            ("WM v2 shield, N=8", "bon_n8_wm2shield", AQUA), ("WM v2 shield loose, N=8", "bon_n8_wm2shieldL", AQUA),
            ("WM v2 shield loose, +prim.", "bon_n8p_wm2shieldL", AQUA)]
    rows = [(l, load(args.run, t), c) for l, t, c in rows]
    rows = [r for r in rows if r[1]]
    y = list(range(len(rows)))[::-1]
    for yi, (label, s, c) in zip(y, rows):
        ax.barh(yi + 0.18, s["success"], height=0.34, color=c)
        ax.barh(yi - 0.18, s["collision_rate"], height=0.34, color=c, alpha=0.35)
        ax.text(s["success"] + 0.01, yi + 0.18, f"{s['success']:.0%}", va="center", fontsize=8)
        ax.text(s["collision_rate"] + 0.01, yi - 0.18, f"{s['collision_rate']:.0%}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("Learned verifiers: success (solid) and collision (faded)", loc="left", fontsize=10.5)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
