"""Figures for docs/report.md (env: sakana_hyperdrone, CPU only).

    PYTHONPATH=. python scripts/report_figures.py --out docs/assets
Diagrams (pipeline, action chunk, model, closed loop) are drawn with matplotlib; data figures read the dataset,
the frame cache, runs/v1_stageA/log.jsonl, closed-loop results.json files and the result videos.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

# reference palette (light mode); model = orange, expert = blue everywhere (as in the videos)
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
TINT_BLUE, TINT_ORANGE, TINT_GRAY, TINT_AQUA = "#e3eefc", "#fde7dc", "#efeeea", "#dcf3ea"
RUN = Path("runs/v1_stageA")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})


# ----------------------------------------------------------------------------------------------- diagram helpers
def canvas(w, h, scale=0.85):
    fig = plt.figure(figsize=(w * scale, h * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body="", fc=TINT_GRAY, ec=MUTED, title_size=10.5, body_size=8.8, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=lw))
    if body:
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=title_size, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=body_size, color=INK2, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, weight="bold", color=INK)


def arrow(ax, p0, p1, text=None, color=INK2, rad=0.0, lw=1.4, text_off=(0, 0.15), fs=8.5, ha="center"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p0[0] + p1[0]) / 2 + text_off[0], (p0[1] + p1[1]) / 2 + text_off[1]
        ax.text(mx, my, text, ha=ha, va="bottom", fontsize=fs, color=INK2, linespacing=1.3)


def save(fig, out, name):
    fig.savefig(out / name, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", out / name, flush=True)


# ----------------------------------------------------------------------------------------------- 1. pipeline
def fig_pipeline(out):
    fig, ax = canvas(16.2, 7.3)
    xs = [0.3 + i * 3.2 for i in range(5)]
    w, h = 2.85, 1.85
    ax.text(0.3, 7.0, "1  Data generation  (datagen/, conda env sakana_hyperdrone)", fontsize=12, weight="bold", color=BLUE)
    row1 = [("ProcTHOR houses", "55 furnished houses\n3D mesh + object labels\nloaded by HyperDrone"),
            ("3D map", "0.1 m voxel grid\ndistance to walls/furniture\nsafe zone 0.5-2 m high"),
            ("Task sampling", "pick a target object\ngoal = a spot ~1 m in front\nstart 3-12 m away"),
            ("Expert flight", "shortest safe path, smoothed\n0.5 m/s, turn to face target\nRAPTOR flies the drone\n(crazyflie dynamics, 100 Hz)"),
            ("Render + label", "10 Hz: RGB, depth, seg\naction [dx,dy,dz,dyaw,stop]\ntarget box, instruction")]
    for i, (t, b) in enumerate(row1):
        box(ax, xs[i], 4.85, w, h, t, b, fc=TINT_BLUE, ec=BLUE)
        if i:
            arrow(ax, (xs[i - 1] + w, 4.85 + h / 2), (xs[i], 4.85 + h / 2))
    ax.text(0.3, 4.2, "2  Training and testing  (sakana_vla/, conda env sakana_vla2)", fontsize=12, weight="bold", color=ORANGE)
    row2 = [("Dataset", "LeRobot v2.1 format\n1,723 episodes, 260k frames\nHF: hemingshan/vla_dataset"),
            ("Training (stage A)", "Qwen3.5-0.8B, frozen\n+ 233M action expert\nflow matching, 16k steps"),
            ("Offline test", "predict the next 1.2 s\non held-out frames\nposition / yaw / stop error"),
            ("Closed-loop test", "the VLA flies the drone\nin 10 unseen houses\nlooks again every 0.5 s"),
            ("Results", "success rate, ablations\nvideos\nfailure analysis")]
    for i, (t, b) in enumerate(row2):
        box(ax, xs[i], 1.45, w, h, t, b, fc=TINT_ORANGE, ec=ORANGE)
        if i:
            arrow(ax, (xs[i - 1] + w, 1.45 + h / 2), (xs[i], 1.45 + h / 2))
    # row 1 end -> row 2 start (runs below the row-2 heading)
    x_end, x_start = xs[4] + w / 2, xs[0] + w / 2
    ax.plot([x_end, x_end, x_start], [4.85, 3.72, 3.72], color=INK2, lw=1.4)
    arrow(ax, (x_start, 3.72), (x_start, 1.45 + h))
    ax.text(0.3, 0.75, "Per house: ~90 s on one H100 (OptiX rendering).  Training: ~85 min on one H100.  "
            "Closed loop: simulator on CPU, VLA on GPU, talking over a local Unix socket.",
            fontsize=9, color=INK2)
    save(fig, out, "fig1_pipeline.png")


# ----------------------------------------------------------------------------------------------- 2. sample episode
def fig_sample(out, subset="indoor_mini_unseen", ep=90):
    from datagen.geometry import build_maps
    cache_eps = json.loads(Path("data/cache/v1/episodes.json").read_text())
    ce = next(e for e in cache_eps if e["subset"] == subset and e["episode_index"] == ep)
    frames = np.load("data/cache/v1/frames.npy", mmap_mode="r")
    root = Path("data/vla_dataset") / subset
    t = pq.read_table(root / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet")
    sf = np.array(t.column("observation.state_full").to_pylist())
    boxes = np.array(t.column("observation.boxes").to_pylist())
    acts = np.array(t.column("actions").to_pylist())
    phase = t.column("phase").to_pylist()
    depth = np.load(root / f"depth/chunk-{ep // 1000:03d}/episode_{ep:06d}.npz")["depth"] / 1000.0
    extra = [json.loads(x) for x in (root / "meta/episodes_extra.jsonl").read_text().splitlines()][ep]
    scene = json.loads(Path(f"data/scenes/{extra['house']}.json").read_text())
    maps = build_maps(scene["glb"])
    g = maps["grid"]
    z = g.origin[2] + (np.arange(g.shape[2]) + 0.5) * g.res - maps["floor_z"]
    band = (z >= 0.5) & (z <= 2.0)
    occ = maps["occ"][:, :, band].any(2)
    img = np.full(occ.shape + (3,), 1.0)
    img[maps["inside"]] = [0.93, 0.93, 0.91]
    img[occ & maps["inside"]] = [0.45, 0.45, 0.43]
    pos = sf[:, :3]
    n = len(pos)
    kf = [0, int(0.36 * n), int(0.62 * n), n - 1]
    pert = [i for i, p in enumerate(phase) if p == "perturb"]

    fig = plt.figure(figsize=(15.5, 6.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.6, 1, 1, 1, 1], wspace=0.06, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    pad = 1.0
    lo, hi = pos[:, :2].min(0) - pad, pos[:, :2].max(0) + pad
    lo_t, hi_t = np.array(extra["target"]["aabb"])
    lo, hi = np.minimum(lo, lo_t[:2] - 0.5), np.maximum(hi, hi_t[:2] + 0.5)
    x0, y0 = g.origin[0], g.origin[1]
    ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=[x0, x0 + g.shape[0] * g.res, y0, y0 + g.shape[1] * g.res],
              interpolation="nearest")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.plot(pos[:, 0], pos[:, 1], color=BLUE, lw=2.2, label="expert path")
    if pert:
        ax.plot(pos[pert, 0], pos[pert, 1], color=YELLOW, lw=5, solid_capstyle="round", label="random push (perturbation)")
    ax.add_patch(Rectangle(lo_t[:2], *(hi_t[:2] - lo_t[:2]), fc="none", ec=RED, lw=2, label=f"target: {extra['target']['type']}"))
    ax.plot(*pos[0, :2], "o", color=INK, ms=8, label="start")
    ax.plot(*extra["goal"][:2], "*", color=RED, ms=15, mec=SURFACE, label="goal viewpoint")
    for k, f in enumerate(kf):
        yaw = np.arctan2(2 * (sf[f, 3] * sf[f, 6] + sf[f, 4] * sf[f, 5]), 1 - 2 * (sf[f, 5] ** 2 + sf[f, 6] ** 2))
        ax.annotate("", xy=pos[f, :2] + 0.55 * np.array([np.cos(yaw), np.sin(yaw)]), xytext=pos[f, :2],
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
        ax.text(pos[f, 0] - 0.25, pos[f, 1] + 0.2, str(k + 1), fontsize=11, weight="bold", color=INK,
                bbox=dict(boxstyle="circle,pad=0.15", fc=SURFACE, ec=INK, lw=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    rooms = " -> ".join(dict.fromkeys(r["type"] for r in extra["rooms"]))
    ax.set_title(f"Top view ({extra['house']}, unseen)", fontsize=10, pad=16)
    ax.text(0.5, 1.005, f"rooms: {rooms}", transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8.5)
    for k, f in enumerate(kf):
        a1 = fig.add_subplot(gs[0, k + 1])
        a1.imshow(frames[ce["offset"] + f])
        b = boxes[f]
        if b[2] > 0:
            a1.add_patch(Rectangle((b[0] * 256, b[1] * 256), b[2] * 256, b[3] * 256, fc="none", ec=RED, lw=1.8))
        st = "stop=1" if acts[f, -1] > 0.5 else "stop=0"
        a1.set_title(f"{k + 1}   t = {f / 10:.1f} s   ({st})", fontsize=10)
        a1.axis("off")
        a2 = fig.add_subplot(gs[1, k + 1])
        im = a2.imshow(depth[f], cmap="Blues", vmin=0, vmax=6)
        a2.axis("off")
        if k == 0:
            a1.text(-12, 128, "RGB", rotation=90, va="center", ha="right", fontsize=10, color=INK2)
            a2.text(-12, 128, "depth", rotation=90, va="center", ha="right", fontsize=10, color=INK2)
    cax = fig.add_axes([0.915, 0.13, 0.008, 0.33])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("m", color=INK2)
    cb.outline.set_visible(False)
    instr = ce["caption"]
    fig.suptitle(f"Instruction (template draft): \"{instr}\"", fontsize=10.5, y=0.99, color=INK)
    save(fig, out, "fig2_sample_episode.png")


# ----------------------------------------------------------------------------------------------- 3. action chunk
def fig_action(out):
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.6], wspace=0.08)
    # (a) body frame
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.7, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")
    for dx, dy in [(-0.35, -0.35), (0.35, -0.35), (-0.35, 0.35), (0.35, 0.35)]:
        ax.add_patch(plt.Circle((dx, dy), 0.22, fc=TINT_GRAY, ec=MUTED))
    ax.plot([-0.35, 0.35], [-0.35, 0.35], color=MUTED, lw=3)
    ax.plot([-0.35, 0.35], [0.35, -0.35], color=MUTED, lw=3)
    ax.annotate("", xy=(0, 1.75), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5, mutation_scale=18))
    ax.text(0.12, 1.55, "dx  (forward)", color=INK, fontsize=10.5, weight="bold")
    ax.annotate("", xy=(-1.6, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5, mutation_scale=18))
    ax.text(-1.95, 0.15, "dy  (left)", color=INK, fontsize=10.5, weight="bold")
    ax.plot(0, 0, "o", ms=11, mfc=SURFACE, mec=ORANGE, mew=2)
    ax.plot(0, 0, "o", ms=3.5, color=ORANGE)
    ax.text(0.68, -0.12, "dz  (up, out of page)", color=INK, fontsize=10.5, weight="bold")
    ax.add_patch(FancyArrowPatch((0.95, 0.75), (-0.6, 1.1), connectionstyle="arc3,rad=0.45", arrowstyle="-|>",
                                 mutation_scale=15, color=VIOLET, lw=2))
    ax.text(0.85, 1.15, "dyaw  (turn left > 0)", color=INK, fontsize=10.5, weight="bold")
    ax.text(0, -1.0, "stop: 1 = \"I have arrived, hover here\"", ha="center", fontsize=10.5, weight="bold")
    ax.text(0, -1.5, "One action = how the drone should move in the next 0.1 s,\nmeasured in its own heading frame. "
            "At 0.5 m/s, dx = 0.05 m.", ha="center", fontsize=8.8, color=INK2)

    # (b) chunk timeline
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(-0.75, 2.75)
    ax.set_ylim(-1.35, 3.4)
    ax.axis("off")
    for r in range(3):
        y = 2.6 - r * 0.85
        t0 = 0.5 * r
        ax.text(t0 - 0.05, y + 0.17, f"VLA call #{r + 1}\n@ t = {t0:.1f} s", ha="right", va="center", fontsize=9, color=INK2)
        for k in range(12):
            flown = k < 5
            ax.add_patch(Rectangle((t0 + 0.1 * k + 0.004, y), 0.092, 0.34, fc=ORANGE if flown else TINT_ORANGE,
                                   ec=SURFACE if flown else ORANGE, lw=0.8))
        ax.text(t0 + 1.22, y + 0.17, "12 steps = 1.2 s", va="center", fontsize=8.5, color=INK2)
    y = 0.0
    ax.text(-0.05, y + 0.17, "what is\nactually flown", ha="right", va="center", fontsize=9, color=INK, weight="bold")
    for k in range(15):
        ax.add_patch(Rectangle((0.1 * k + 0.004, y), 0.092, 0.34, fc=ORANGE, ec=SURFACE, lw=0.8))
    for s in (0.5, 1.0):
        ax.plot([s, s], [-0.15, 3.1], color=INK2, lw=1, ls=(0, (3, 3)))
    ax.plot([0, 2.5], [-0.35, -0.35], color=MUTED, lw=1)
    for s in np.arange(0, 2.51, 0.5):
        ax.plot([s, s], [-0.4, -0.3], color=MUTED, lw=1)
        ax.text(s, -0.55, f"{s:.1f} s", ha="center", va="top", fontsize=8.5, color=INK2)
    ax.add_patch(Rectangle((1.62, -1.2), 0.092, 0.25, fc=ORANGE, ec=SURFACE))
    ax.text(1.75, -1.07, "flown (first 5 steps)", va="center", fontsize=8.8, color=INK2)
    ax.add_patch(Rectangle((2.45, -1.2), 0.092, 0.25, fc=TINT_ORANGE, ec=ORANGE))
    ax.text(2.58, -1.07, "thrown away", va="center", fontsize=8.8, color=INK2)
    fig.text(0.02, 0.97, "(a)  One action = [dx, dy, dz, dyaw, stop]", fontsize=11, weight="bold")
    fig.text(0.31, 0.97, "(b)  Action chunk: predict 1.2 s, fly 0.5 s, look again (re-plan / correct)", fontsize=11,
             weight="bold")
    save(fig, out, "fig3_action_chunk.png")


# ----------------------------------------------------------------------------------------------- 4. model
def fig_model(out):
    fig, ax = canvas(15.0, 11.2)
    full = {3, 7, 11, 15, 19, 23}
    y0, step, sh = 2.45, 0.235, 0.19
    cols = {"vlm": (0.6, 4.6), "exp": (8.6, 11.4)}
    for key, (x0, x1) in cols.items():
        title = ("Qwen3.5-0.8B  (VLM, frozen)", "853M params, 24 layers, width 1024") if key == "vlm" else \
                ("Action expert  (trained)", "233M params, 24 layers, width 640")
        ec, fc = (BLUE, TINT_BLUE) if key == "vlm" else (ORANGE, TINT_ORANGE)
        ax.add_patch(FancyBboxPatch((x0 - 0.25, y0 - 0.3), x1 - x0 + 0.5, 24 * step + 1.35,
                                    boxstyle="round,pad=0,rounding_size=0.15", fc=fc, ec=ec, lw=1.4))
        ax.text((x0 + x1) / 2, y0 + 24 * step + 0.82, title[0], ha="center", fontsize=11.5, weight="bold")
        ax.text((x0 + x1) / 2, y0 + 24 * step + 0.42, title[1], ha="center", fontsize=9, color=INK2)
        for i in range(24):
            y = y0 + i * step
            is_full = i in full
            ax.add_patch(Rectangle((x0, y), x1 - x0, sh, fc=(ec if is_full else "#ffffff"), ec=(ec if is_full else MUTED), lw=0.6))
            if is_full:
                ax.text((x0 + x1) / 2, y + sh / 2, f"layer {i}: full attention", ha="center", va="center", fontsize=7.5,
                        color="#ffffff", weight="bold")
        ax.text(x0 - 0.12, y0 + 1 * step + sh / 2, "linear attention\n(Gated DeltaNet)", ha="right", va="center", fontsize=7.5,
                color=INK2) if key == "vlm" else None
    # KV sharing arrows
    for i in sorted(full):
        y = y0 + i * step + sh / 2
        arrow(ax, (cols["vlm"][1] + 0.27, y), (cols["exp"][0] - 0.27, y), color=VIOLET, lw=1.6)
    ax.text(6.6, y0 + 13 * step + 0.06, "K, V of the image + text\n(read only, computed once)", ha="center", fontsize=8.8,
            color=VIOLET, weight="bold")
    # inputs
    box(ax, 0.1, 0.25, 2.55, 1.45, "Camera RGB", "256 x 256 image\n-> vision encoder\n-> 64 tokens", fc="#ffffff", ec=BLUE,
        body_size=8.3)
    box(ax, 2.85, 0.25, 2.5, 1.45, "Instruction", "\"Find the television in\nthis living room and\nstop in front of it ...\"",
        fc="#ffffff", ec=BLUE, body_size=8.3)
    box(ax, 7.6, 0.25, 2.25, 1.45, "State (1 token)", "body velocity vx, vy, vz\nyaw rate, height", fc="#ffffff", ec=ORANGE,
        body_size=8.3)
    box(ax, 10.05, 0.25, 2.45, 1.45, "Noisy chunk (12 tokens)", "x: 12 x 5 numbers\n+ flow time t", fc="#ffffff",
        ec=ORANGE, body_size=8.3)
    for xa in (1.375, 4.1):
        arrow(ax, (xa, 1.7), (xa, y0 - 0.3))
    for xa in (8.7, 11.27):
        arrow(ax, (xa, 1.7), (min(max(xa, 8.75), 11.25), y0 - 0.3))
    # output and the flow loop
    yt = y0 + 24 * step + 1.05
    box(ax, 8.3, yt + 0.35, 3.4, 0.95, "velocity v  (12 x 5)", "", fc="#ffffff", ec=ORANGE)
    arrow(ax, (10.0, yt), (10.0, yt + 0.35))
    ax.add_patch(FancyArrowPatch((11.7, yt + 0.82), (12.5, 1.2), connectionstyle="arc3,rad=-0.2", arrowstyle="-|>",
                                 mutation_scale=14, color=ORANGE, lw=1.8))
    ax.text(13.2, 5.6, "x <- x - 0.2 v\nrepeat 5 times\n(t: 1.0 -> 0.0)", ha="left", fontsize=9.5, color=INK,
            weight="bold")
    box(ax, 12.5, yt + 0.35, 2.4, 0.95, "action chunk", "", fc=ORANGE, ec=ORANGE)
    ax.text(13.7, yt + 0.48, "12 x [dx,dy,dz,dyaw,stop]", ha="center", fontsize=7.8, color="#ffffff")
    arrow(ax, (11.7, yt + 0.95), (12.5, yt + 0.95), text="after\n5 steps", fs=7.5, text_off=(0, 0.02))
    ax.text(0.1, yt + 0.85, "Start from random noise x and \"denoise\" it into a chunk of actions (flow matching).\n"
            "Training: take a real chunk a and noise z, mix x = t z + (1 - t) a,\n"
            "and teach the expert to output v = z - a (mean squared error). Only the expert learns.",
            fontsize=9, color=INK2, va="center", linespacing=1.5)
    save(fig, out, "fig4_model.png")


# ----------------------------------------------------------------------------------------------- 5. closed loop
def fig_loop(out):
    fig, ax = canvas(15.0, 5.0)
    yb, hb = 2.25, 2.35
    box(ax, 0.2, yb, 3.9, hb, "HyperDrone simulator", "crazyflie dynamics (L2F), 100 Hz\nhyperlux camera 256 x 256,\n"
        "FOV 90 deg\ncollision check vs. 3D map", fc=TINT_BLUE, ec=BLUE)
    box(ax, 5.55, yb, 3.9, hb, "Policy server (GPU)", "SakanaVLA checkpoint\nimage + instruction + state\n"
        "-> 5 flow steps -> 12 x 5 chunk\n(own process, local Unix socket)", fc=TINT_ORANGE, ec=ORANGE)
    box(ax, 10.9, yb, 3.9, hb, "Executor", "chunk -> world waypoints\n(attached to the reference path)\n"
        "-> 100 Hz setpoints\n-> RAPTOR position controller", fc=TINT_GRAY, ec=MUTED)
    ym = yb + hb / 2
    arrow(ax, (4.1, ym), (5.55, ym), text="RGB + state\nevery 0.5 s", text_off=(0, 0.08))
    arrow(ax, (9.45, ym), (10.9, ym), text="chunk\n12 x 5", text_off=(0, 0.08))
    ax.plot([12.85, 12.85, 2.15], [yb, 1.6, 1.6], color=INK2, lw=1.4)
    arrow(ax, (2.15, 1.6), (2.15, yb))
    ax.text(7.5, 1.68, "fly the first 0.5 s of the chunk (50 physics steps), then ask the VLA again", ha="center",
            va="bottom", fontsize=9, color=INK2)
    ax.text(0.2, 0.15, "Episode ends when: stop is predicted twice in a row  |  collision (< 0.12 m from a surface)  |  40 s.\n"
            "Success = stopped within 1.0 m of the goal viewpoint, target visible (>= 150 px), no collision.",
            fontsize=9.5, color=INK, linespacing=1.6)
    save(fig, out, "fig5_closed_loop.png")


# ----------------------------------------------------------------------------------------------- 6. training curves
def fig_training(out):
    rows = [json.loads(x) for x in (RUN / "log.jsonl").read_text().splitlines()]
    tr = [r for r in rows if "loss_ema" in r]
    ev = [r for r in rows if "val_seen" in r]
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.9))
    ax = axs[0]
    ax.plot([r["step"] for r in tr], [r["loss_ema"] for r in tr], color=ORANGE, lw=2)
    ax.set_yscale("log")
    ax.set_title("Training loss (flow matching, EMA)", loc="left")
    ax.set_xlabel("training step")
    ax.grid(axis="y", color=GRID, lw=0.8)
    s = [r["step"] for r in ev]
    for ax, key, cv, title, mul, unit in ((axs[1], "model_ade", "constvel_ade", "Position error over 1.2 s (ADE)", 100, "cm"),
                                          (axs[2], "model_yaw_fde_deg", "constvel_yaw_fde_deg", "Heading error at 1.2 s", 1, "deg")):
        for split, c in (("val_seen", BLUE), ("val_unseen", ORANGE)):
            y = [r[split][key] * mul for r in ev]
            lab = "held-out episodes, seen houses" if split == "val_seen" else "unseen houses"
            ax.plot(s, y, "-o", color=c, lw=2, ms=6, mec=SURFACE, mew=1.5, label=f"VLA, {lab}")
            ax.axhline(ev[-1][split][cv] * mul, color=c, lw=1.2, ls=(0, (4, 3)))
        ax.set_ylim(0, max(ev[-1]["val_seen"][cv], ev[-1]["val_unseen"][cv]) * mul * 1.12)
        ax.set_title(title, loc="left")
        ax.set_xlabel("training step")
        ax.set_ylabel(unit)
        ax.grid(axis="y", color=GRID, lw=0.8)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=BLUE, lw=2, marker="o"), Line2D([], [], color=ORANGE, lw=2, marker="o"),
          Line2D([], [], color=MUTED, lw=1.2, ls=(0, (4, 3)))]
    fig.legend(hs, ["VLA, held-out episodes in training houses (val_seen)", "VLA, 10 unseen houses (val_unseen)",
                    "constant-velocity guess (same color per split)"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=9.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "fig6_training.png")


# ----------------------------------------------------------------------------------------------- 7. closed-loop results
def bars(axs, policies, metrics, colors):
    for ax, (key, title) in zip(axs, metrics):
        vals = [p[1].get(key) for p in policies]
        x = np.arange(len(policies))
        ax.bar(x, [v or 0 for v in vals], width=0.68, color=colors, edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(x, vals):
            lab = "not\nmeasured" if v is None else (f"{v:.0%}" if v < 0.995 else "100%")
            ax.text(xi, (v or 0) + 0.025, lab, ha="center", va="bottom", fontsize=9 if v is not None else 8,
                    color=INK if v is not None else MUTED)
        ax.set_ylim(0, 1.12)
        ax.set_xticks([])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "50%", "100%"])
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def fig_results(out):
    def summ(path):
        return json.loads(Path(path).read_text())["summary"]
    expert = summ("runs/v1_snap8k/closed_loop_val_unseen_oracle_ref_yl30/results.json")
    policies = [("expert replay (check of the executor)", expert),
                ("VLA", summ(RUN / "closed_loop_val_unseen_main/results.json")),
                ("VLA, black images", summ(RUN / "closed_loop_val_unseen_blind/results.json")),
                ("VLA, shuffled instructions", summ(RUN / "closed_loop_val_unseen_shuffle/results.json"))]
    colors = [BLUE, ORANGE, AQUA, VIOLET]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    bars(axs, policies, [("success", "Success rate"), ("near_target_ever", "Got within 1.5 m of target"),
                         ("collision_rate", "Collision"), ("timeout_rate", "Timeout (40 s)")], colors)
    handles = [Rectangle((0, 0), 1, 1, fc=c) for c in colors]
    fig.legend(handles, [p[0] for p in policies], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=9.5)
    fig.suptitle("Closed loop, 32 episodes in 10 unseen houses", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    save(fig, out, "fig7_closed_loop_results.png")

    variants = [("chunk starts at the measured pose", "runs/v1_snap8k/closed_loop_val_unseen_oracle/results.json"),
                ("measured pose + 0.3 s yaw lead", "runs/v1_snap8k/closed_loop_val_unseen_oracle_yl30/results.json"),
                ("chunk continues the reference path\n+ 0.3 s yaw lead  (final)",
                 "runs/v1_snap8k/closed_loop_val_unseen_oracle_ref_yl30/results.json")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 2.6), sharey=True)
    for ax, (key, title, c) in zip(axs, (("success", "Success rate", BLUE), ("collision_rate", "Collision", BLUE))):
        vals = [summ(p)[key] for _, p in variants]
        y = np.arange(len(variants))[::-1]
        ax.barh(y, vals, height=0.6, color=c, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + 0.02, yi, f"{v:.0%}", va="center", fontsize=9.5)
        ax.set_yticks(y)
        ax.set_yticklabels([v[0] for v in variants], fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50%", "100%"])
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Executor check: replay the expert's own chunks (32 unseen episodes)", x=0.01, ha="left", fontsize=11.5,
                 weight="bold")
    fig.tight_layout()
    save(fig, out, "fig8_executor_fix.png")


# ----------------------------------------------------------------------------------------------- 9. episode overview
NOTES = {139: "followed the expert route, then stopped\nat a laptop instead of the TV",
         157: "followed the expert route to the goal,\ndid not stop, overshot into the wall",
         124: "flew away from the target and back (7.6 m),\nthen hit the wall next to the start",
         15: "turned 140 deg in place,\nthen stopped at the start",
         155: "spun ~2.8 turns, wandered,\nstopped 3.9 m from the goal",
         158: "stopped right after take-off (1.5 s)"}


def last_frame(path):
    r = imageio.get_reader(path)
    f = None
    for f in r:
        pass
    r.close()
    return f


def fig_episodes(out, vids=RUN / "videos"):
    res = {e["id"].split("/")[1]: e for e in json.loads((RUN / "closed_loop_val_unseen_video/results.json").read_text())["episodes"]}
    order = [139, 157, 124, 155, 15, 158]
    fig, axs = plt.subplots(2, 3, figsize=(14, 10.4))
    for ax, i in zip(axs.flat, order):
        f = next(vids.glob(f"indoor_mini_unseen_{i}_*.mp4"))
        img = last_frame(f)[112:112 + 384 - 26, 2 * 384:3 * 384]
        r = res[str(i)]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"#{i}   target: {r['target']}\nVLA: {r['status']}, {r['ne']:.1f} m from goal", fontsize=10.5)
        ax.text(0.5, -0.02, NOTES[i], transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK2)
    fig.suptitle("Final top views (orange = VLA, blue = expert replay, red = target box / goal star, black dot = start)",
                 fontsize=11.5, weight="bold", y=0.995)
    fig.tight_layout(h_pad=3.0)
    save(fig, out, "fig9_episodes.png")


# ----------------------------------------------------------------------------------------------- 12. Dreamer x VLA
def fig_wm_arch(out):
    """docs/report_v1.md: the VLA proposes N candidate chunks, a DreamerV3 world model scores them in imagination."""
    fig, ax = canvas(16.4, 9.0)
    yb, hb = 5.3, 2.6
    box(ax, 0.2, yb, 2.6, hb, "Inputs", "RGB 256 x 256\ninstruction\nstate (velocity,\nyaw rate, height)", fc="#ffffff", ec=BLUE)
    box(ax, 3.4, yb, 3.8, hb, "SakanaVLA (v0, unchanged)", "Qwen3.5-0.8B + action expert\none VLM pass, N noise seeds\n"
        "-> N action chunks\n12 x [dx, dy, dz, dyaw, stop]", fc=TINT_ORANGE, ec=ORANGE)
    box(ax, 7.8, yb, 2.6, hb, "Candidates", "N policy samples\n+ 4 hand-made:\nhover, turn left / right,\nslow forward",
        fc="#ffffff", ec=ORANGE)
    box(ax, 11.0, yb, 5.2, hb, "World model (DreamerV3 RSSM)", "53M params, one step = 0.5 s = one decision\n"
        "imagine the first 1.0 s of every candidate\nheads: progress, collision, clearance,\ntarget visible, at goal",
        fc=TINT_AQUA, ec=AQUA)
    ym = yb + hb / 2
    arrow(ax, (2.8, ym), (3.4, ym))
    arrow(ax, (7.2, ym), (7.8, ym))
    arrow(ax, (10.4, ym), (11.0, ym))
    yb2, hb2 = 2.0, 2.0
    box(ax, 11.0, yb2, 5.2, hb2, "Score and pick", "score = progress - 10 P(collision)\n- 5 max(0, 0.22 m - clearance) + 0.5 P(target visible)\n"
        "stop candidate: +2 if P(at goal) > 0.5, else -2", fc="#ffffff", ec=AQUA, body_size=8.3)
    arrow(ax, (13.6, yb), (13.6, yb2 + hb2), text="predicted outcomes\nof each candidate", text_off=(0.15, -0.3), ha="left", fs=8)
    box(ax, 5.6, yb2, 3.9, hb2, "Executor (simulator)", "fly the chosen chunk for 0.5 s\n(RAPTOR position control)\nthen ask again",
        fc=TINT_GRAY, ec=MUTED)
    arrow(ax, (11.0, yb2 + hb2 / 2), (9.5, yb2 + hb2 / 2), text="best chunk", text_off=(0, 0.08))
    # feedback 1: new camera image and state -> inputs
    ax.plot([5.6, 1.5, 1.5], [yb2 + 0.5, yb2 + 0.5, yb - 0.6], color=INK2, lw=1.4)
    arrow(ax, (1.5, yb - 0.6), (1.5, yb))
    ax.text(3.5, yb2 + 0.58, "next RGB + state (every 0.5 s)", ha="center", va="bottom", fontsize=8.5, color=INK2)
    # feedback 2: RGB-D + executed increments -> RSSM posterior
    ax.add_patch(FancyArrowPatch((8.6, yb2 + hb2), (11.3, yb), connectionstyle="arc3,rad=-0.25", arrowstyle="-|>",
                                 mutation_scale=13, color=AQUA, lw=1.6))
    ax.text(8.3, 4.12, "RGB-D 64 x 64 + the 5 executed\nincrements -> update the RSSM\nstate (the model's memory)", ha="right",
            va="bottom", fontsize=8.3, color=AQUA, weight="bold", linespacing=1.3)
    ax.text(0.2, 1.45, "Oracle verifier (upper bound): the same loop, but the score comes from privileged simulator information\n"
            "(true shortest-path progress, true clearance, true 'at goal').  Learned verifier: the world model above.\n"
            "Selection rules: wm = stop only when the policy proposes it and the model agrees;  wm_shield = keep the policy's\n"
            "own chunk unless the model predicts a collision or low clearance.", fontsize=9.3, color=INK, va="top", linespacing=1.55)
    save(fig, out, "fig12_wm_arch.png")


def fig_wm_step(out):
    """docs/report_v1.md: what one world-model step (0.5 s) contains and where the labels come from."""
    fig, ax = canvas(16.4, 6.9)
    y_ax = 3.7
    ax.plot([0.8, 15.6], [y_ax, y_ax], color=INK2, lw=1.6)
    ax.add_patch(FancyArrowPatch((15.3, y_ax), (15.8, y_ax), arrowstyle="-|>", mutation_scale=14, color=INK2, lw=1.6))
    ticks = {3.0: "t - 0.5 s\n(decision k - 1)", 8.0: "t\n(decision k)", 13.0: "t + 0.5 s\n(decision k + 1)"}
    for x, lab in ticks.items():
        ax.plot([x, x], [y_ax - 0.15, y_ax + 0.15], color=INK2, lw=1.6)
        ax.text(x, y_ax - 0.3, lab, ha="center", va="top", fontsize=9, color=INK2, linespacing=1.3)
    for x in (3.0, 8.0, 13.0):
        box(ax, x - 1.25, y_ax + 0.45, 2.5, 1.55, "Observation", "RGB-D 64 x 64 (4 ch)\nstate: velocity (3),\nyaw rate, height",
            fc=TINT_BLUE, ec=BLUE, body_size=8.2)
    for x0, x1 in ((3.0, 8.0), (8.0, 13.0)):
        xm = (x0 + x1) / 2
        ax.add_patch(FancyArrowPatch((x0 + 1.3, y_ax + 1.2), (x1 - 1.3, y_ax + 1.2), arrowstyle="-|>", mutation_scale=12,
                                     color=ORANGE, lw=1.6))
        ax.text(xm, y_ax + 1.3, "action: 5 increments\n[dx, dy, dz, dyaw, stop] x 5\n(the first 0.5 s of a chunk)", ha="center",
                va="bottom", fontsize=8.3, color=ORANGE, weight="bold", linespacing=1.3)
    ax.text(8.2, 6.4, "RSSM:  h(k+1) = f(h(k), z(k), a(k));   z from the observation (training, posterior)  or  from h alone "
            "(imagination, prior)", ha="center", va="top", fontsize=9.3, color=INK, weight="bold")
    # labels at k+1
    box(ax, 9.3, 0.3, 6.9, 2.2, "Labels at t + 0.5 s  (the outcome of action k)",
        "progress  = shortest-path distance(t) - distance(t + 0.5 s)   [m]\n"
        "collision  = hit a surface during the step (< 0.12 m)         [0/1]\n"
        "clearance = smallest distance to any surface during the step [m]\n"
        "target visible = target covers >= 150 pixels                  [0/1]\n"
        "at goal    = <= 1 m from the goal view and target visible      [0/1]",
        fc="#ffffff", ec=AQUA, body_size=8.4)
    arrow(ax, (13.0, y_ax - 1.0), (13.0, 2.5), color=AQUA)
    box(ax, 0.3, 0.3, 8.3, 2.2, "Where the steps come from",
        "expert dataset: 1,723 episodes -> 52k steps (every 5th frame); no collisions\n"
        "VLA's own closed-loop flights (closed_loop --dump-wm): one step per decision,\n"
        "a collision ends the episode with a terminal step (collision = 1)\n"
        "shortest-path distance and clearance come from the 3D map of the house",
        fc=TINT_GRAY, ec=MUTED, body_size=8.4)
    save(fig, out, "fig13_wm_step.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figs = {"pipeline": fig_pipeline, "sample": fig_sample, "action": fig_action, "model": fig_model, "loop": fig_loop,
            "training": fig_training, "results": fig_results, "episodes": fig_episodes,
            "wm_arch": fig_wm_arch, "wm_step": fig_wm_step}
    for k, fn in figs.items():
        if args.only is None or k in args.only:
            fn(out)


if __name__ == "__main__":
    main()
