"""Train the chunk world model (env: sakana_vla2, GPU).

    PYTHONNOUSERSITE=1 python -m sakana_vla.wm.train_wm --data data/wm/expert [data/wm/onpolicy_*] --out runs/wm_v0 \
        --size size50m --steps 20000
Validation = episodes of the unseen houses (subset indoor_mini_unseen by default). Open-loop check: observe `ctx` steps,
imagine `horizon` steps under the logged actions, compare predicted progress / clearance / visibility / collision with
what really happened.
"""
import argparse
import json
import math
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch

from .dv3.nets import TwoHot, symexp
from .model import ChunkWorldModel
from .replay import Replay, file_meta, load_files, to_torch


def prefetch(replay, b, t, n=6):
    q = queue.Queue(n)

    def work():
        while True:
            q.put(replay.sample(b, t))
    for _ in range(2):
        threading.Thread(target=work, daemon=True).start()
    return q


@torch.no_grad()
def evaluate(wm, twohot, val, device, batches=16, b=16, ctx=4, horizon=4):
    wm.eval()
    st = val.rng.bit_generator.state
    val.rng = np.random.default_rng(123)
    agg, ol = {}, {k: [] for k in ("prog_mae", "prog_sign_acc", "clr_mae", "vis_acc", "goal_acc", "col_pos", "col_neg")}
    for _ in range(batches):
        batch = to_torch(val.sample(b, ctx + horizon), device)
        with torch.autocast("cuda", torch.bfloat16):
            _, m, _, _ = wm.loss(batch, twohot)
            for k, v in m.items():
                agg.setdefault(k, []).append(float(v))
            c = {k: v[:, :ctx] for k, v in batch.items()}
            tokens = wm.enc(c["image"].flatten(0, 1), c["vector"].flatten(0, 1)).reshape(b, ctx, -1)
            _, (state, prev_a) = wm.rssm.observe(tokens, c["action"], c["is_first"])
            alive = torch.ones(b, dtype=torch.bool, device=device)
            for k in range(horizon):
                a = batch["action"][:, ctx + k - 1] if k else prev_a
                state = wm.rssm.img_step(state, a.to(state["deter"].dtype))
                o = wm.outcomes(wm.feat(state), twohot)
                alive &= ~batch["is_first"][:, ctx + k]
                r = batch["reward"][:, ctx + k]
                ol["prog_mae"].append((o["progress"] - r).abs()[alive].cpu().numpy())
                ol["prog_sign_acc"].append(((o["progress"] > 0.02) == (r > 0.02))[alive].float().cpu().numpy())
                ol["clr_mae"].append((o["clearance"] - batch["clearance"][:, ctx + k].clamp(0, 1)).abs()[alive].cpu().numpy())
                ol["vis_acc"].append(((o["vis"] > 0.5) == batch["target_vis"][:, ctx + k])[alive].float().cpu().numpy())
                ol["goal_acc"].append(((o["at_goal"] > 0.5) == batch["at_goal"][:, ctx + k])[alive].float().cpu().numpy())
                term = batch["is_terminal"][:, ctx + k]
                ol["col_pos"].append((1 - o["cont"])[alive & term].cpu().numpy())
                ol["col_neg"].append((1 - o["cont"])[alive & ~term].cpu().numpy())
                alive &= ~term
    val.rng.bit_generator.state = st
    res = {f"val_{k}": float(np.mean(v)) for k, v in agg.items()}
    for k in ("prog_mae", "prog_sign_acc", "clr_mae", "vis_acc", "goal_acc"):
        res[f"ol_{k}"] = float(np.mean(np.concatenate(ol[k])))
    pos, neg = np.concatenate(ol["col_pos"]), np.concatenate(ol["col_neg"])
    if len(pos) and len(neg):
        res["ol_col_auc"] = float((pos[:, None] > neg[None, :]).mean())
        res["ol_collisions"] = int(len(pos))
    wm.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="size50m")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-subset", default="indoor_mini_unseen")
    ap.add_argument("--val-houses", nargs="*", default=[])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = load_files(args.data)
    val_f, train_f = [], []
    for f in files:
        meta = file_meta(f)
        (val_f if meta.get("subset") == args.val_subset or meta.get("house") in args.val_houses else train_f).append(f)
    t0 = time.time()
    train, val = Replay(train_f, args.seed), Replay(val_f, args.seed + 1)
    print(f"replay: train {len(train_f)} eps / {train.n} steps, val {len(val_f)} eps / {val.n} steps, loaded {time.time() - t0:.0f}s",
          flush=True)
    device = "cuda"
    wm = ChunkWorldModel(size=args.size).to(device)
    if args.compile:
        wm.rssm.compile()
    twohot = TwoHot(device=device)
    n_par = sum(p.numel() for p in wm.parameters())
    print(f"world model {args.size}: {n_par / 1e6:.1f}M params", flush=True)
    opt = torch.optim.Adam(wm.parameters(), lr=args.lr, eps=1e-8)
    q = prefetch(train, args.batch, args.seq)
    log = open(out / "log.jsonl", "a")
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    best = math.inf
    t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, step / args.warmup)
        batch = to_torch(q.get(), device)
        with torch.autocast("cuda", torch.bfloat16):
            loss, m, _, _ = wm.loss(batch, twohot)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(wm.parameters(), 1000.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            rec = {"step": step, **{k: round(float(v), 5) for k, v in m.items()}, "gnorm": round(float(gn), 2),
                   "it_s": round(step / (time.time() - t0), 2)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.eval_every == 0 or step == args.steps:
            res = {"step": step, **evaluate(wm, twohot, val, device)}
            print("EVAL " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
            log.write(json.dumps(res) + "\n")
            log.flush()
            ck = {"wm": wm.state_dict(), "cfg": wm.cfg, "step": step, "args": vars(args), "eval": res}
            torch.save(ck, out / "last.pt")
            if res["val_loss"] < best:
                best = res["val_loss"]
                torch.save(ck, out / "best.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
