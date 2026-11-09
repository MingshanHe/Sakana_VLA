"""Train SakanaVLA v0 (stage A: frozen Qwen3.5-0.8B, flow-matching Expert only).

    PYTHONNOUSERSITE=1 python -m sakana_vla.train --cache data/cache/v0 --out runs/v0_stageA --steps 20000
    ... --overfit-batch          # sanity: one fixed batch, loss -> 0, sampled chunk -> target
Offline metrics on val_seen / val_unseen: ADE / FDE (m) of cumulative waypoints, final yaw error (deg),
stop F1 (any stop in chunk), against a constant-velocity baseline.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

from .data.dataset import (Collator, FrameCache, VLADataset, compute_norm_stats, denormalize_actions,
                           split_indices, state_vector)
from .model.vla import SakanaVLA


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def chunk_metrics(pred_inc, gt_inc):
    """pred/gt (B, H, 5) raw increments -> dict of per-sample arrays."""
    pc, gc = np.cumsum(pred_inc[..., :3], 1), np.cumsum(gt_inc[..., :3], 1)
    d = np.linalg.norm(pc - gc, axis=-1)
    yaw_err = np.degrees(np.abs(np.cumsum(pred_inc[..., 3], 1)[:, -1] - np.cumsum(gt_inc[..., 3], 1)[:, -1]))
    return {"ade": d.mean(1), "fde": d[:, -1], "yaw_fde_deg": yaw_err,
            "stop_pred": (pred_inc[..., 4] > 0.5).any(1), "stop_gt": (gt_inc[..., 4] > 0.5).any(1)}


def const_velocity(cache, indices, horizon, dt=0.1):
    out = []
    for i in indices:
        v = state_vector(cache, int(i))
        inc = np.zeros((horizon, 5), np.float32)
        inc[:, 0], inc[:, 1], inc[:, 2], inc[:, 3] = v[0] * dt, v[1] * dt, v[2] * dt, v[3] * dt
        out.append(inc)
    return np.stack(out)


@torch.no_grad()
def evaluate(model, loader, stats, cache, device, steps=5, max_batches=8):
    model.eval()
    agg = {}
    for b, batch in enumerate(loader):
        if b >= max_batches:
            break
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = model.sample(batch, steps=steps)
        pred = denormalize_actions(x.float().cpu().numpy(), stats)
        gt = batch["raw_actions"].cpu().numpy()
        cv = const_velocity(cache, batch["index"].cpu().numpy(), gt.shape[1])
        for name, p in (("model", pred), ("constvel", cv)):
            for k, v in chunk_metrics(p, gt).items():
                agg.setdefault(f"{name}_{k}", []).append(v)
    model.train()
    out = {}
    for k, v in agg.items():
        v = np.concatenate(v)
        if k.endswith("stop_pred"):
            continue
        out[k] = float(v.mean()) if not k.endswith("stop_gt") else None
    sp, sg = np.concatenate(agg["model_stop_pred"]), np.concatenate(agg["model_stop_gt"])
    tp = (sp & sg).sum()
    out["model_stop_f1"] = float(2 * tp / max(1, sp.sum() + sg.sum()))
    out["n"] = int(len(sp))
    return {k: v for k, v in out.items() if v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--vlm", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vlm-lr", type=float, default=0.0, help=">0 unfreezes the VLM (stage B)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--stop-weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--denoise-steps", type=int, default=5)
    ap.add_argument("--overfit-batch", action="store_true")
    ap.add_argument("--blind", action="store_true", help="black images (vision ablation)")
    ap.add_argument("--unseen-houses", nargs="*", default=[], help="houses held out as val_unseen")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    cache = FrameCache(args.cache)
    train_idx, val_seen, val_unseen = split_indices(cache, unseen_houses=tuple(args.unseen_houses), seed=args.seed)
    stats = compute_norm_stats(cache, train_idx, args.horizon)
    (out / "norm_stats.json").write_text(json.dumps(stats, indent=1))
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    print(f"frames: train {len(train_idx)} val_seen {len(val_seen)} val_unseen {len(val_unseen)}; "
          f"action bound {np.round(stats['action_bound'], 4)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.vlm)
    collate = Collator(processor)
    mk = lambda idx, shuffle, bs: torch.utils.data.DataLoader(
        VLADataset(cache, idx, stats, args.horizon, blind=args.blind), batch_size=bs, shuffle=shuffle,
        num_workers=args.workers, collate_fn=collate, drop_last=shuffle, persistent_workers=shuffle and args.workers > 0,
        pin_memory=True)
    rng = np.random.default_rng(args.seed)
    ev_seen = mk(rng.choice(val_seen, min(len(val_seen), args.eval_batches * args.bs), replace=False), False, args.bs)
    ev_unseen = (mk(rng.choice(val_unseen, min(len(val_unseen), args.eval_batches * args.bs), replace=False), False, args.bs)
                 if len(val_unseen) else None)

    model = SakanaVLA(args.vlm, horizon=args.horizon, freeze_vlm=args.vlm_lr <= 0, stop_weight=args.stop_weight).to(device)
    n_exp = sum(p.numel() for p in model.expert.parameters())
    n_vlm = sum(p.numel() for p in model.vlm.parameters())
    print(f"params: expert {n_exp / 1e6:.1f}M, vlm {n_vlm / 1e6:.1f}M (frozen={model.freeze_vlm})", flush=True)
    groups = [{"params": [p for p in model.expert.parameters()], "lr": args.lr}]
    if not model.freeze_vlm:
        groups.append({"params": [p for p in model.vlm.parameters() if p.requires_grad], "lr": args.vlm_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in groups]

    if args.overfit_batch:
        fixed = next(iter(mk(train_idx[:: max(1, len(train_idx) // args.bs)][:args.bs], False, args.bs)))
        batches = iter(lambda: fixed, None)
    else:
        train_loader = mk(train_idx, True, args.bs)

        def gen():
            while True:
                yield from train_loader
        batches = gen()

    log = open(out / "log.jsonl", "a")
    model.train()
    t0, ema = time.time(), None
    for step in range(1, args.steps + 1):
        lr_scale = min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * lr_scale
        batch = to_device(next(batches), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, m = model(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 50 == 0 or step == 1:
            rec = {"step": step, "loss": loss.item(), "loss_ema": ema, "grad_norm": gn.item(), "lr": opt.param_groups[0]["lr"],
                   "it_s": step / (time.time() - t0), **m}
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.eval_every == 0 or step == args.steps:
            if args.overfit_batch:
                ev = evaluate(model, [fixed], stats, cache, device, args.denoise_steps, 1)
                res = {"step": step, "overfit": ev}
            else:
                res = {"step": step, "val_seen": evaluate(model, ev_seen, stats, cache, device, args.denoise_steps,
                                                          args.eval_batches)}
                if ev_unseen is not None:
                    res["val_unseen"] = evaluate(model, ev_unseen, stats, cache, device, args.denoise_steps,
                                                 args.eval_batches)
            print("EVAL " + json.dumps(res), flush=True)
            log.write(json.dumps(res) + "\n")
            log.flush()
            ckpt = {"expert": model.expert.state_dict(), "stats": stats, "args": vars(args), "step": step}
            if not model.freeze_vlm:
                ckpt["vlm"] = model.vlm.state_dict()
            torch.save(ckpt, out / "checkpoint.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
