"""VLA policy server (env: sakana_vla2). Serves chunk predictions to the closed-loop simulator (env: sakana_hyperdrone).

    PYTHONNOUSERSITE=1 python -m sakana_vla.eval.policy_server --run runs/v0_nano_stageA --socket /tmp/sakana_vla.sock
Listens on a node-local Unix domain socket only (no network port). Normally spawned by closed_loop.py.
Request: {"images": (B,H,W,3) uint8, "instructions": [str], "states": (B,5) raw state, "seeds": [int] | None,
          "n_samples": N (default 1)} -> (B, H, 5) raw increments, or (B, N, H, 5) candidates when N > 1.
"""
import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

from ..data.dataset import Collator, denormalize_actions
from ..model.vla import SakanaVLA

AUTHKEY = b"sakana-vla"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--socket", required=True, help="Unix domain socket path (node-local)")
    ap.add_argument("--denoise-steps", type=int, default=5)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--shuffle-instructions", action="store_true", help="language ablation: permute within batch")
    args = ap.parse_args()
    run = Path(args.run)
    ckpt = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    targs, stats = ckpt["args"], ckpt["stats"]
    model = SakanaVLA(targs["vlm"], horizon=targs["horizon"], freeze_vlm=True).cuda()
    model.expert.load_state_dict(ckpt["expert"])
    if "vlm" in ckpt:
        model.vlm.load_state_dict(ckpt["vlm"])
    model.eval()
    collate = Collator(AutoProcessor.from_pretrained(targs["vlm"]))
    s_mean, s_std = np.asarray(stats["state_mean"], np.float32), np.asarray(stats["state_std"], np.float32)
    rng = np.random.default_rng(0)
    import os
    if os.path.exists(args.socket):
        os.remove(args.socket)
    with Listener(args.socket, family="AF_UNIX", authkey=AUTHKEY) as listener:
        print(f"serving {run} (step {ckpt['step']}) on {args.socket}", flush=True)
        while True:
            conn = listener.accept()
            try:
                while True:
                    req = conn.recv()
                    if req == "close":
                        return
                    instr = list(req["instructions"])
                    if args.shuffle_instructions and len(instr) > 1:
                        instr = [instr[i] for i in rng.permutation(len(instr))]
                    imgs = req["images"] if not args.blind else np.zeros_like(req["images"])
                    items = [{"image": im, "instruction": tx, "state": (st - s_mean) / s_std,
                              "actions": np.zeros((targs["horizon"], 5), np.float32),
                              "raw_actions": np.zeros((targs["horizon"], 5), np.float32), "index": 0}
                             for im, tx, st in zip(imgs, instr, np.asarray(req["states"], np.float32))]
                    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in collate(items).items()}
                    n_samples = int(req.get("n_samples", 1))
                    noise = None
                    if req.get("seeds") is not None:  # per-(episode, candidate) fixed source noise: consistent modes across replans
                        noise = torch.stack([torch.randn(targs["horizon"], 5, generator=torch.Generator().manual_seed(int(sd) + 7919 * k))
                                             for sd in req["seeds"] for k in range(n_samples)]).cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        x = model.sample(batch, steps=args.denoise_steps, noise=noise, n_samples=n_samples)
                    conn.send(denormalize_actions(x.float().cpu().numpy(), stats))
            except EOFError:
                pass
            finally:
                conn.close()


if __name__ == "__main__":
    main()
