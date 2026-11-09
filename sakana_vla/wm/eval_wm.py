"""Offline verifier-quality check of a chunk world model on a replay directory (env: sakana_vla2).

    PYTHONNOUSERSITE=1 python -m sakana_vla.wm.eval_wm --ckpt runs/wm_v1/best.pt --data data/wm/onpolicy_v0_unseen
For every episode: run the posterior over the logged observations, then at each step imagine ONE step with the
logged action and compare the predicted outcomes with what happened: collision AUC (predicted 1 - cont vs the
next step being terminal), clearance MAE / correlation, progress MAE / correlation, target-visible and at-goal accuracy.
"""
import argparse
import json

import numpy as np
import torch

from .dv3.nets import TwoHot
from .labels import scale_actions
from .model import ChunkWorldModel
from .replay import load_files


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    wm = ChunkWorldModel(**ck["cfg"]).to(dev).eval()
    wm.load_state_dict(ck["wm"])
    th = TwoHot(device=dev)
    P = {k: [] for k in ("col_pred", "col_true", "clr_pred", "clr_true", "prog_pred", "prog_true", "vis_pred", "vis_true",
                         "goal_pred", "goal_true")}
    n_eps = 0
    for f in load_files(args.data):
        with np.load(f) as d:
            img = torch.as_tensor(d["image"], device=dev)[None]
            vec = torch.as_tensor(d["vector"], device=dev)[None].float()
            act = torch.as_tensor(scale_actions(d["action"]), device=dev)[None]
            first = torch.as_tensor(d["is_first"], device=dev)[None]
            T = img.shape[1]
            with torch.autocast(dev, torch.bfloat16, enabled=dev == "cuda"):
                tokens = wm.enc(img.flatten(0, 1), vec.flatten(0, 1)).reshape(1, T, -1)
                out, _ = wm.rssm.observe(tokens, act, first)
                for t in range(T - 1):  # posterior at t, imagine a_t -> predicted outcome at t+1
                    st = {"deter": out["deter"][:, t], "stoch": out["stoch"][:, t]}
                    st = wm.rssm.img_step(st, act[:, t].to(st["deter"].dtype))
                    o = wm.outcomes(wm.feat(st), th)
                    P["col_pred"].append(float(1 - o["cont"][0])); P["col_true"].append(bool(d["is_terminal"][t + 1]))
                    P["clr_pred"].append(float(o["clearance"][0])); P["clr_true"].append(float(min(d["clearance"][t + 1], 1.0)))
                    P["prog_pred"].append(float(o["progress"][0])); P["prog_true"].append(float(d["reward"][t + 1]))
                    P["vis_pred"].append(float(o["vis"][0])); P["vis_true"].append(bool(d["target_vis"][t + 1]))
                    P["goal_pred"].append(float(o["at_goal"][0])); P["goal_true"].append(bool(d["at_goal"][t + 1]))
        n_eps += 1
    A = {k: np.asarray(v) for k, v in P.items()}
    res = {"episodes": n_eps, "steps": int(len(A["col_true"])), "collisions": int(A["col_true"].sum()),
           "collision_auc": auc(A["col_pred"][A["col_true"]], A["col_pred"][~A["col_true"]]),
           "collision_pred_mean_on_terminal": float(A["col_pred"][A["col_true"]].mean()) if A["col_true"].any() else float("nan"),
           "collision_pred_mean_safe": float(A["col_pred"][~A["col_true"]].mean()),
           "low_clearance_auc(<0.2)": auc(A["clr_pred"][A["clr_true"] < 0.2] * -1, A["clr_pred"][A["clr_true"] >= 0.2] * -1),
           "clearance_mae": float(np.abs(A["clr_pred"] - A["clr_true"]).mean()),
           "clearance_corr": float(np.corrcoef(A["clr_pred"], A["clr_true"])[0, 1]),
           "progress_mae": float(np.abs(A["prog_pred"] - A["prog_true"]).mean()),
           "progress_corr": float(np.corrcoef(A["prog_pred"], A["prog_true"])[0, 1]),
           "vis_acc": float(((A["vis_pred"] > 0.5) == A["vis_true"]).mean()),
           "goal_acc": float(((A["goal_pred"] > 0.5) == A["goal_true"]).mean()),
           "goal_auc": auc(A["goal_pred"][A["goal_true"]], A["goal_pred"][~A["goal_true"]])}
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
