"""Shared definitions for the chunk world model (CPU only: numpy + cv2).

One world-model step = WM_STEP frames = 0.5 s, i.e. one policy decision. The observation is the 64x64 RGB-D frame at
the decision, the vector is the policy's 5-dim state, and the action is the 5 increments executed until the next
decision, in the yaw frame of the decision frame (exactly the first 5 steps of a policy chunk).
"""
import cv2
import numpy as np

WM_STEP = 5
IMG = 64
DEPTH_MAX = 10.0      # m; depth channel = clip(d, 0, 10) / 10 * 255
VIS_PIXELS = 150      # target counts as visible (same as the success criterion)
GOAL_DIST = 1.0       # m to the goal viewpoint (same as the success criterion)
CLR_CLIP = 1.0        # clearance label is clipped to [0, 1] m
# increments are ~0.05 m / ~0.1 rad per frame; the RSSM expects O(1) actions -> fixed scale applied at load/score time
ACTION_SCALE = np.tile(np.array([20.0, 20.0, 20.0, 8.0, 1.0], np.float32), WM_STEP)


def scale_actions(a):
    return np.asarray(a, np.float32) * ACTION_SCALE


def rgbd64(rgb, depth_m):
    """rgb (256,256,3) uint8, depth (256,256) metres (float) or millimetres (uint16) -> (64,64,4) uint8."""
    d = depth_m.astype(np.float32)
    if depth_m.dtype == np.uint16:
        d = d / 1000.0
    d = np.nan_to_num(d, nan=DEPTH_MAX, posinf=DEPTH_MAX, neginf=0.0)
    small_rgb = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)
    small_d = cv2.resize(d, (IMG, IMG), interpolation=cv2.INTER_AREA)
    d8 = (np.clip(small_d, 0, DEPTH_MAX) / DEPTH_MAX * 255).astype(np.uint8)
    return np.concatenate([small_rgb, d8[..., None]], -1)


def clearance_at(maps, positions):
    """Clearance (m) of each world position; 0 where outside the grid."""
    g, clr = maps["grid"], maps["clearance"]
    ijk = g.to_index(np.atleast_2d(positions))
    ok = g.valid(ijk)
    out = np.zeros(len(ijk), np.float32)
    out[ok] = clr[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    return out


def chunk_to_wm_actions(chunk, n_steps=2):
    """(H, 5) chunk -> (n_steps, 25) world-model actions (consecutive groups of WM_STEP increments)."""
    H = chunk.shape[0]
    out = np.zeros((n_steps, WM_STEP * 5), np.float32)
    for s in range(n_steps):
        seg = chunk[s * WM_STEP:(s + 1) * WM_STEP]
        if len(seg) < WM_STEP:  # pad with hover
            seg = np.concatenate([seg, np.zeros((WM_STEP - len(seg), 5), np.float32)], 0)
        out[s] = seg.reshape(-1)
    return out
