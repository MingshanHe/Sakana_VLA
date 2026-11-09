# Sakana_VLA

Indoor drone VLA: dataset generation in HyperDrone (ProcTHOR houses, L2F dynamics, hyperlux rendering)
and the training-side loader. The data format follows OpenFly v2 / [AeroBrain](https://huggingface.co/datasets/Ideallll/AeroBrain).
The data lives in [hemingshan/vla_dataset](https://huggingface.co/datasets/hemingshan/vla_dataset).

## Design (v0, small datasets)

| | |
|---|---|
| Scenes | ProcTHOR Train200 houses (hyperlux / conta cache) |
| Rate / speed | 10 Hz recording, 100 Hz physics, 0.5 m/s cruise, 0.5 m/s² accel |
| Observation | monocular RGB 256×256 + depth (uint16 mm), HFOV 90°, body-mounted camera (roll and pitch from the dynamics are visible) |
| Action (per frame) | `[dx, dy, dz, dyaw, stop]` in the yaw frame of the current frame (`actions`). An AeroBrain-compatible `[dx, dy, dz, stop]` is in `actions_aerobrain` |
| Action chunk | built at load time: H future poses relative to the current frame (default H=10, i.e. 1 s). Execute k steps (e.g. 5), then re-infer (replan / correction) |
| Expert | geometric plan (Dijkstra → shortcut → spline → trapezoid speed, yaw along travel, then faces the target) tracked by **RAPTOR position control on L2F crazyflie dynamics**. The setpoint uses a 0.55 s lookahead and a +5.3 cm z-sag feed-forward. Labels come from the realized state, not the plan |
| Recovery data | 50 % of episodes get a mid-flight lateral kick (0.5–1 m/s) plus a yaw kick (15–45°). The expert holds the stale setpoint for 0.4 s, then replans from the disturbed state (`perturbed` flag) |
| Instructions | `instructions/raw.jsonl` holds the template drafts. Structured metadata (target, rooms, primitive segments, landmarks seen) is in `meta/episodes_extra.jsonl`, and keyframes are in `keyframes/`. The LLM rewrite and human check go to `llm_v1.jsonl` / `verified_v1.jsonl` |

## Pipeline (`datagen/`)

| file | role |
|---|---|
| `scene_parse.py` | GLB → objects (type, `segmentation_class`, world AABB, room) and rooms (type inferred from contents). Writes `class_labels.json` |
| `geometry.py` | triangles → 0.1 m occupancy, clearance (EDT), house-interior mask, flight mask (0.5–2.0 m above the floor, clearance ≥ 0.3 m) |
| `tasks.py` | target (class unique in the house, large enough) → goal viewpoint (standoff, line of sight, in FOV) → starts (geodesic 3–12 m) |
| `planner.py` | `GoalField` (one Dijkstra per goal; it serves all starts and the replans), smoothing, time and yaw profiles |
| `expert_rollout.py` | batched dynamics expert with perturbation and replanning, collision/tracking checks |
| `render.py` | hyperlux RGB, depth and semantic segmentation → target bbox |
| `label.py` | actions, stop (near the goal and facing the target, trailing run), motion primitives, room sequence, instruction draft |
| `write_dataset.py` | LeRobot v2.1 writer (AV1 video, depth npz, meta, instructions, keyframes) |
| `generate.py` | builds a subset end to end with QA filtering |
| `spike.py` | single-house diagnostics (top-down plot, overlay videos, stats) |

```bash
conda activate sakana_hyperdrone            # PATH must put the env's cmake first
export PYTHONPATH=$PWD
python -m datagen.scene_parse ProcTHOR-Train-{1..92} --build-labels      # once
python -m datagen.generate --subset indoor_nano --houses 1,3,4,5,6 --episodes-per-house 40 --out data/vla_dataset
# Parallel per-house parts + merge (what built indoor_nano):
#   bash scripts/run_part.sh indoor_nano <house> 40 <seed>   (HYPERLUX_BACKEND=OPTIX on GPU: ~90 s / house)
#   python -m datagen.merge_subsets --parts data/vla_parts/indoor_nano_h* --out data/vla_dataset/indoor_nano
# On a GPU node the renderer uses OptiX; on CPU nodes set HYPERLUX_BACKEND=GENERIC (~4 frames/s on 4 cores).
```

Loader: `vla_data/indoor_dataset.py` (`IndoorVLADataset(root, horizon=10)` returns RGB, depth, instruction, state, per-step action and chunk).

## Dataset layout (per subset, LeRobot v2.1)

```
meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl, episodes_extra.jsonl, generation_report.json}
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.rgb/episode_000000.mp4
depth/chunk-000/episode_000000.npz            # "depth": (T,H,W) uint16 mm
instructions/raw.jsonl
keyframes/episode_000000/*.jpg
```

Per-frame columns:
- `observation.states` `[x,y,z,yaw]`, relative to the start pose, in the start yaw frame.
- `observation.state_full` (world position, quaternion wxyz, world velocity, body rates).
- `observation.boxes` (normalized target bbox x,y,w,h; zeros when the target is not visible) and `observation.target_pixels`.
- `actions`, `actions_aerobrain`.
- `caption` (the instruction), `primitive`, `task_index` (index into the 13 primitives), `phase`, `perturbed`, `env` (house).

## Known caveats

- On fresh cluster nodes, hyperlux's JIT configure can stall when its output is piped. `datagen/_jitfix.py` streams the output instead; import it first.
- RAPTOR position control builds only for CPU dynamics. Batching is cheap: about 12 ms per step for 1 or 64 drones.
- Room types are inferred from contents, because the GLBs carry no roomType.

## VLA (`sakana_vla/`)

v0 follows OpenFlyV2 without MAM and MSP: a Qwen3.5-0.8B prefix plus a 233M flow-matching action Expert in pi0 style. Results are in [docs/v0_results.md](docs/v0_results.md) (bilingual).

| file | role |
|---|---|
| `sakana_vla/model/expert.py` | Expert decoder (Qwen3.5 hybrid layers, 640 wide). `PrefixKV` is a read-only view of the VLM cache |
| `sakana_vla/model/vla.py` | prefix encoding, flow-matching loss (Beta(1.5,1) time), Euler sampling |
| `sakana_vla/data/` | frame memmap cache, chunk targets (H=12, `[dx,dy,dz,dyaw,stop]` in the current yaw frame), normalization, collator |
| `sakana_vla/train.py` | training (stage A: frozen VLM; `--vlm-lr` > 0 for stage B) and offline ADE/FDE/yaw/stop metrics |
| `sakana_vla/eval/closed_loop.py` | closed loop in HyperDrone (env `sakana_hyperdrone`). Spawns `policy_server.py` (env `sakana_vla2`) on a node-local Unix socket. Options: `--oracle-replay` (executor check), `--server-args=--blind` / `--shuffle-instructions` (ablations), `--record` (10 Hz frames for videos) |

```bash
conda activate sakana_vla2   # py3.11, torch 2.7.1 cu126, transformers 5.5.0, flash-linear-attention
PYTHONNOUSERSITE=1 python -m sakana_vla.train --cache data/cache/v1 --out runs/v1_stageA --steps 16000 --bs 64
```

## Third-party code

`sakana_vla/model/expert.py` adapts the action Expert of [AeroBrain-project](https://github.com/Ideal-ljl/AeroBrain-project), Copyright 2026 Junli Liu, Apache-2.0. That code builds on Qwen3.5 modeling code from transformers, Copyright The Qwen Team and The HuggingFace Inc. team, Apache-2.0.
