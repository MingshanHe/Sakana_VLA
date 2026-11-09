---
license: other
task_categories:
- robotics
tags:
- vision-language-action
- drone
- uav
- indoor-navigation
- lerobot
- procthor
pretty_name: Sakana Indoor Drone VLA
---

# Sakana Indoor Drone VLA dataset

Small synthetic **indoor** drone vision-language-navigation datasets for fast VLA architecture checks. They are
generated with [HyperDrone](https://pypi.org/project/hyperdrone/): L2F quadrotor dynamics, the RAPTOR position
controller, and the hyperlux ray tracer, in ProcTHOR houses. The layout follows OpenFly v2
[AeroBrain](https://huggingface.co/datasets/Ideallll/AeroBrain): LeRobot v2.1, one folder per subset.
Generation code: [MingshanHe/Sakana_VLA](https://github.com/MingshanHe/Sakana_VLA) (`datagen/`).

## Subsets

| subset | houses | episodes | purpose |
|---|---|---|---|
| `indoor_nano` | ProcTHOR-Train 1, 3, 4, 5, 6 | 184 (30,535 frames, 0.85 h) | overfit / pipeline debugging |

## Episode

The drone starts at a random free pose 3–12 m (geodesic) from a goal viewpoint. A dynamics expert flies there at 0.5 m/s,
turns to face the target object, and hovers with `stop=1`. In about 50 % of episodes a mid-flight disturbance is injected
(lateral velocity kick plus yaw kick). The expert then replans from the disturbed state (`perturbed=True` frames),
which gives deviation-and-recovery data for chunked replanning.

- 10 Hz frames: RGB 256×256 (AV1) and depth (uint16 mm, `depth/*.npz`), HFOV 90°, body-mounted forward camera.
- `actions` = `[dx, dy, dz, dyaw, stop]` in the yaw frame of the current frame, from the **realized** trajectory.
  `actions_aerobrain` = `[dx, dy, dz, stop]` (AeroBrain-compatible).
- Action chunks are built at load time from `observation.states`. They are cumulative future poses relative to the current frame;
  see `vla_data/indoor_dataset.py`.
- `observation.boxes`: normalized target bbox from semantic segmentation.
- `caption`: instruction. v0 is a **template draft**. The LLM rewrite and human verification will be added as
  `instructions/llm_v1.jsonl` and `instructions/verified_v1.jsonl`.
- `meta/episodes_extra.jsonl`: target (type, AABB, room), start and goal, room sequence, primitive segments, landmarks seen,
  tracking error, and path length. `keyframes/` holds the start, segment-boundary, and end frames for annotation.
- 13 motion primitives (`meta/tasks.jsonl`, `primitive` column): AeroBrain's 7 plus rotate left/right, ascend, descend, hover, and stop.

## Versions

- `v0.1-raw`: template instructions only.
