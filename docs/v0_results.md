# SakanaVLA v0: first feasibility run / 第一版可行性实验 (2026-11-09)

**TL;DR (EN).** The OpenFlyV2 recipe minus MAM and MSP is Qwen3.5-0.8B plus a 233M flow-matching action Expert. Stage A trains the Expert with the VLM frozen. With 45 ProcTHOR houses it learns accurate short-horizon motion: on unseen houses ADE@1.2s is 3.2 cm, against 6.0 cm for constant velocity. The closed-loop executor is validated (expert replay reaches SR 100%). End-to-end closed-loop navigation still **fails** (SR 0 on unseen houses). Low-level flight through doors is plausible. Goal selection fails: wrong room or wrong object, stops at the start, and the policy collides once it drifts. The likely causes are a single frame with no history, a frozen VLM (weak grounding), and template instructions. The v1 changes below address these.

**结论（中文）。** v0 采用 OpenFlyV2 的方案，去掉了 MAM 和 MSP：Qwen3.5-0.8B 加一个 233M 参数的 flow-matching 动作专家，VLM 冻结，只训练 expert。在 45 栋 ProcTHOR 房子上训练后，短时动作预测已经很准：未见房子上 1.2 s 内的 ADE 为 3.2 cm，匀速外推基线是 6.0 cm。闭环执行器也验证过了，用专家轨迹回放可以达到 SR 100%。但端到端闭环导航**仍然失败**，未见房子上 SR 为 0。底层飞行（穿门、进房间）基本合理，失败出在目标选择上：去错房间、找错物体、起点就 stop，偏离路线后发生碰撞。主要原因有三点：只用单帧、没有历史信息；VLM 冻结导致语言与视觉的 grounding 不足；指令只是模板。下一版的改进见文末。

## Setup / 设置

| item | value |
|---|---|
| Backbone | `Qwen/Qwen3.5-0.8B` (853M, frozen, bf16). Prefix = 256² RGB (64 visual tokens) + instruction |
| Expert | Qwen3.5-style hybrid decoder, 24 layers, width 640, 233M params (fp32). Its full-attention layers read the VLM layer-i prefix K/V (pi0-style). Code adapted from [AeroBrain-project](https://github.com/Ideal-ljl/AeroBrain-project) (Apache-2.0) |
| Removed | MAM (motion-adaptive memory), MSP (motion prior → plain N(0, I) source), VQA co-training |
| Action | chunk H=12 at 10 Hz, per-step `[dx, dy, dz, dyaw, stop]` in the yaw frame of the current frame; robust min-max normalization |
| State | body velocity (3), yaw rate, height |
| Flow matching | t ~ 0.999·Beta(1.5,1)+0.001, linear path, 5 Euler steps at inference |
| Data | `indoor_nano` + `indoor_mini` (houses 1,3–6, 7–46): 222k train frames. val_seen = 5 % held-out episodes; val_unseen = `indoor_mini_unseen` (houses 47–56, 27k frames) |
| Training | AdamW lr 1e-4 (cosine, warmup 500), bs 64, 16k steps, 1×H100, ~85 min |

## Offline (chunk prediction) / 离线指标

| step | split | ADE (m) | FDE (m) | yaw err @1.2 s (°) | stop F1 | const-vel ADE / yaw |
|---|---|---|---|---|---|---|
| 16k | val_seen | 0.037 | 0.069 | 6.8 | 0.95 | 0.073 / 14.2 |
| 16k | val_unseen | **0.032** | 0.063 | **7.9** | 0.89 | 0.060 / 14.0 |
| nano-only 8k | house 6 (unseen) | 0.036 | 0.071 | 7.4 | – | 0.058 / 9.3 |

Seen and unseen numbers are close, so the 45-house data generalizes for short-horizon motion. 已见和未见房子上的指标接近，说明在短时运动预测上，45 栋房子的数据已经能泛化。

## Closed loop (HyperDrone, unseen houses 47–56, 32 episodes, 40 s limit) / 闭环

Success requires stopping within 1.0 m of the goal viewpoint with the target visible and no collision. "near target" means within 1.5 m of the target AABB.
成功的定义：在目标视点 1.0 m 内停下、目标在画面中、且没有碰撞。"到达目标附近"指离目标包围盒不超过 1.5 m。

| policy | SR | near target ever | near target at end | NE (m) | collision | timeout |
|---|---|---|---|---|---|---|
| **expert replay** (executor check) | **1.00** | – | – | 0.06 | 0.00 | 0.00 |
| v0 (full) | 0.00 | **0.22** | 0.13 | 4.26 | 0.75 | 0.00 |
| v0, blind (black images) | 0.00 | 0.13 | 0.09 | 4.39 | 0.56 | 0.44 |
| v0, shuffled instructions | 0.00 | 0.13 | 0.09 | 4.41 | 0.63 | 0.00 |

Images and language both matter somewhat: blind or shuffled inputs reach the target region less often. But the absolute level is far from usable.
图像和语言都起了一定作用：遮掉图像或打乱指令后，到达目标附近的比例都会下降。但绝对水平离可用还很远。

### Executor bug found and fixed / 执行器问题（已修复）

The first closed-loop runs re-anchored each chunk at the **measured** pose. RAPTOR lags its setpoint (about 0.2 rad in yaw), so every replan discarded that lag and the drone under-turned systematically.
最初的闭环每个 chunk 都以**实测**位姿为起点。RAPTOR 跟踪有滞后（yaw 约 0.2 rad），每次重新规划都会把这部分滞后丢掉，导致无人机系统性地转不够。

| executor (expert replay) | SR | collision |
|---|---|---|
| measured anchor, no yaw lead | 0.03 | 0.81 |
| measured anchor, yaw lead 0.3 s | 0.41 | 0.50 |
| **reference anchor** (continuous setpoint trajectory) + yaw lead 0.3 s | **1.00** | **0.00** |

The fix: chunks are anchored on the continuous reference trajectory, while increments are still rotated with the measured yaw. This is now the default in `sakana_vla/eval/closed_loop.py`.
修复方法：chunk 接在连续参考轨迹上，增量仍用实测 yaw 转换到世界系。现在 `sakana_vla/eval/closed_loop.py` 默认就用这种方式。

## Failure analysis / 失败分析

1. **Start-vs-goal ambiguity / 起点与终点混淆。** At t=0 the drone hovers, like at the goal, with a random yaw and the target out of view. Sampled chunks disagree on the turn direction, and some predict `stop` at the start. Single-frame input has no notion of progress.
   起点时无人机在悬停，和到达终点时一样；朝向随机，目标不在视野里。几次采样给出的转向方向互相矛盾，有的还在起点就预测 stop。单帧输入无从判断已经飞了多少。
2. **Grounding / 目标对应。** The drone flies through doors plausibly but often into the wrong room or towards the wrong object. The frozen VLM plus template instructions give weak object/room grounding.
   无人机能合理地穿过门，但常常进错房间、飞向错误的物体。VLM 冻结加上模板指令，导致物体和房间的 grounding 很弱。
3. **Compounding drift → collisions / 误差累积导致碰撞。** Mid-flight chunks are accurate (cm-level offline), but closed-loop states drift outside the expert's tube. The data has only 50 % lightly perturbed episodes.
   飞行途中的 chunk 很准（离线误差是厘米级），但闭环状态会逐渐偏出专家轨迹的覆盖范围。数据里只有 50% 的 episode 带有轻度扰动。

## Next (v1) / 下一步

1. **History (OpenFlyV2 "Base")**: 4 past frames plus the current frame (~320 visual tokens), or a past-motion summary in the state. This resolves start vs goal and gives progress.
   **加入历史**（即 OpenFlyV2 的 Base 配置）：4 张历史帧加当前帧（约 320 个视觉 token），或者在 state 里加入过去运动的摘要。用来区分起点和终点，并提供进度信息。
2. **Stage B**: unfreeze the VLM (lr 1e-5). Optionally add an auxiliary target-grounding loss from `observation.boxes`, which the dataset already provides.
   **阶段 B**：放开 VLM 训练（lr 1e-5）。可以再加一个目标 grounding 的辅助 loss，监督信号直接用数据集里已有的 `observation.boxes`。
3. **Data**: richer and verified instructions (the planned LLM rewrite), more houses (Train200), a stronger and more frequent perturbation/recovery (DAgger-style), and start yaw biased towards the first path segment.
   **数据**：更丰富、经过校验的指令（即计划中的 LLM 改写）；更多房子（Train200）；更强、更频繁的扰动与恢复（DAgger 式）；起点朝向偏向第一段路径的方向。
4. Evaluate on a fixed 64-episode unseen set plus val_seen, with the blind and shuffled-language controls.
   评测改用固定的 64 条 unseen episode 加上 val_seen，并保留遮图像和打乱指令两组对照。

## Reproduce / 复现

```bash
conda activate sakana_vla2   # torch 2.7.1 cu126, transformers 5.5.0, flash-linear-attention
python -m sakana_vla.data.frames_cache --subsets data/vla_dataset/indoor_{nano,mini,mini_unseen} --out data/cache/v1
python -m sakana_vla.train --cache data/cache/v1 --out runs/v1_stageA --steps 16000 --bs 64
# closed loop (env sakana_hyperdrone; spawns the policy server itself on a node-local Unix socket)
PYTHONPATH=. python -m sakana_vla.eval.closed_loop --run runs/v1_stageA --split val_unseen --limit 32 --max-seconds 40
```
