#!/bin/bash
# Simple single-room tasks (fly_to / land). usage: run_simple.sh <subset_part> <houses e.g. 1-12> <episodes_per_house> <seed> [cam_pitch_deg]
# Run ONE of these at a time per user: hyperlux's JIT lock on ~/.cache/hyperlux (NFS) deadlocks concurrent start-ups.
set -e
export PATH=$HOME/miniconda3/envs/sakana_hyperdrone/bin:$PATH HYPERLUX_BACKEND=${HYPERLUX_BACKEND:-OPTIX} PYTHONPATH=$HOME/vla/Sakana_VLA
export UV_USE_IO_URING=0  # set HYPERLUX_SKIP_BUILD=1 yourself only when every JIT library is already built (any value, even 0, skips)
cd $HOME/vla/Sakana_VLA
python -u -m datagen.generate --task-set simple --subset $1 --houses $2 --episodes-per-house $3 --per-target 4 --batch $3 \
    --perturb-prob ${PERTURB:-0.3} --seed $4 --cam-pitch ${5:-0} --out data/vla_parts
