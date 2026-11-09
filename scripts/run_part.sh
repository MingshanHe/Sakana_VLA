#!/bin/bash
# usage: run_part.sh <subset> <house> <episodes> <seed>
set -e
export PATH=$HOME/miniconda3/envs/sakana_hyperdrone/bin:$PATH HYPERLUX_BACKEND=${HYPERLUX_BACKEND:-GENERIC} PYTHONPATH=$HOME/vla/Sakana_VLA
cd $HOME/vla/Sakana_VLA
python -u -m datagen.generate --subset $1_h$2 --houses $2 --episodes-per-house $3 --per-target 5 --batch $3 --seed $4 --out data/vla_parts
