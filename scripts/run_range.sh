#!/bin/bash
# usage: run_range.sh <subset_part> <houses e.g. 7-19> <episodes_per_house> <seed>
set -e
export UV_USE_IO_URING=0 PATH=$HOME/miniconda3/envs/sakana_hyperdrone/bin:$PATH HYPERLUX_BACKEND=${HYPERLUX_BACKEND:-OPTIX} PYTHONPATH=$HOME/vla/Sakana_VLA
cd $HOME/vla/Sakana_VLA
python -u -m datagen.generate --subset $1 --houses $2 --episodes-per-house $3 --per-target 5 --batch $3 --seed $4 --out data/vla_parts
