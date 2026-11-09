#!/bin/bash
# Oracle best-of-N go/no-go sweep (docs/wm_verifier_plan.md, section 3). Run inside a GPU allocation:
#   srun --jobid=<id> --overlap bash scripts/bestofn_sweep.sh [run_dir] [limit] [max_seconds]
# Same 32-episode unseen set as v0 (limit 32, seed 1). Each run writes runs/<run>/closed_loop_val_unseen_<tag>/results.json.
set -e
cd "$(dirname "$0")/.."
export HYPERLUX_SKIP_BUILD=${HYPERLUX_SKIP_BUILD:-1} UV_USE_IO_URING=0 PATH=$HOME/miniconda3/envs/sakana_hyperdrone/bin:$PATH HYPERLUX_BACKEND=${HYPERLUX_BACKEND:-OPTIX} PYTHONPATH=$PWD
RUN=${1:-runs/v1_stageA}; LIMIT=${2:-32}; SECS=${3:-40}
COMMON="--run $RUN --split val_unseen --limit $LIMIT --max-seconds $SECS --batch 16 --videos 0"
run() { tag=$1; shift; echo "=== $tag  $*"; python -u -m sakana_vla.eval.closed_loop $COMMON --tag "$tag" "$@" 2>&1 | grep -E "^ProcTHOR|SUMMARY|Traceback|Error" ; }
run bon_n1_first                                                    # baseline (should reproduce the v0 'main' run)
run bon_n8_first        --candidates 8  --select first               # candidate 0 == baseline: determinism check
run bon_n8_random       --candidates 8  --select random
run bon_n8_oracle       --candidates 8  --select oracle
run bon_n8_oraclemotion --candidates 8  --select oracle_motion
run bon_n16_oracle      --candidates 16 --select oracle
run bon_n1_continue     --tentative-stop continue                   # v1_plan P0.1: no hover on the first stop vote
echo ALL_DONE
