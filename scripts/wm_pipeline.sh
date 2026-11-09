#!/bin/bash
# World-model verifier pipeline, run stage by stage inside a GPU allocation:
#   srun --jobid=<id> --overlap bash scripts/wm_pipeline.sh <stage> [args]
# stages:
#   sweep                 oracle best-of-N sweep (scripts/bestofn_sweep.sh)
#   dump <limit> <tag>    on-policy replay from training houses (N=1 policy + N=4 random) -> data/wm/onpolicy_<tag>
#   train <out> [steps]   train the chunk world model on expert + on-policy replay (env sakana_vla2)
#   eval <wm_ckpt> <N>    closed loop with --select wm / wm_motion on the 32-episode unseen set
set -e
cd "$(dirname "$0")/.."
export PATH=$HOME/miniconda3/envs/sakana_hyperdrone/bin:$PATH HYPERLUX_BACKEND=${HYPERLUX_BACKEND:-OPTIX} PYTHONPATH=$PWD
VLA2=$HOME/miniconda3/envs/sakana_vla2/bin/python
RUN=${RUN:-runs/v1_stageA}
CL="python -u -m sakana_vla.eval.closed_loop --run $RUN --batch 16 --videos 0"
filt() { grep -E "^ProcTHOR|SUMMARY|Traceback|Error|replay|world model|EVAL|done" ; }
case "$1" in
  sweep)
    bash scripts/bestofn_sweep.sh "$RUN" 32 40 ;;
  dump)
    LIMIT=${2:-256}; TAG=${3:-v0}
    $CL --split train --limit $LIMIT --max-seconds 30 --tag dump_${TAG}_a --dump-wm data/wm/onpolicy_$TAG 2>&1 | filt
    $CL --split train --limit $LIMIT --max-seconds 30 --tag dump_${TAG}_b --candidates 4 --select random \
        --dump-wm data/wm/onpolicy_$TAG --resample-noise 2>&1 | filt
    $CL --split val_unseen --limit 32 --max-seconds 40 --tag dump_${TAG}_u --dump-wm data/wm/onpolicy_${TAG}_unseen 2>&1 | filt
    ls data/wm/onpolicy_$TAG | wc -l ;;
  train)
    OUT=${2:-runs/wm_v0}; STEPS=${3:-15000}
    PYTHONNOUSERSITE=1 $VLA2 -u -m sakana_vla.wm.train_wm --data data/wm/expert data/wm/onpolicy_* --out $OUT \
        --size ${SIZE:-size50m} --steps $STEPS --batch 16 --seq 16 --eval-every 500 2>&1 | tee -a logs/$(basename $OUT).log | filt ;;
  eval)
    WM=${2:-runs/wm_v0/best.pt}; N=${3:-8}
    for sel in wm wm_motion; do
      $CL --split val_unseen --limit 32 --max-seconds 40 --tag bon_n${N}_${sel} --candidates $N --select $sel \
          --server-args "--wm $WM" 2>&1 | filt
    done
    $VLA2 scripts/bestofn_table.py $RUN ;;
  *) echo "unknown stage $1"; exit 1 ;;
esac
