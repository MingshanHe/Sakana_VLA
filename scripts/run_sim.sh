#!/bin/bash
# One closed-loop run inside a Slurm allocation with SERIALIZED START-UP.
#   bash scripts/run_sim.sh <jobid> <tag> <logfile> <closed_loop args...>
# hyperlux's JIT takes a POSIX flock on ~/.cache/hyperlux (NFS); several sims initialising at once (same node or
# across nodes) deadlock there. So a login-node flock is held from launch until the first house is flying, then
# released; the run itself proceeds concurrently with others. A 25 min timeout guards against hangs.
# HYPERLUX_SKIP_BUILD=1 skips the per-process `cmake --build` check (which hangs on this cluster); everything must
# already be built for this node type (run one sim normally first, or set HYPERLUX_SKIP_BUILD=0).
set -u
J=$1; TAG=$2; LOG=$3; shift 3
cd "$(dirname "$0")/.."
export HYPERLUX_SKIP_BUILD=${HYPERLUX_SKIP_BUILD:-1} UV_USE_IO_URING=0 PATH=$HOME/miniconda3/envs/sakana_hyperdrone/bin:$PATH HYPERLUX_BACKEND=${HYPERLUX_BACKEND:-OPTIX} PYTHONPATH=$PWD
unset HYPERLUX_CACHE_DIR
RUN=${RUN:-runs/v1_stageA}
exec 9>/tmp/${USER}-sim-start.lock
flock 9
echo "=== $TAG start $(date +%T) (job $J)"
timeout ${SIM_TIMEOUT:-1500} srun --jobid=$J --overlap python -u -m sakana_vla.eval.closed_loop --run $RUN --batch 16 --videos 0 \
    --tag "$TAG" "$@" > "$LOG" 2>&1 &
PID=$!
until grep -qE "^ProcTHOR|Traceback" "$LOG" 2>/dev/null || ! kill -0 $PID 2>/dev/null; do sleep 5; done
flock -u 9
echo "    flying $(date +%T)"
wait $PID
echo "    exit $? $(date +%T)"
grep -E "SUMMARY|Traceback" "$LOG" | cut -c1-320
