#!/usr/bin/env bash
# Run the full scheduling experiment: baseline vs LB4OMP configurations.
#
# Prerequisites:
#   1. Build the harness:   cmake -S . -B build && cmake --build build
#   2. Build LB4OMP:        ./scripts/setup_lb4omp.sh
#
# Usage: ./scripts/run_experiment.sh
#
# Tune CPU_MHZ to match your machine (cat /proc/cpuinfo | grep MHz).
# Results are written to results/*.csv.

set -euo pipefail

HARNESS=./build/sumfact_harness
LB4OMP_LIB=./lb4omp/build/runtime/src
RESULTS=./results

DEGREE="${DEGREE:-4}"
ELEMENTS="${ELEMENTS:-20000}"
TIMESTEPS="${TIMESTEPS:-200}"
CPU_MHZ="${CPU_MHZ:-2500}"
NCORES=$(nproc)

mkdir -p "$RESULTS"

if [ ! -x "$HARNESS" ]; then
    echo "ERROR: $HARNESS not found. Build first:" >&2
    echo "  cmake -S . -B build && cmake --build build" >&2
    exit 1
fi

echo "========================================"
echo " LB4OMP scheduling experiment"
echo "========================================"
echo " cores=$NCORES  degree=$DEGREE  elements=$ELEMENTS"
echo " timesteps=$TIMESTEPS  CPU_MHZ=$CPU_MHZ"
echo ""

run() {
    local label="$1"; shift
    echo "--- $label ---"
    "$@" 2>&1 | tail -6
    echo ""
}

# 1. Baseline: system libomp, static
run "baseline_static (system libomp)" \
    env OMP_SCHEDULE=static OMP_NUM_THREADS="$NCORES" \
    "$HARNESS" --degree "$DEGREE" --elements "$ELEMENTS" \
    --timesteps "$TIMESTEPS" --csv "$RESULTS/baseline_static.csv"

# Verify LB4OMP is available
if [ ! -f "$LB4OMP_LIB/libomp.so" ]; then
    echo "WARNING: LB4OMP not built at $LB4OMP_LIB"
    echo "Run ./scripts/setup_lb4omp.sh first."
    echo "Skipping LB4OMP runs."
    exit 0
fi

# 2. LB4OMP with static
run "lb4omp_static" \
    env LD_LIBRARY_PATH="$LB4OMP_LIB:${LD_LIBRARY_PATH:-}" \
    OMP_SCHEDULE=static OMP_NUM_THREADS="$NCORES" \
    KMP_CPU_SPEED="$CPU_MHZ" \
    "$HARNESS" --degree "$DEGREE" --elements "$ELEMENTS" \
    --timesteps "$TIMESTEPS" --csv "$RESULTS/lb4omp_static.csv"

# 3. LB4OMP with dynamic,1 (worst-case scheduling overhead reference)
run "lb4omp_dynamic1" \
    env LD_LIBRARY_PATH="$LB4OMP_LIB:${LD_LIBRARY_PATH:-}" \
    OMP_SCHEDULE="dynamic,1" OMP_NUM_THREADS="$NCORES" \
    KMP_CPU_SPEED="$CPU_MHZ" \
    "$HARNESS" --degree "$DEGREE" --elements "$ELEMENTS" \
    --timesteps "$TIMESTEPS" --csv "$RESULTS/lb4omp_dynamic1.csv"

# 4. LB4OMP with guided
run "lb4omp_guided" \
    env LD_LIBRARY_PATH="$LB4OMP_LIB:${LD_LIBRARY_PATH:-}" \
    OMP_SCHEDULE=guided OMP_NUM_THREADS="$NCORES" \
    KMP_CPU_SPEED="$CPU_MHZ" \
    "$HARNESS" --degree "$DEGREE" --elements "$ELEMENTS" \
    --timesteps "$TIMESTEPS" --csv "$RESULTS/lb4omp_guided.csv"

# 5-7. LB4OMP RL auto-selection (Q-Learning=6, Double-Q=7, SARSA=8)
for RL_CODE in 6 7 8; do
    case $RL_CODE in
        6) RL_NAME="q_learning" ;;
        7) RL_NAME="double_q"   ;;
        8) RL_NAME="sarsa"      ;;
    esac

    run "lb4omp_rl_${RL_NAME} (OMP_SCHEDULE=$RL_CODE)" \
        env LD_LIBRARY_PATH="$LB4OMP_LIB:${LD_LIBRARY_PATH:-}" \
        OMP_SCHEDULE="$RL_CODE" OMP_NUM_THREADS="$NCORES" \
        KMP_CPU_SPEED="$CPU_MHZ" \
        KMP_RL_POLICY=epsilon_greedy \
        KMP_EPSILON=0.9 KMP_EPSILON_DECAY=0.9 KMP_EPSILON_MIN=0.1 \
        KMP_REWARD=looptime \
        KMP_ALPHA=0.85 KMP_ALPHA_DECAY=0.9 KMP_ALPHA_MIN=0.1 \
        KMP_GAMMA=0.95 \
        KMP_TIME_LOOPS="$RESULTS/lb4omp_rl_${RL_NAME}_looptimes.csv" \
        "$HARNESS" --degree "$DEGREE" --elements "$ELEMENTS" \
        --timesteps "$TIMESTEPS" --csv "$RESULTS/lb4omp_rl_${RL_NAME}.csv"
done

echo "========================================"
echo " All runs complete.  Results in $RESULTS/"
echo " Run: python3 scripts/summarize.py"
echo "========================================"
