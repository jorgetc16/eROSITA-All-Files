#!/bin/bash
set -eo pipefail

# === Config ===
SCRIPT="/home/jortecal/GitHub/eRosita/Fitpoly_TMall_MT_Multiresponse_Method_Profile_1-2keV_MT_3.py"

TELESCOPE_LIST=(TM1 TM2 TM3 TM4 TM5 TM6 TM7)

# === Parallelism ===
# max concurrent jobs (default: number of cores, or set manually, e.g., MAX_PROCS=16)
MAX_PROCS="${MAX_PROCS:-$(nproc 2>/dev/null || echo 4)}"

echo "Running fits with the following configuration:"
echo "Max parallel jobs: $MAX_PROCS"
echo ""

# === Energy List ===
# ENERGY_LIST="1.000000 1.034644 1.069603 1.104879 1.140475 1.176389 1.212593 1.249089 1.285879 1.322965 1.360360 1.398088 1.436154 1.513310 1.552406 1.591783 1.631432 1.671353 1.711548 1.752048 1.792854 1.833968 1.875393 1.917119 1.959139"
ENERGY_LIST="1.08"

# Sanity check: python script exists
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 2
fi

# Export variables for subshells
export SCRIPT 

# Function to run a single (telescope, energy) pair
run_single_fit() {
    local TELESCOPE="$1"
    local ELINE="$2"
    
    local OUTPUT_DIR="/home/jortecal/GitHub/eRosita/LMC5Deg/Results/Proves_Corr_Prof/$TELESCOPE"
    
    # Create output directory if needed
    mkdir -p "$OUTPUT_DIR"
    
    # Format energy string
    printf -v ESTR "%.3f" "$ELINE"
    local EFILE="${ESTR//./_}"
    
    echo "[$(date +%H:%M:%S)] Starting: $TELESCOPE E=${ESTR} keV (PID=$$)"
    
    # Run Python script
    python "$SCRIPT" "$ESTR" "$OUTPUT_DIR" "$TELESCOPE" \
        > "$OUTPUT_DIR/Fit_${TYPEFIT}_E${EFILE}.log" 2>&1 || {
        local STATUS=$?
        echo "[$(date +%H:%M:%S)] FAILED: $TELESCOPE E=${ESTR} keV (exit $STATUS)"
        return 1
    }
    
    echo "[$(date +%H:%M:%S)] Completed: $TELESCOPE E=${ESTR} keV"
}

# Export function so xargs subshells can use it
export -f run_single_fit

# Count total jobs
TOTAL_JOBS=$((${#TELESCOPE_LIST[@]} * $(echo $ENERGY_LIST | wc -w)))
echo "Generating $TOTAL_JOBS jobs (${#TELESCOPE_LIST[@]} telescopes × $(echo $ENERGY_LIST | wc -w) energies)"
echo "Using up to $MAX_PROCS parallel processes"
echo ""

# Generate all (telescope, energy) pairs and process in parallel
for TELESCOPE in "${TELESCOPE_LIST[@]}"; do
    for ELINE in $ENERGY_LIST; do
        echo "$TELESCOPE $ELINE"
    done
done | xargs -n2 -P "$MAX_PROCS" bash -c 'run_single_fit "$@"' _

echo
echo "===================================="
echo "All $TOTAL_JOBS fits finished."
echo "===================================="
