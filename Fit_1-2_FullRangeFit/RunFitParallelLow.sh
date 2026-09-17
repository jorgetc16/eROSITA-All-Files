#!/bin/bash
set -eo pipefail

# === Config ===
SCRIPT="/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_global/Fit_1-2/Fitpoly_MT_1to2_DE_DMprofile.py"

TELESCOPE_LIST=(TM1 TM2 TM3 TM4 TM5 TM6 TM7)
#TELESCOPE_LIST=(TM4)

# === Parallelism ===
# max concurrent jobs (default: number of cores, or set manually, e.g., MAX_PROCS=16)
MAX_PROCS="${MAX_PROCS:-$(nproc 2>/dev/null || echo 4)}"

echo "Running fits with the following configuration:"
echo "Type Fit (1=Minuit, 2=XSpec, 3=DiffeV): $TYPEFIT"
echo "Number of Steps in profiling: $NSTEPS"
echo "Model (1=Powerlaw0+Powerlaw1+Powerlaw2, 2=bknpowerlaw, 3=PowerlawsNegative): $MODEL"
echo "Max parallel jobs: $MAX_PROCS"
echo ""

# === Energy List ===
ENERGY_LIST="1.133 1.158 1.182 1.230 1.255"





#1.000 1.012 1.024 1.036 1.048 1.061 1.073 1.085 1.097 1.109
#1.120 1.133 1.145 1.158 1.170 1.182 1.194 1.206 1.218 1.230
#1.242 1.255 1.267 1.279 1.291 1.303 1.315 1.327 1.339 1.352
#1.364 1.376 1.388 1.400 1.412 1.424 1.436 1.448 1.461 1.473
#1.485 1.497 1.509 1.521 1.533 1.545 1.558 1.570 1.582 1.594
#1.606 1.618 1.630 1.642 1.655 1.667 1.679 1.691 1.703 1.715
#1.727 1.739 1.752 1.764 1.776 1.788 1.800 1.812 1.824 1.836
#1.848 1.861 1.873 1.885 1.897 1.909 1.921 1.933 1.945 1.958
#1.970 1.982 1.994 2.006 2.018 2.030 2.042 2.055 2.067 2.079
#2.091 2.103 2.115 2.127 2.139 2.152 2.164 2.176 2.188 2.200

# peaks
# 1.206 1.424 2.103

# To correct
# 1.473

#1.073 1.097 1.109 1.133 1.158 1.182 1.206
#1.230 1.255 1.279 1.303 1.327 1.352 1.376
#1.400 1.424 1.448 1.473 1.485 1.497 1.521
#1.545 1.570 1.594 1.618 1.642 1.667 1.691
#1.715 1.727 1.752 1.764 1.776 1.800 1.824
#1.848 1.861 1.873 1.897 1.921 1.945 1.958
#1.970 1.994 2.000 2.018 2.042 2.067 2.091
#2.103 2.115 2.139 2.152 2.164 2.176 2.200



# Sanity check: python script exists
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 2
fi

# Export variables for subshells
export SCRIPT TYPEFIT NSTEPS MODEL

# Function to run a single (telescope, energy) pair
run_single_fit() {
    local TELESCOPE="$1"
    local ELINE="$2"
    
    local OUTPUT_DIR="/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_global/Fit_1-2/output/$TELESCOPE"
    
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
