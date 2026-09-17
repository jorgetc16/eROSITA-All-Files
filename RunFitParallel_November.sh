#!/bin/bash
set -eo pipefail

# Force C locale to ensure '.' is used as decimal separator
export LC_ALL=C
export LANG=C

# === Config ===
#SCRIPT="SCRIPT/Fitpoly_Novembre_Final_MT_2to9.py"
SCRIPT="/home/jortecal/GitHub/eRosita/Fit_1-2_FullRangeFit/Fitpoly_MT_1to2_DE_Astro.py"

TELESCOPE_LIST=(TM5)
# TELESCOPE_LIST=(TM1 TM2 TM3 TM4 TM5 TM6 TM7)

# === Parallelism ===
MAX_PROCS="${MAX_PROCS:-$(nproc 2>/dev/null || echo 4)}"

# Counter files (shared across parallel jobs)
STARTED_FILE="/tmp/erosita_jobs_started.$$"
COMPLETED_FILE="/tmp/erosita_jobs_completed.$$"
# Initialize counters
: > "$STARTED_FILE"
: > "$COMPLETED_FILE"

# Utility: atomically increment and print counters
inc_counter() {
  # $1: file path
  local file="$1"
  {
    flock -w 5 9 || exit 1
    echo 1 >> "$file"
    wc -l < "$file"
  } 9>>"$file"
}

echo "Running fits with the following configuration:"
echo "Max parallel jobs: $MAX_PROCS"
echo ""

# === Energy List ===
ENERGY_LIST="1.000"
#ENERGY_LIST="1.073 1.097 1.109 1.133 1.158 1.182 1.206 1.23 1.255 1.279 1.303 1.327 1.352 1.376 1.400 1.424 1.448 1.473 1.485 1.497 1.521 1.545 1.570 1.594 1.618 1.642 1.667 1.691 1.715 1.727 1.752 1.764 1.776 1.800 1.824 1.848 1.861 1.873 1.897 1.921 1.945 1.958 1.970 1.994 2.00 2.018 2.042 2.067 2.091 2.103 2.115 2.139 2.152 2.164 2.176 2.2"
#ENERGY_LIST="2. 2.015 2.03 2.045 2.06 2.075 2.09 2.105 2.12 2.135 2.15 2.165 2.18 2.195 2.21 2.225 2.24 2.255 2.27 2.285 2.3 2.315 2.33 2.345 2.36 2.375 2.39 2.405 2.42 2.435 2.45 2.465 2.48 2.495 2.51 2.525 2.54 2.555 2.57 2.585 2.6 2.615 2.63 2.645 2.66 2.675 2.69 2.705 2.72 2.735 2.75 2.765 2.78 2.795 2.81 2.825 2.84 2.855 2.87 2.885 2.9 2.915 2.93 2.945 2.96 2.975 2.99 3.005 3.02 3.035 3.05 3.065 3.08 3.095 3.11 3.125 3.14 3.155 3.17 3.185 3.2 3.215 3.23 3.245 3.26 3.275 3.29 3.305 3.32 3.335 3.35 3.365 3.38 3.395 3.41 3.425 3.44 3.455 3.47 3.485 3.5 3.515 3.53 3.545 3.56 3.575 3.59 3.605 3.62 3.635 3.65 3.665 3.68 3.695 3.71 3.725 3.74 3.755 3.77 3.785 3.8 3.815 3.83 3.845 3.86 3.875 3.89 3.905 3.92 3.935 3.95 3.965 3.98 3.995 4.01 4.025 4.04 4.055 4.07 4.085 4.1 4.115 4.13 4.145 4.16 4.175 4.19 4.205 4.22 4.235 4.25 4.265 4.28 4.295 4.31 4.325 4.34 4.355 4.37 4.385 4.4 4.415 4.43 4.445 4.46 4.475 4.49 4.505 4.52 4.535 4.55 4.565 4.58 4.595 4.61 4.625 4.64 4.655 4.67 4.685 4.7 4.715 4.73 4.745 4.76 4.775 4.79 4.805 4.82 4.835 4.85 4.865 4.88 4.895 4.91 4.925 4.94 4.955 4.97 4.985 5. 5.015 5.03 5.045 5.06 5.075 5.09 5.105 5.12 5.135 5.15 5.165 5.18 5.195 5.21 5.225 5.24 5.255 5.27 5.285 5.3 5.315 5.33 5.345 5.36 5.375 5.39 5.405 5.42 5.435 5.45 5.465 5.48 5.495 5.51 5.525 5.54 5.555 5.57 5.585 5.6 5.615 5.63 5.645 5.66 5.675 5.69 5.705 5.72 5.735 5.75 5.765 5.78 5.795 5.81 5.825 5.84 5.855 5.87 5.885 5.9 5.915 5.93 5.945 5.96 5.975 5.99 6.005 6.02 6.035 6.05 6.065 6.08 6.095 6.11 6.125 6.14 6.155 6.17 6.185 6.2 6.215 6.23 6.245 6.26 6.275 6.29 6.305 6.32 6.335 6.35 6.365 6.38 6.395 6.41 6.425 6.44 6.455 6.47 6.485 6.5 6.515 6.53 6.545 6.56 6.575 6.59 6.605 6.62 6.635 6.65 6.665 6.68 6.695 6.71 6.725 6.74 6.755 6.77 6.785 6.8 6.815 6.83 6.845 6.86 6.875 6.89 6.905 6.92 6.935 6.95 6.965 6.98 6.995 7.01 7.025 7.04 7.055 7.07 7.085 7.1 7.115 7.13 7.145 7.16 7.175 7.19 7.205 7.22 7.235 7.25 7.265 7.28 7.295 7.31 7.325 7.34 7.355 7.37 7.385 7.4 7.415 7.43 7.445 7.46 7.475 7.49 7.505 7.52 7.535 7.55 7.565 7.58 7.595 7.61 7.625 7.64 7.655 7.67 7.685 7.7 7.715 7.73 7.745 7.76 7.775 7.79 7.805 7.82 7.835 7.85 7.865 7.88 7.895 7.91 7.925 7.94 7.955 7.97 7.985 8. 8.015 8.03 8.045 8.06 8.075 8.09 8.105 8.12 8.135 8.15 8.165 8.18 8.195 8.21 8.225 8.24 8.255 8.27 8.285 8.3 8.315 8.33 8.345 8.36 8.375 8.39 8.405 8.42 8.435 8.45 8.465 8.48 8.495 8.51 8.525 8.54 8.555 8.57 8.585 8.6 8.615 8.63 8.645 8.66 8.675 8.69 8.705 8.72 8.735 8.75 8.765 8.78 8.795 8.81 8.825 8.84 8.855 8.87 8.885 8.9 8.915 8.93 8.945 8.96 8.975 8.99"



# Sanity check: python script exists
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 2
fi

# Export variables for subshells
export SCRIPT STARTED_FILE COMPLETED_FILE

run_single_fit() {
    local TELESCOPE="$1"
    local ELINE="$2"

    local OUTPUT_DIR="/home/jortecal/GitHub/eRosita/LMC5Deg/1to2FullRange/$TELESCOPE"
    mkdir -p "$OUTPUT_DIR"

    printf -v ESTR "%.3f" "$ELINE"
    local EFILE="${ESTR//./_}"

    local STARTED_COUNT=$(inc_counter "$STARTED_FILE")
    echo "[$(date +%H:%M:%S)] Starting: $TELESCOPE E=${ESTR} keV (PID=$$) - Job $STARTED_COUNT/$TOTAL_JOBS started"

    # Run Python script
    python "$SCRIPT" "$ESTR" "$OUTPUT_DIR" "$TELESCOPE" \
        > "$OUTPUT_DIR/Fit_Astro_${TELESCOPE}.log" 2>&1 || {
        local STATUS=$?
        local COMPLETED_COUNT=$(inc_counter "$COMPLETED_FILE")
        echo "[$(date +%H:%M:%S)] FAILED: $TELESCOPE E=${ESTR} keV (exit $STATUS) - Job $COMPLETED_COUNT/$TOTAL_JOBS completed"
        return 1
    }

    local COMPLETED_COUNT=$(inc_counter "$COMPLETED_FILE")
    echo "[$(date +%H:%M:%S)] Completed: $TELESCOPE E=${ESTR} keV - Job $COMPLETED_COUNT/$TOTAL_JOBS completed"
}

export -f run_single_fit inc_counter
export TOTAL_JOBS

TOTAL_JOBS=$((${#TELESCOPE_LIST[@]} * $(echo $ENERGY_LIST | wc -w)))
echo "Generating $TOTAL_JOBS jobs (${#TELESCOPE_LIST[@]} telescopes × $(echo $ENERGY_LIST | wc -w) energies)"
echo "Using up to $MAX_PROCS parallel processes"
echo ""

for ELINE in $ENERGY_LIST; do
    for TELESCOPE in "${TELESCOPE_LIST[@]}"; do
        echo "$TELESCOPE $ELINE"
    done
done | xargs -n2 -P "$MAX_PROCS" bash -c 'run_single_fit "$@"' _

# Final summary
STARTED_COUNT=$(wc -l < "$STARTED_FILE")
COMPLETED_COUNT=$(wc -l < "$COMPLETED_FILE")
echo
echo "===================================="
echo "Jobs started:   $STARTED_COUNT / $TOTAL_JOBS"
echo "Jobs completed: $COMPLETED_COUNT / $TOTAL_JOBS"
echo "All $TOTAL_JOBS fits finished."
echo "===================================="

# Cleanup counter files
rm -f "$STARTED_FILE" "$COMPLETED_FILE"
