#!/bin/bash
set -eo pipefail

# === Config ===
# SCRIPT="/home/jortecal/GitHub/eRosita/Fitpoly_Novembre_Final_MT_2to9.py"
SCRIPT="/home/jortecal/GitHub/eRosita/Fitpoly_MT_1to2_DE.py"

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

# === Energy-Telescope Combinations ===
# Format: "ENERGY:TM1,TM2,TM3"
# Each line defines which telescopes to run for a specific energy
ENERGY_TM_PAIRS=(
    "1.642:TM2,TM4,TM7"
    "1.594:TM5"
    "1.521:TM2,TM6"
    "1.497:TM6,TM1,TM3"
    "2.139:TM2,TM7"
    "2.164:TM5,TM7"
    # Add more combinations here
)

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

    local OUTPUT_DIR="/home/jortecal/GitHub/eRosita/LMC5Deg/ResultsNewData/1to2/$TELESCOPE"
    mkdir -p "$OUTPUT_DIR"

    printf -v ESTR "%.3f" "$ELINE"
    local EFILE="${ESTR//./_}"

    local STARTED_COUNT=$(inc_counter "$STARTED_FILE")
    echo "[$(date +%H:%M:%S)] Starting: $TELESCOPE E=${ESTR} keV (PID=$$) - Job $STARTED_COUNT/$TOTAL_JOBS started"

    # Run Python script
    python "$SCRIPT" "$ESTR" "$OUTPUT_DIR" "$TELESCOPE" \
        > "$OUTPUT_DIR/Fit_E${EFILE}.log" 2>&1 || {
        local STATUS=$?
        local COMPLETED_COUNT=$(inc_counter "$COMPLETED_FILE")
        echo "[$(date +%H:%M:%S)] FAILED: $TELESCOPE E=${ESTR} keV (exit $STATUS) - Job $COMPLETED_COUNT/$TOTAL_JOBS completed"
        return 1
    }

    local COMPLETED_COUNT=$(inc_counter "$COMPLETED_FILE")
    echo "[$(date +%H:%M:%S)] Completed: $TELESCOPE E=${ESTR} keV - Job $COMPLETED_COUNT/$TOTAL_JOBS completed"
}

export -f run_single_fit inc_counter

# Parse energy-telescope pairs and generate job list
JOB_LIST=()
for pair in "${ENERGY_TM_PAIRS[@]}"; do
    # Split by colon: energy:telescopes
    IFS=':' read -r energy telescopes <<< "$pair"
    
    # Split telescopes by comma
    IFS=',' read -ra TM_ARRAY <<< "$telescopes"
    
    # Add each telescope for this energy
    for tm in "${TM_ARRAY[@]}"; do
        JOB_LIST+=("$tm $energy")
    done
done

TOTAL_JOBS=${#JOB_LIST[@]}
export TOTAL_JOBS

echo "Generating $TOTAL_JOBS jobs from ${#ENERGY_TM_PAIRS[@]} energy-TM combinations"
echo "Using up to $MAX_PROCS parallel processes"
echo ""

# Print job list for verification
echo "Job list:"
printf '%s\n' "${JOB_LIST[@]}"
echo ""

# Run jobs in parallel
printf '%s\n' "${JOB_LIST[@]}" | xargs -n2 -P "$MAX_PROCS" bash -c 'run_single_fit "$@"' _

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