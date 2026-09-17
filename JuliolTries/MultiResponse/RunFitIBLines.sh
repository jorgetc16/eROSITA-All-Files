#!/bin/bash
set -eo pipefail

# === Config ===
SCRIPT="/home/jortecal/GitHub/eRosita/JuliolTries/MultiResponse/FitIBLines.py"

# Optional: override a single telescope by exporting TELESCOPE in the environment:
# TELESCOPE=TM3 ./script.sh
# Otherwise the script will loop TELESCOPE_LIST.
TELESCOPE_LIST=(TM1 TM2 TM3 TM4 TM5 TM6 TM7)

# === Parallelism ===
# max concurrent telescope jobs (default: number of cores)
MAX_PROCS="${MAX_PROCS:-$(nproc 2>/dev/null || echo 4)}"

# === Base path for IB line files ===
IB_LINES_DIR="/home/jortecal/GitHub/eRosita/IBLines"

echo "Running fits with the following configuration:"
echo "Script: $SCRIPT"
echo "Max parallel telescope jobs: $MAX_PROCS"
echo "IB lines directory: $IB_LINES_DIR"

# sanity check: python script exists
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 2
fi

# If TELESCOPE is set externally, run only that one; otherwise loop TELESCOPE_LIST
if [ -n "${TELESCOPE-}" ]; then
    SELECTED_TELESCOPES=("$TELESCOPE")
else
    SELECTED_TELESCOPES=("${TELESCOPE_LIST[@]}")
fi

# Export variables so subshells can see them (used by xargs/bash -c)
export SCRIPT IB_LINES_DIR

run_for_telescope() {
    local TELESCOPE="$1"
    echo
    echo "===================================="
    echo " Starting runs for telescope: $TELESCOPE"
    echo "===================================="

    # === Get telescope number (1-7) ===
    local TM_NUM="${TELESCOPE#TM}"  # Remove "TM" prefix to get number
    
    # === Load energy list from IB lines file ===
    local IB_FILE="${IB_LINES_DIR}/Lines${TELESCOPE}.txt"
    
    if [ ! -f "$IB_FILE" ]; then
        echo "ERROR: IB lines file not found: $IB_FILE"
        return 1
    fi
    
    # Read first column (energies) from file, skip empty lines
    local ENERGY_LIST=$(awk 'NF > 0 {print $1}' "$IB_FILE")
    local NUM_ENERGIES=$(echo "$ENERGY_LIST" | wc -l)
    
    echo "Loaded $NUM_ENERGIES energies from $IB_FILE"
    echo "Energy range: $(echo "$ENERGY_LIST" | head -n1) - $(echo "$ENERGY_LIST" | tail -n1) keV"

    # === Output directory ===
    local OUTPUT_DIR="/home/jortecal/GitHub/eRosita/LMC5Deg/Results/IBLines_bestfit/$TELESCOPE"

    # Create output directory if it does not exist
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Creating output directory: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    else
        echo "Output directory exists: $OUTPUT_DIR"
    fi

    local count=0
    for ELINE in $ENERGY_LIST; do
        count=$((count+1))
        # normalize to fixed 3 decimals
        printf -v ESTR "%.3f" "$ELINE"
        # filesystem-friendly label
        local EFILE="${ESTR//./_}"

        echo "-----------------------------"
        echo "[$count/$NUM_ENERGIES] Running fit for Eline = ${ESTR} keV (telescope: $TELESCOPE)"

        # run python script, redirect stdout/stderr to per-telescope per-energy log
        python "$SCRIPT" "$ESTR" "$OUTPUT_DIR" "$TELESCOPE" > "$OUTPUT_DIR/Fit__E${EFILE}.log" 2>&1 || {
            local STATUS=$?
            echo "Fit failed for E = ${ESTR} keV (telescope: $TELESCOPE) (exit $STATUS)"
            # continue to next energy (do not exit whole script)
        }

        echo "Fit completed (or failed) for E = ${ESTR} keV (telescope: $TELESCOPE)"
    done

    echo "All $NUM_ENERGIES fits complete for $TELESCOPE"
}

# Use xargs -P if available for simple concurrency
if command -v xargs >/dev/null 2>&1; then
    printf "%s\n" "${SELECTED_TELESCOPES[@]}" | xargs -n1 -P "$MAX_PROCS" -I{} bash -c '
        TELESCOPE="$1"
        # re-exported variables visible: SCRIPT, IB_LINES_DIR
        run_for_telescope() {
            local TELESCOPE="$1"
            echo
            echo "===================================="
            echo " Starting runs for telescope: $TELESCOPE"
            echo "===================================="

            local TM_NUM="${TELESCOPE#TM}"
            local IB_FILE="${IB_LINES_DIR}/Lines${TELESCOPE}.txt"
            
            if [ ! -f "$IB_FILE" ]; then
                echo "ERROR: IB lines file not found: $IB_FILE"
                return 1
            fi
            
            local ENERGY_LIST=$(awk "NF > 0 {print \$1}" "$IB_FILE")
            local NUM_ENERGIES=$(echo "$ENERGY_LIST" | wc -l)
            
            echo "Loaded $NUM_ENERGIES energies from $IB_FILE"

            OUTPUT_DIR="/home/jortecal/GitHub/eRosita/LMC5Deg/Results/IBLines_bestfit/$TELESCOPE"
            if [ ! -d "$OUTPUT_DIR" ]; then
                echo "Creating output directory: $OUTPUT_DIR"
                mkdir -p "$OUTPUT_DIR"
            else
                echo "Output directory exists: $OUTPUT_DIR"
            fi

            local count=0
            for ELINE in $ENERGY_LIST; do
                count=$((count+1))
                printf -v ESTR "%.3f" "$ELINE"
                EFILE="${ESTR//./_}"
                echo "-----------------------------"
                echo "[$count/$NUM_ENERGIES] Running fit for Eline = ${ESTR} keV (telescope: $TELESCOPE)"
                python "$SCRIPT" "$ESTR" "$OUTPUT_DIR" "$TELESCOPE" > "$OUTPUT_DIR/Fit__E${EFILE}.log" 2>&1 || {
                    STATUS=$?
                    echo "Fit failed for E = ${ESTR} keV (telescope: $TELESCOPE) (exit $STATUS)"
                }
                echo "Fit completed (or failed) for E = ${ESTR} keV (telescope: $TELESCOPE)"
            done

            echo "All $NUM_ENERGIES fits complete for $TELESCOPE"
        }

        run_for_telescope "$TELESCOPE"
    ' _ {}
else
    # fallback if xargs not present: manual job control
    echo "xargs not found; falling back to manual concurrency control using background jobs."
    running=0
    for TELESCOPE in "${SELECTED_TELESCOPES[@]}"; do
        # Start job in background
        run_for_telescope "$TELESCOPE" &
        running=$((running+1))
        if [ "$running" -ge "$MAX_PROCS" ]; then
            # wait for any job to finish (wait -n is preferable if available)
            if wait -n 2>/dev/null; then
                running=$((running-1))
            else
                # portable fallback: wait for all then reset counter
                wait
                running=0
            fi
        fi
    done
    # wait for remaining background jobs
    wait
fi

echo
echo "All telescope runs finished."