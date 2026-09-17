#!/bin/bash
set -eo pipefail

# === Config ===
SCRIPT="/home/jortecal/GitHub/eRosita/Fitpoly_TMall_MT_Novembre.py"

# Optional: override a single telescope by exporting TELESCOPE in the environment:
# TELESCOPE=TM3 ./script.sh
# Otherwise the script will loop TELESCOPE_LIST.
# TELESCOPE_LIST=(TM1 TM2 TM3 TM4 TM5 TM6 TM7)
TELESCOPE_LIST=(TM4)
# === Parallelism ===
# max concurrent telescope jobs (default: number of cores)
MAX_PROCS="${MAX_PROCS:-$(nproc 2>/dev/null || echo 4)}"

# === Energy List ===
NUM_POINTS=200
EMIN=8.47
EMAX=8.475

echo "Running fits with the following configuration:"
echo "Type Fit (1=Minuit, 2=XSpec, 3=DiffeV): $TYPEFIT"
echo "Number of Steps in profiling: $NSTEPS"
echo "Model (1=Powerlaw0+Powerlaw1+Powerlaw2, 2=bknpowerlaw, 3=PowerlawsNegative): $MODEL"
echo "Max parallel telescope jobs: $MAX_PROCS"

# ENERGY_LIST=$(awk -v N="$NUM_POINTS" -v EMIN="$EMIN" -v EMAX="$EMAX" 'BEGIN {
#     for (i=0; i<N; i++) {
#         printf "%.3f ", EMIN + i*(EMAX-EMIN)/(N-1)
#     }
# }')

# THIS A ENERGY LIST WITH AN ARRAY OF ENERGIES
# ENERGY_LIST="2.000000 2.042605 2.085514 2.128735 2.172270 2.216123 2.260291 2.304761 2.349535 2.394615 2.440002 2.485694 2.531693 2.578001 2.624623 2.671569 2.718841 2.766441 2.814357 2.862573 2.911092 2.959915 3.009028 3.058426 3.108113 3.158091 3.208386 3.259000 3.309933 3.361174 3.412713 3.464553 3.516695 3.569153 3.621928 3.675022 3.728439 3.782183 3.836254 3.890642 3.945326 4.000306 4.055584 4.111157 4.167025 4.223190 4.279685 4.336519 4.393692 4.451159 4.508921 4.566980 4.625338 4.683995 4.742954 4.802227 4.861825 4.921751 4.981984 5.042511 5.103333 5.164455 5.225881 5.287613 5.349657 5.412017 5.474695 5.537695 5.601019 5.664670 5.728593 5.792788 5.857266 5.922084 5.987242 6.052722 6.118490 6.184550 6.250927 6.317637 6.384666 6.452013 6.519681 6.587671 6.655984 6.724592 6.793458 6.862581 6.932017 7.001780 7.071867 7.142267 7.212983 7.283995 7.355297 7.426887 7.498758 7.570911 7.643411 7.716288 7.789526 7.863067 7.936913 8.011017 8.085370 8.160005 8.234974 8.310278 8.385898 8.461835 8.538074 8.614608 8.691444 8.768602 8.846084 8.923870 9.000000"
ENERGY_LIST="8.386"

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
export SCRIPT ENERGY_LIST TYPEFIT NSTEPS MODEL

run_for_telescope() {
    local TELESCOPE="$1"
    echo
    echo "===================================="
    echo " Starting runs for telescope: $TELESCOPE"
    echo "===================================="

    # === Output directory ===
    local OUTPUT_DIR="/home/jortecal/GitHub/eRosita/LMC5Deg/Results/Proves_MT_Script/$TELESCOPE"

    # Create output directory if it does not exist
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Creating output directory: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    else
        echo "Output directory exists: $OUTPUT_DIR"
    fi

    for ELINE in $ENERGY_LIST; do
        # normalize to fixed 3 decimals
        printf -v ESTR "%.3f" "$ELINE"
        # filesystem-friendly label
        local EFILE="${ESTR//./_}"

        echo "-----------------------------"
        echo "Running fit for Eline = ${ESTR} keV (telescope: $TELESCOPE)"

        # run python script, redirect stdout/stderr to per-telescope per-energy log
        python "$SCRIPT" "$ESTR" "$OUTPUT_DIR" "$TELESCOPE" > "$OUTPUT_DIR/Fit_${TYPEFIT}_E${EFILE}.log" 2>&1 || {
            local STATUS=$?
            echo "Fit failed for E = ${ESTR} keV (telescope: $TELESCOPE) (exit $STATUS)"
            # continue to next energy (do not exit whole script)
        }

        echo "Fit completed (or failed) for E = ${ESTR} keV (telescope: $TELESCOPE)"
    done

    echo "All fits complete for $TELESCOPE"
}

# Use xargs -P if available for simple concurrency
if command -v xargs >/dev/null 2>&1; then
    printf "%s\n" "${SELECTED_TELESCOPES[@]}" | xargs -n1 -P "$MAX_PROCS" -I{} bash -c '
        TELESCOPE="$1"
        # re-exported variables visible: SCRIPT, ENERGY_LIST, TYPEFIT, NSTEPS, MODEL
        run_for_telescope() {
            local TELESCOPE="$1"
            echo
            echo "===================================="
            echo " Starting runs for telescope: $TELESCOPE"
            echo "===================================="

            OUTPUT_DIR="/home/jortecal/GitHub/eRosita/LMC5Deg/Results/Proves_MT_Script/$TELESCOPE"
            if [ ! -d "$OUTPUT_DIR" ]; then
                echo "Creating output directory: $OUTPUT_DIR"
                mkdir -p "$OUTPUT_DIR"
            else
                echo "Output directory exists: $OUTPUT_DIR"
            fi

            for ELINE in $ENERGY_LIST; do
                printf -v ESTR "%.3f" "$ELINE"
                EFILE="${ESTR//./_}"
                echo "-----------------------------"
                echo "Running fit for Eline = ${ESTR} keV (telescope: $TELESCOPE)"
                python "$SCRIPT" "$ESTR" "$OUTPUT_DIR" "$TELESCOPE" > "$OUTPUT_DIR/Fit_${TYPEFIT}_E${EFILE}.log" 2>&1 || {
                    STATUS=$?
                    echo "Fit failed for E = ${ESTR} keV (telescope: $TELESCOPE) (exit $STATUS)"
                }
                echo "Fit completed (or failed) for E = ${ESTR} keV (telescope: $TELESCOPE)"
            done

            echo "All fits complete for $TELESCOPE"
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
