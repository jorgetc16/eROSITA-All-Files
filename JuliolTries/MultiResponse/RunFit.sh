#!/bin/bash

# === Config ===
SCRIPT="/home/jortecal/GitHub/eRosita/JuliolTries/MultiResponse/Fitpoly_TMall_MT_Multiresponse_Loop_PreFit.py"

# If you want to run a single telescope set TELESCOPE (e.g. TELESCOPE=TM3 ./script.sh)
# Otherwise the script will loop the list below.
# TELESCOPE="TM7" # optional override

# List of telescopes to loop over. Edit this list if you want different sets.
TELESCOPE_LIST=(TM1 TM2 TM3 TM4 TM5 TM6 TM7)

# You can also include other identifiers like ALL or ALL_CLEAN if desired:
# TELESCOPE_LIST=(TM1 TM2 TM3 TM4 TM5 TM6 TM7 ALL ALL_CLEAN)

# === Energy List ===
NUM_POINTS=2
EMIN=6.749
EMAX=8.050

echo "Running fits with the following configuration:"
echo "Type Fit (1=Minuit, 2=XSpec, 3=DiffeV): $TYPEFIT"
echo "Number of Steps in profiling: $NSTEPS"
echo "Model (1=Powerlaw0+Powerlaw1+Powerlaw2, 2=bknpowerlaw, 3=PowerlawsNegative): $MODEL"

ENERGY_LIST=$(awk -v N="$NUM_POINTS" -v EMIN="$EMIN" -v EMAX="$EMAX" 'BEGIN {
    for (i=0; i<N; i++) {
        printf "%.3f ", EMIN + i*(EMAX-EMIN)/(N-1)
    }
}')

# sanity check: python script exists
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 2
fi

# If TELESCOPE is set externally, run only that one; otherwise loop TELESCOPE_LIST
if [ -n "${TELESCOPE-}" ]; then
    # allow single telescope override
    SELECTED_TELESCOPES=("$TELESCOPE")
else
    SELECTED_TELESCOPES=("${TELESCOPE_LIST[@]}")
fi

for TELESCOPE in "${SELECTED_TELESCOPES[@]}"; do
    echo
    echo "===================================="
    echo " Starting runs for telescope: $TELESCOPE"
    echo "===================================="

    # === Output directory ===
    OUTPUT_DIR="/home/jortecal/GitHub/eRosita/JuliolTries/MultiResponse/Results/Proves/$TELESCOPE"

    # Create output directory if it does not exist
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Creating output directory: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    else
        echo "Output directory exists: $OUTPUT_DIR"
    fi

    # === Loop over energies ===
    for ELINE in $ENERGY_LIST; do
        # normalize ELINE to fixed 3 decimals (avoid weird formatting)
        printf -v ESTR "%.3f" "$ELINE"
        # make a filesystem-friendly label (replace '.' with '_')
        EFILE="${ESTR//./_}"

        echo "-----------------------------"
        echo "Running fit for Eline = ${ESTR} keV (telescope: $TELESCOPE)"

        # run python script, redirect stdout/stderr to per-telescope per-energy log
        python "$SCRIPT" "$ESTR" "$OUTPUT_DIR" "$TELESCOPE" > "$OUTPUT_DIR/Fit_${TYPEFIT}_E${EFILE}.log" 2>&1
        STATUS=$?

        if [ $STATUS -eq 0 ]; then
            echo "Fit succeeded for E = ${ESTR} keV (telescope: $TELESCOPE)"
        else
            echo "Fit failed for E = ${ESTR} keV (telescope: $TELESCOPE) (exit $STATUS)"
            # optionally: continue to next run (we already continue)
        fi
    done

    echo "All fits complete for $TELESCOPE"
done

echo
echo "All telescope runs finished."
