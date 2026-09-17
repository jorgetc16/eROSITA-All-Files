#!/bin/bash

# === Config ===
SCRIPT="/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_test_3deg/Fitpoly/Fitpoly_TMall_MT_Model4.py"
TYPEFIT="3"       # 1 = Minuit, 2 = XSpec, 3 = DiffeV
NSTEPS="10"       # Profile steps
NMODEL="4"        # Type of model

# === Energy List ===
NUM_POINTS=2 #200
EMIN=0.842 #0.5
EMAX=0.842 #9.0

ENERGY_LIST=$(awk -v N="$NUM_POINTS" -v EMIN="$EMIN" -v EMAX="$EMAX" 'BEGIN {
    for (i=0; i<N; i++) {
        printf "%.3f ", EMIN + i*(EMAX-EMIN)/(N-1)
    }
}')
# echo "Running fits for $NUM_POINTS energy points between $EMIN and $EMAX keV"

# === Output directory ===
OUTPUT_DIR="/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_test_3deg/Fitpoly/Fit/XSpec/Model4/TMsingle_rebinned/test"
if [ ! -d "$OUTPUT_DIR" ]; then
    # echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
else
    echo "Output directory exists: $OUTPUT_DIR"
fi

# === Loop over energies ===
for ELINE in $ENERGY_LIST; do
    echo "-----------------------------"
    echo "Running fit for Eline = $ELINE keV"
    python3 "$SCRIPT" "$ELINE" "$TYPEFIT" "$NSTEPS" "$OUTPUT_DIR" "$NMODEL" > "$OUTPUT_DIR/Fit_E${ELINE}.log" 2>&1

    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "Fit succeeded for E = $ELINE keV"
    else
        echo "Fit failed for E = $ELINE keV (exit $STATUS)"
    fi
done

echo "All fits complete."
