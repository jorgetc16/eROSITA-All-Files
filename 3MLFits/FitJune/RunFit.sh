#!/bin/bash

# === Config ===
SCRIPT="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Fitpoly_TMall_MT_intent_APEC.py"
TYPEFIT="2"       # 1 = Minuit, 2 = XSpec, 3 = DiffeV
NSTEPS="12"       # Profile steps
MODEL="5"       # Model type (1 = Powerlaw0+Powerlaw1+Powerlaw2, 2 = bknpowerlaw, 3=Like 1 but with a minus sign on the indices, 4 = Tbabs*(APEC+Powerlaw)), 5 = Tbabs*(APEC+APEC+Powerlaw))
TELESCOPE = "TM1" # Telescope name (TM1, TM2, TM3, TM4, TM6, ALL)
# === Energy List ===
NUM_POINTS=200
EMIN=0.5
EMAX=9

echo "Running fits with the following configuration:"

echo "Type Fit (1=Minuit, 2=XSpec, 3=DiffeV): $TYPEFIT"
echo "Number of Steps in profiling: $NSTEPS"
echo "Model (1=Powerlaw0+Powerlaw1+Powerlaw2, 2=bknpowerlaw, 3=PowerlawsNegative): $MODEL"


ENERGY_LIST=$(awk -v N="$NUM_POINTS" -v EMIN="$EMIN" -v EMAX="$EMAX" 'BEGIN {
    for (i=0; i<N; i++) {
        printf "%.3f ", EMIN + i*(EMAX-EMIN)/(N-1)
    }
}')
# echo "Running fits for $NUM_POINTS energy points between $EMIN and $EMAX keV"

# === Output directory ===
# Set output directory based on TYPEFIT
case "$TYPEFIT" in
    "1")
        case "$MODEL" in
            "1")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/Minuit/Model1"
                ;;
            "2")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/Minuit/Model2"
                ;;
            "3")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/Minuit/Model3"
                ;;
            "4")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/Minuit/Model4"
                ;;
            "5")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/Minuit/Model5"
                ;;
            *)
                echo "Unknown MODEL: $MODEL"
                exit 1
                ;;
        esac
        ;;
    "2")
        case "$MODEL" in
            "1")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/XSpec/Model1_1"
                ;;
            "2")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/XSpec/Model2"
                ;;
            "3")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/XSpec/Model3"
                ;;
            "4")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/XSpec/Model4"
                ;;
            "5")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/XSpec/Model5"
                ;;
            *)
                echo "Unknown MODEL: $MODEL"
                exit 1
                ;;
        esac    
        ;;
    "3")
        case "$MODEL" in
            "1")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/DiffeV/Model1"
                ;;
            "2")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/DiffeV/Model2"
                ;;
            "3")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/DiffeV/Model3"
                ;;
            "4")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/DiffeV/Model4"
                ;;
            "5")
                OUTPUT_DIR="/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/DiffeV/Model5"
                ;;
            *)
                echo "Unknown MODEL: $MODEL"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Unknown TYPEFIT: $TYPEFIT"
        exit 1
        ;;
esac

# Create output directory if it does not exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
else
    echo "Output directory exists: $OUTPUT_DIR"
fi

# === Loop over energies ===
for ELINE in $ENERGY_LIST; do
    echo "-----------------------------"
    echo "Running fit for Eline = $ELINE keV"
    python "$SCRIPT" "$ELINE" "$TYPEFIT" "$NSTEPS" "$OUTPUT_DIR" "$MODEL"> "$OUTPUT_DIR/Fit_${TYPEFIT}_E${ELINE}.log" 2>&1

    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "Fit succeeded for E = $ELINE keV"
    else
        echo "Fit failed for E = $ELINE keV (exit $STATUS)"
    fi
done

echo "All fits complete."