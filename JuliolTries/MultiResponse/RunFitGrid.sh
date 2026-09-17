#!/usr/bin/env bash
set -eo pipefail

# ============================================================
# run_fits_smartgrid.sh
#
# Smart driver to run per-telescope XSPEC fits over a resolution-aware
# energy grid, or over a manually specified small set of energies.
#
# Edit the configuration block below or pass variables via the
# environment (TELESCOPE, MODE, TEST_ENERGIES, etc).
# ============================================================

# -----------------------
# === CONFIGURATION ===
# -----------------------

# Path to your main Python fit script (positional args: E_keV OUTDIR TELESCOPE)
SCRIPT="/home/jortecal/GitHub/eRosita/JuliolTries/MultiResponse/Fitpoly_TMall_MT_Multiresponse_Loop_PreFit.py"

# Which telescopes to run (default list)
TELESCOPE_LIST=(TM1 TM2 TM3 TM4 TM5 TM6 TM7)

# Maximum simultaneous telescope jobs (global parallelism)
MAX_PROCS="${MAX_PROCS:-$(nproc 2>/dev/null || echo 4)}"

# Per-telescope concurrency: how many energies per telescope run in parallel
# (useful to limit CPU when telescopes themselves are parallelized)
PER_TELESCOPE_PROCS="${PER_TELESCOPE_PROCS:-4}"

# Path to measured widths table: two columns "energy_keV  fwhm_eV"
WIDTHS_TABLE="/home/jortecal/GitHub/eRosita/3MLFits/Energy_width.txt"

# Path to helper that builds the resolution-aware energy grid
GRID_HELPER="/home/jortecal/GitHub/eRosita/GridSelection.py"

# Energy scan range (keV) — used when MODE=grid
EMIN="${EMIN:-1.0}"
EMAX="${EMAX:-2.0}"

# Grid stepping: fraction of local FWHM (0.5 = one point per half-FWHM)
F_FWHM="${F_FWHM:-0.5}"

# If MODE=grid and you want exactly N energies sampled from the resolution grid,
# set SAMPLE_N > 0. If SAMPLE_N <= 0 we use the full resolution grid.
SAMPLE_N="${SAMPLE_N:-0}"   # e.g. 200 or 0

# Output base directory
OUT_BASE="${OUT_BASE:-/home/jortecal/GitHub/eRosita/JuliolTries/MultiResponse/Results/Proves}"

# Choose driver MODE:
#  - grid    : build resolution-aware grid from WIDTHS_TABLE
#  - list    : use explicit TEST_ENERGIES (space-separated) or TEST_FILE
# Default choose 'grid' if WIDTHS_TABLE present, else 'list' fallback.
MODE="${MODE:-auto}"

# If MODE=list: provide TEST_ENERGIES as space-separated energies (keV)
# Example: TEST_ENERGIES="3.407 3.686"
TEST_ENERGIES="${TEST_ENERGIES:-}"

# Or provide a file with energies, one per line (overrides TEST_ENERGIES if present)
TEST_FILE="${TEST_FILE:-}"

# Extra python args forwarded to your script (optional)
EXTRA_PY_ARGS="${EXTRA_PY_ARGS:-}"

# Verbose diagnostics
VERBOSE="${VERBOSE:-1}"

# -----------------------
# === end configuration ===
# -----------------------

# Helper: print header
echo "=== run_fits_smartgrid.sh starting ==="
echo "SCRIPT: $SCRIPT"
echo "MODE: $MODE"
echo "EMIN..EMAX: $EMIN .. $EMAX keV"
echo "F_FWHM: $F_FWHM  SAMPLE_N: $SAMPLE_N"
echo "MAX_PROCS: $MAX_PROCS   PER_TELESCOPE_PROCS: $PER_TELESCOPE_PROCS"
echo "OUT_BASE: $OUT_BASE"
echo "WIDTHS_TABLE: $WIDTHS_TABLE"
echo "GRID_HELPER: $GRID_HELPER"
echo

# Validate python script
if [ ! -x "$SCRIPT" ] && [ ! -f "$SCRIPT" ]; then
    # some people don't make python script executable; still accept it if readable
    if [ ! -f "$SCRIPT" ]; then
        echo "ERROR: Fit script not found: $SCRIPT" >&2
        exit 2
    fi
fi

# Decide selected telescopes
if [ -n "${TELESCOPE-}" ]; then
    SELECTED_TELESCOPES=("$TELESCOPE")
else
    SELECTED_TELESCOPES=("${TELESCOPE_LIST[@]}")
fi

# Build ENERGY_LIST depending on MODE
build_energy_list_grid() {
    # returns global variable ENERGY_LIST (space separated)
    if [ ! -f "$WIDTHS_TABLE" ]; then
        echo "ERROR: widths table not found: $WIDTHS_TABLE" >&2
        return 1
    fi
    if [ ! -x "$GRID_HELPER" ]; then
        if [ -f "$GRID_HELPER" ]; then
            echo "Note: $GRID_HELPER exists but is not executable; invoking with python3"
            GRID_CMD="python3 $GRID_HELPER"
        else
            echo "ERROR: grid helper not found: $GRID_HELPER" >&2
            return 1
        fi
    else
        GRID_CMD="$GRID_HELPER"
    fi

    # produce full resolution grid string
    FULL_GRID=$($GRID_CMD "$WIDTHS_TABLE" --emin "$EMIN" --emax "$EMAX" --f_fwhm "$F_FWHM" --mode full) || {
        echo "ERROR: grid helper failed" >&2
        return 1
    }

    if [ "${SAMPLE_N:-0}" -gt 0 ]; then
        # sample exactly SAMPLE_N points evenly from FULL_GRID
        ENERGY_LIST=$(python3 - "$SAMPLE_N" <<'PY'
import sys, numpy as np
grid = np.array(sys.stdin.read().split(), dtype=float)
N = int(sys.argv[1])
if N >= len(grid):
    sel = grid
else:
    idx = np.round(np.linspace(0, len(grid)-1, N)).astype(int)
    sel = grid[idx]
print(" ".join([f"{e:.6f}" for e in sel]))
PY
<<<"$FULL_GRID")
    else
        ENERGY_LIST="$FULL_GRID"
    fi
    return 0
}

build_energy_list_list() {
    # If TEST_FILE present, read it; else use TEST_ENERGIES variable.
    if [ -n "$TEST_FILE" ] && [ -f "$TEST_FILE" ]; then
        # read all non-empty non-comment lines
        ENERGY_LIST=$(awk '!/^#/ && NF{printf "%s ", $1}' "$TEST_FILE")
    elif [ -n "$TEST_ENERGIES" ]; then
        ENERGY_LIST="$TEST_ENERGIES"
    else
        echo "ERROR: MODE=list selected but TEST_FILE/TEST_ENERGIES not provided" >&2
        return 1
    fi
    return 0
}

# Determine mode if auto
if [ "$MODE" = "auto" ]; then
    if [ -f "$WIDTHS_TABLE" ] && [ -f "$GRID_HELPER" ]; then
        MODE="grid"
    else
        MODE="list"
    fi
fi

if [ "$MODE" = "grid" ]; then
    build_energy_list_grid || exit 2
elif [ "$MODE" = "list" ]; then
    build_energy_list_list || exit 2
else
    echo "ERROR: unknown MODE: $MODE" >&2
    exit 2
fi

# Final check: ENERGY_LIST non-empty
if [ -z "$ENERGY_LIST" ]; then
    echo "ERROR: ENERGY_LIST is empty (no energies to run)" >&2
    exit 2
fi

echo "Will run on $(echo $ENERGY_LIST | wc -w) energy points."
if [ "$VERBOSE" -ge 1 ]; then
    echo "ENERGIES: $ENERGY_LIST"
fi
echo

# Create output base dir
mkdir -p "$OUT_BASE"

# export variables for subshells
export SCRIPT EXTRA_PY_ARGS

# run per telescope; use xargs to parallelize telescopes (bounded by MAX_PROCS)
run_for_telescope() {
    local TELESCOPE="$1"
    echo
    echo "===================================="
    echo " Starting runs for telescope: $TELESCOPE"
    echo "===================================="

    local OUTPUT_DIR="${OUT_BASE}/${TELESCOPE}"
    mkdir -p "$OUTPUT_DIR"

    # build a temp commands file and run per-energy jobs in parallel
    local CMD_FILE
    CMD_FILE="$(mktemp)"
    trap 'rm -f "$CMD_FILE"' EXIT

    for ELINE in $ENERGY_LIST; do
        printf -v ESTR "%.6f" "$ELINE"
        local EFILE="${ESTR//./_}"
        local LOG="$OUTPUT_DIR/Fit_E${EFILE}.log"
        # Write a safe command line (use bash -c to allow redirection)
        printf "python3 \"%s\" \"%s\" \"%s\" \"%s\" %s > \"%s\" 2>&1 || echo \"[FAIL] %s %s\" >> \"%s/errors.log\"\n" \
            "$SCRIPT" "$ESTR" "$OUTPUT_DIR" "$TELESCOPE" "$EXTRA_PY_ARGS" "$LOG" "$TELESCOPE" "$ESTR" "$OUTPUT_DIR" >> "$CMD_FILE"
    done

    # run parallel commands with xargs -P (fallback to seq if xargs missing)
    if command -v xargs >/dev/null 2>&1; then
        # -d '\n' ensures lines are separate; -I CMD bash -c 'CMD' runs each line in a shell
        xargs -a "$CMD_FILE" -d '\n' -P "${PER_TELESCOPE_PROCS}" -I CMD bash -c 'CMD'
    else
        # portable fallback: run in background batches
        local running=0
        while IFS= read -r cmd_line; do
            bash -c "$cmd_line" &
            running=$((running+1))
            if [ "$running" -ge "$PER_TELESCOPE_PROCS" ]; then
                if wait -n 2>/dev/null; then
                    running=$((running-1))
                else
                    wait
                    running=0
                fi
            fi
        done < "$CMD_FILE"
        wait
    fi

    rm -f "$CMD_FILE"
    echo "All fits complete for $TELESCOPE"
}

# Launch telescopes in parallel (bounded by MAX_PROCS)
if command -v xargs >/dev/null 2>&1; then
    printf "%s\n" "${SELECTED_TELESCOPES[@]}" | xargs -n1 -P "$MAX_PROCS" -I{} bash -c 'run_for_telescope "$@"' _ {}
else
    running=0
    for TELESCOPE in "${SELECTED_TELESCOPES[@]}"; do
        run_for_telescope "$TELESCOPE" &
        running=$((running+1))
        if [ "$running" -ge "$MAX_PROCS" ]; then
            if wait -n 2>/dev/null; then
                running=$((running-1))
            else
                wait
                running=0
            fi
        fi
    done
    wait
fi

echo
echo "All telescope runs finished."
