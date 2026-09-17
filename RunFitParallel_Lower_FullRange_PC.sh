#!/bin/bash
set -eo pipefail

export LC_ALL=C
export LANG=C

# === Usage ===
usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

OPTIONS:
  --phase astro          Run only astro fit (Phase 1)
  --phase dm             Run only DM scan (Phase 2)
  --phase both           Run both astro fit and DM scan (default)
  --help                 Show this message

Examples:
  $0 --phase astro       # Only fit astro background
  $0 --phase dm          # Only run DM scans (requires existing fit_results_Astro.h5)
  $0 --phase both        # Run both phases (default)
  $0                     # Equivalent to --phase both
EOF
  exit 0
}

# === Default phase ===
PHASE="both"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      PHASE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      ;;
    *)
      echo "ERROR: Unknown option: $1"
      usage
      ;;
  esac
done

# Validate phase
if [[ ! "$PHASE" =~ ^(astro|dm|both)$ ]]; then
  echo "ERROR: --phase must be 'astro', 'dm', or 'both'"
  exit 1
fi

# === Scripts ===
ASTRO_SCRIPT="SCRIPT/Fitpoly_MT_1to2_DE_Astro.py"
DM_SCRIPT="SCRIPT/Fitpoly_MT_1to2_DE_DMprofile.py"

# === Telescopes ===
# TELESCOPE_LIST=(TM5)
TELESCOPE_LIST=(TM1 TM2 TM3 TM4 TM5 TM6 TM7)

# === Output bases ===
OUT_BASE="OUTPUT/1to2_doublepowerlaw_24"

# === Energy List ===
# ENERGY_LIST="1.037 1.085 1.103 1.121 1.146 1.170 1.194 1.218 1.243 1.267 1.291 1.315 1.340 1.364 1.388 1.412 1.436 1.461 1.479 1.491 1.509 1.533 1.558 1.582 1.606 1.630 1.655 1.679 1.703 1.721 1.740 1.758 1.770 1.788 1.812 1.836 1.855 1.867 1.885 1.909 1.933 1.952 1.964 1.982 1.997 2.009 2.030 2.055 2.079 2.097 2.109 2.127 2.146 2.158 2.170 2.188"
ENERGY_LIST="1.000 1.073 1.097 1.109 1.133 1.158 1.182 1.206 1.23 1.255"# 1.279 1.303 1.327 1.352 1.376 1.400 1.424 1.448 1.473 1.485 1.497 1.521 1.545 1.570 1.594 1.618 1.642 1.667 1.691 1.715 1.727 1.752 1.764 1.776 1.800 1.824 1.848 1.861 1.873 1.897 1.921 1.945 1.958 1.970 1.994 2.00 2.018 2.042 2.067 2.091 2.103 2.115 2.139 2.152 2.164 2.176 2.2 2.21 2.225 2.24 2.255 2.27 2.285 2.3 2.315 2.33 2.345 2.36 2.375 2.39 2.4"

MAX_PROCS="${MAX_PROCS:-$(nproc 2>/dev/null || echo 4)}"

# Sanity check: scripts exist
if [[ "$PHASE" == "dm" ]]; then
  if [ ! -f "$DM_SCRIPT" ]; then
    echo "ERROR: DM script not found: $DM_SCRIPT"
    exit 2
  fi
  
  # Check that astro results exist for all telescopes
  for TELESCOPE in "${TELESCOPE_LIST[@]}"; do
    ASTRO_H5="${OUT_BASE}/${TELESCOPE}/fit_results_Astro.h5"
    if [ ! -f "$ASTRO_H5" ]; then
      echo "ERROR: Missing astro H5 for $TELESCOPE: $ASTRO_H5"
      echo "Run with --phase astro first, or --phase both"
      exit 3
    fi
  done
elif [[ "$PHASE" =~ ^(astro|both)$ ]]; then
  if [ ! -f "$ASTRO_SCRIPT" ]; then
    echo "ERROR: Astro script not found: $ASTRO_SCRIPT"
    exit 2
  fi
fi

# Counter files
STARTED_FILE_ASTRO="/tmp/erosita_astro_started.$$"
COMPLETED_FILE_ASTRO="/tmp/erosita_astro_completed.$$"
STARTED_FILE_DM="/tmp/erosita_dm_started.$$"
COMPLETED_FILE_DM="/tmp/erosita_dm_completed.$$"
: > "$STARTED_FILE_ASTRO"
: > "$COMPLETED_FILE_ASTRO"
: > "$STARTED_FILE_DM"
: > "$COMPLETED_FILE_DM"

inc_counter() {
  local file="$1"
  {
    flock -w 5 9 || exit 1
    echo 1 >> "$file"
    wc -l < "$file"
  } 9>>"$file"
}

export ASTRO_SCRIPT DM_SCRIPT OUT_BASE
export STARTED_FILE_ASTRO COMPLETED_FILE_ASTRO STARTED_FILE_DM COMPLETED_FILE_DM
export -f inc_counter

# === Phase 1: Astro fit per telescope ===
run_astro_for_telescope() {
  local TELESCOPE="$1"
  local OUT_DIR="${OUT_BASE}/${TELESCOPE}"
  mkdir -p "$OUT_DIR"

  local STARTED_COUNT
  STARTED_COUNT=$(inc_counter "$STARTED_FILE_ASTRO")
  echo "[$(date +%H:%M:%S)] [ASTRO] Starting: $TELESCOPE (Job $STARTED_COUNT/$TOTAL_JOBS_ASTRO)"

  python "$ASTRO_SCRIPT" "$OUT_DIR" "$TELESCOPE" \
    > "$OUT_DIR/Fit_Astro_${TELESCOPE}.log" 2>&1 || {
    local STATUS=$?
    local COMPLETED_COUNT
    COMPLETED_COUNT=$(inc_counter "$COMPLETED_FILE_ASTRO")
    echo "[$(date +%H:%M:%S)] [ASTRO] FAILED: $TELESCOPE (exit $STATUS) - Job $COMPLETED_COUNT/$TOTAL_JOBS_ASTRO"
    return 1
  }

  local COMPLETED_COUNT
  COMPLETED_COUNT=$(inc_counter "$COMPLETED_FILE_ASTRO")
  echo "[$(date +%H:%M:%S)] [ASTRO] Completed: $TELESCOPE (Job $COMPLETED_COUNT/$TOTAL_JOBS_ASTRO)"
}

export -f run_astro_for_telescope

if [[ "$PHASE" =~ ^(astro|both)$ ]]; then
  TOTAL_JOBS_ASTRO=${#TELESCOPE_LIST[@]}
  echo "Phase 1: Astro fits"
  echo "Telescopes: ${#TELESCOPE_LIST[@]}"
  echo "Max parallel jobs: $MAX_PROCS"
  echo ""

  printf "%s\n" "${TELESCOPE_LIST[@]}" | xargs -n1 -P "$MAX_PROCS" bash -c 'run_astro_for_telescope "$@"' _

  STARTED_COUNT_ASTRO=$(wc -l < "$STARTED_FILE_ASTRO")
  COMPLETED_COUNT_ASTRO=$(wc -l < "$COMPLETED_FILE_ASTRO")
  echo ""
  echo "==== Astro Summary ===="
  echo "Jobs started:   $STARTED_COUNT_ASTRO / $TOTAL_JOBS_ASTRO"
  echo "Jobs completed: $COMPLETED_COUNT_ASTRO / $TOTAL_JOBS_ASTRO"
  echo "========================"
  echo ""
fi

# === Phase 2: DM scan (per telescope × energy) ===
run_dm_scan() {
  local TELESCOPE="$1"
  local ELINE_STR="$2"

  local DM_OUT_DIR="${OUT_BASE}/${TELESCOPE}"
  mkdir -p "$DM_OUT_DIR"

  printf -v ESTR "%.3f" "$ELINE_STR"
  local EFILE="${ESTR//./_}"

  local STARTED_COUNT
  STARTED_COUNT=$(inc_counter "$STARTED_FILE_DM")
  echo "[$(date +%H:%M:%S)] [DM] Starting: $TELESCOPE E=${ESTR} keV (Job $STARTED_COUNT/$TOTAL_JOBS_DM)"

  python "$DM_SCRIPT" "$ESTR" "$DM_OUT_DIR" "$TELESCOPE" \
    > "$DM_OUT_DIR/Fit_DM_${TELESCOPE}_E_${EFILE}.log" 2>&1 || {
    local STATUS=$?
    local COMPLETED_COUNT
    COMPLETED_COUNT=$(inc_counter "$COMPLETED_FILE_DM")
    echo "[$(date +%H:%M:%S)] [DM] FAILED: $TELESCOPE E=${ESTR} keV (exit $STATUS) - Job $COMPLETED_COUNT/$TOTAL_JOBS_DM"
    return 1
  }

  local COMPLETED_COUNT
  COMPLETED_COUNT=$(inc_counter "$COMPLETED_FILE_DM")
  echo "[$(date +%H:%M:%S)] [DM] Completed: $TELESCOPE E=${ESTR} keV (Job $COMPLETED_COUNT/$TOTAL_JOBS_DM)"
}

export -f run_dm_scan

if [[ "$PHASE" =~ ^(dm|both)$ ]]; then
  TOTAL_JOBS_DM=$((${#TELESCOPE_LIST[@]} * $(echo $ENERGY_LIST | wc -w)))
  echo "Phase 2: DM scans"
  echo "Telescopes × energies: ${#TELESCOPE_LIST[@]} × $(echo $ENERGY_LIST | wc -w) = $TOTAL_JOBS_DM"
  echo "Max parallel jobs: $MAX_PROCS"
  echo ""

  for ELINE in $ENERGY_LIST; do
    for TELESCOPE in "${TELESCOPE_LIST[@]}"; do
      echo "$TELESCOPE $ELINE"
    done
  done | xargs -n2 -P "$MAX_PROCS" bash -c 'run_dm_scan "$@"' _

  STARTED_COUNT_DM=$(wc -l < "$STARTED_FILE_DM")
  COMPLETED_COUNT_DM=$(wc -l < "$COMPLETED_FILE_DM")
  echo ""
  echo "==== DM Summary ===="
  echo "Jobs started:   $STARTED_COUNT_DM / $TOTAL_JOBS_DM"
  echo "Jobs completed: $COMPLETED_COUNT_DM / $TOTAL_JOBS_DM"
  echo "All DM scans finished."
  echo "====================="
fi

# Cleanup
rm -f "$STARTED_FILE_ASTRO" "$COMPLETED_FILE_ASTRO" "$STARTED_FILE_DM" "$COMPLETED_FILE_DM"
