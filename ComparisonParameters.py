#!/usr/bin/env python
"""
ComparisonParameters.py

Reads fit_results_Astro.h5 from each telescope (TM1–TM7) and plots
each MW-model parameter across telescopes, compared to a reference value
(which can also vary per telescope).

Usage:
    python ComparisonParameters.py <base_directory>

Example:
    python ComparisonParameters.py 1to2_FullRange_APEC321/1to2FullRange_CheckAstro
"""

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
# The MW model parameters, in the order stored by Extract_array_dic
# applied to model_dic (same dict structure used in the Fitpoly scripts).
MW_PARAM_LABELS = [
    ("TBabs",      "nH"),
    ("TBabs_4",    "nH"),
    ("apec",       "kT"),
    ("apec",       "Abundanc"),
    ("apec",       "Redshift"),
    ("apec",       "norm"),
    ("powerlaw",   "PhoIndex"),
    ("powerlaw",   "norm"),
    ("gaussian",   "LineE"),
    ("gaussian",   "Sigma"),
    ("gaussian",   "norm"),
]

TELESCOPES = ["TM1", "TM2", "TM3", "TM4", "TM5", "TM6", "TM7"]

# Reference values to compare against (one per parameter above).
# Each entry is a dict mapping telescope name -> value.
# *** Replace these with your own values ***
REFERENCE_VALUES = {
    ("TBabs",    "nH"):       {"TM1": 0.044, "TM2": 0.044, "TM3": 0.044, "TM4": 0.044, "TM5": 0.044, "TM6": 0.044, "TM7": 0.044},
    ("TBabs_4",  "nH"):       {"TM1": 0.22,  "TM2": 0.22,  "TM3": 0.22,  "TM4": 0.22,  "TM5": 0.22,  "TM6": 0.22,  "TM7": 0.22},
    ("apec",     "kT"):       {"TM1": 0.67,  "TM2": 0.76,  "TM3": 0.7,  "TM4": 0.76,  "TM5": 0.71,  "TM6": 0.66,  "TM7": 0.86},
    ("apec",     "Abundanc"):  {"TM1": 1.0,   "TM2": 1.0,   "TM3": 1.0,   "TM4": 1.0,   "TM5": 1.0,   "TM6": 1.0,   "TM7": 1.0},
    ("apec",     "Redshift"):  {"TM1": 0.0,   "TM2": 0.0,   "TM3": 0.0,   "TM4": 0.0,   "TM5": 0.0,   "TM6": 0.0,   "TM7": 0.0},
    ("apec",     "norm"):      {"TM1": 0.092,  "TM2": 0.065,  "TM3": 0.075,  "TM4": 0.064,  "TM5": 0.047,  "TM6": 0.061,  "TM7": 0.041},
    ("powerlaw", "PhoIndex"):  {"TM1": 1.51,   "TM2": 1.44,   "TM3": 1.50,   "TM4": 1.46,   "TM5": 1.58,   "TM6": 1.47,   "TM7": 1.5},
    ("powerlaw", "norm"):      {"TM1": .37, "TM2": .34, "TM3": .37, "TM4": .35, "TM5": .34, "TM6": .29, "TM7": .36},
    ("gaussian", "LineE"):     {"TM1": 1.5,   "TM2": 1.5,   "TM3": 1.5,   "TM4": 1.5,   "TM5": 1.5,   "TM6": 1.5,   "TM7": 1.5},
    ("gaussian", "Sigma"):     {"TM1": 1e-4,  "TM2": 1e-4,  "TM3": 1e-4,  "TM4": 1e-4,  "TM5": 1e-4,  "TM6": 1e-4,  "TM7": 1e-4},
    ("gaussian", "norm"):      {"TM1": 1e-10, "TM2": 1e-10, "TM3": 1e-10, "TM4": 1e-10, "TM5": 1e-10, "TM6": 1e-10, "TM7": 1e-10},
}

# -------------------------------------------------------------------------
# Read h5 file
# -------------------------------------------------------------------------
def read_fit_astro(folder):
    """Read params_astro_mw from fit_results_Astro.h5 in *folder*."""
    astro_file = os.path.join(folder, "fit_results_Astro.h5")
    with h5py.File(astro_file, "r") as f:
        astro_grp = f["Fit_Astro"]["astro"]
        params_astro_mw = astro_grp["params_astro_mw"][:]
    return params_astro_mw


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python ComparisonParameters.py <base_directory>")
        print("  <base_directory> should contain TM1/ … TM7/ sub-folders,")
        print("  each with a fit_results_Astro.h5 file.")
        sys.exit(1)

    basedir = sys.argv[1]

    # Collect parameters for every telescope
    all_params = {}  # telescope -> numpy array of mw params
    for tm in TELESCOPES:
        tm_dir = os.path.join(basedir, tm)
        if not os.path.isdir(tm_dir):
            print(f"Warning: {tm_dir} not found, skipping {tm}")
            continue
        try:
            all_params[tm] = read_fit_astro(tm_dir)
        except Exception as e:
            print(f"Warning: could not read {tm}: {e}")

    if not all_params:
        print("No data found. Exiting.")
        sys.exit(1)

    available_tms = [tm for tm in TELESCOPES if tm in all_params]
    x = np.arange(len(available_tms))

    # One figure per parameter
    os.makedirs("FiguresImage/ComparisonParams", exist_ok=True)
    # os.makedirs("FiguresPDF/ComparisonParams", exist_ok=True)

    n_params = len(all_params[available_tms[0]])  # number of mw params stored
    n_labels = min(n_params, len(MW_PARAM_LABELS))

    for ip in range(n_labels):
        comp_name, par_name = MW_PARAM_LABELS[ip]
        label = f"{comp_name}.{par_name}"

        vals = np.array([all_params[tm][ip] for tm in available_tms])

        ref_dict = REFERENCE_VALUES.get((comp_name, par_name), None)
        ref_vals = None
        if ref_dict is not None:
            ref_vals = np.array([ref_dict.get(tm, np.nan) for tm in available_tms])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, vals, "o", markersize=4, color="royalblue", label="Ours")

        if ref_vals is not None:
            ax.plot(x, ref_vals, "s", markersize=4, color="crimson", label="eFEDS")

        ax.set_xticks(x)
        ax.set_xticklabels(available_tms)
        ax.set_xlabel("Telescope")
        ax.set_ylabel(label)
        ax.set_title(f"Parameter: {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = f"param_{comp_name}_{par_name}"
        fig.savefig(f"FiguresImage/ComparisonParams/{fname}.png", dpi=150)
        # fig.savefig(f"FiguresPDF/ComparisonParams/{fname}.pdf")
        plt.close(fig)
        print(f"Saved plot for {label}")

    print("Done.")


if __name__ == "__main__":
    main()