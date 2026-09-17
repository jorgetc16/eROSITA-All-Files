"""
Parse the output of the HEASARC nH tool (w3nh.pl) into a pandas DataFrame,
and produce a scatter map + a regular gridded map of nH around the LMC.

Usage:
    1. Save the full raw output of the nH query (all header/warning lines
       AND all data rows) into a plain text file, e.g. 'nh_output.txt'.
       (Select-all + copy from the results page, or save the page source.)
    2. Run: python parse_nh_map.py nh_output.txt
"""

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle  # add this at the top


def parse_nh_file(filepath):
    """
    Extract RA, DEC, Dist, nH columns from a w3nh.pl output file.
    Lines that are not data rows (headers, warnings, box info, etc.)
    are silently skipped because they don't match the numeric pattern.
    """
    pattern = re.compile(
        r"^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+([\d.]+[Ee][+-]?\d+)"
    )

    ra, dec, dist, nh = [], [], [], []
    with open(filepath, "r") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                ra.append(float(m.group(1)))
                dec.append(float(m.group(2)))
                dist.append(float(m.group(3)))
                nh.append(float(m.group(4)))

    if not ra:
        raise ValueError(
            "No data rows matched — check the file actually contains the "
            "'RA DEC Dist nH' table rows (not just the header)."
        )

    df = pd.DataFrame({"RA": ra, "DEC": dec, "Dist": dist, "nH": nh})
    return df


def plot_scatter_map(df, outpath="nH_scatter_map.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df["RA"], df["DEC"], c=df["nH"], cmap="viridis", s=18)
    plt.colorbar(sc, ax=ax, label=r"$n_H$ [cm$^{-2}$]")

    # LMC 5 deg radius circle
    lmc_ra, lmc_dec = 80.260, -69.260
    circle = Circle((lmc_ra, lmc_dec), 5.0,
                    fill=False, edgecolor="red", linewidth=1.5,
                    linestyle="--", label="LMC (5°)")
    ax.add_patch(circle)
    ax.plot(lmc_ra, lmc_dec, "rx", markersize=5)

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("DEC [deg]")
    ax.invert_xaxis()
    ax.set_title("HI column density around the LMC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")


def plot_gridded_map(df, pixel_size=0.083, outpath="nH_grid_map.png"):
    """
    Use Delaunay triangulation for scattered data.
    Handles the radial sampling pattern of the HEASARC nH tool.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.tricontourf(df["RA"], df["DEC"], df["nH"],
                        levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, label=r"$n_H$ [cm$^{-2}$]")

    # LMC 5 deg radius circle
    lmc_ra, lmc_dec = 80.260, -69.260
    circle = Circle((lmc_ra, lmc_dec), 5.0,
                    fill=False, edgecolor="red", linewidth=1.5,
                    linestyle="--", label="LMC (5°)")
    ax.add_patch(circle)
    ax.plot(lmc_ra, lmc_dec, "rx", markersize=5)  # center marker

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("DEC [deg]")
    ax.invert_xaxis()  # astronomical convention
    ax.set_title("HI column density around the LMC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")


def plot_nH_distribution(df, outpath="nH_distribution.png"):
    """Plot a histogram of nH values."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["nH"] / 1e21, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(df["nH"].mean() / 1e21, color="red", linestyle="--",
               label=f"Mean: {df['nH'].mean():.2e}")
    ax.axvline(df["nH"].median() / 1e21, color="orange", linestyle=":",
               label=f"Median: {df['nH'].median():.2e}")
    ax.set_xlabel(r"$n_H$ [$10^{21}$ cm$^{-2}$]")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of HI column density around the LMC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")

    
if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "nh_output.txt"

    df = parse_nh_file(filepath)
    print(f"Parsed {len(df)} data points.")
    print(df.describe())
    print(f"\nMean nH (weighted equally per pixel): {df['nH'].mean():.3e} cm^-2")

    # Optional: restrict to points within the requested 5 deg radius only
    df_5deg = df[df["Dist"] <= 5.0]
    print(f"Mean nH within 5 deg: {df_5deg['nH'].mean():.3e} cm^-2")

    plot_scatter_map(df)
    try:
        plot_gridded_map(df)
    except Exception as e:
        print(f"Gridded map skipped (grid not perfectly regular?): {e}")

    plot_nH_distribution(df)