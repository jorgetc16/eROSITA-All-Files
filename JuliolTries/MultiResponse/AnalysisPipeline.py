"""
Comprehensive analysis pipeline for eROSITA DM search results.
Combines functionality from Results_read.py, CombinedBounds.py, and PlotResults.py

Usage:
  Single dataset:  python AnalysisPipeline.py <base_folder> [output_dir]
  Compare two:     python AnalysisPipeline.py <dataset1> <dataset2> --compare [output_dir]
"""

import h5py
import numpy as np
import pandas as pd
import glob
import re
import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.stats import chi2
from matplotlib import rc
import matplotlib.patheffects as path_effects


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}' #package mathpazo siunitx
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') #serif
plt.rcParams['axes.linewidth'] = 2

# ============================================================================
# CONFIGURATION
# ============================================================================

TSth = 2.71  # TS threshold for 95% CL
TELESCOPES = ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7']

# Plot styling
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def MTroots(vx, vy):
    """Find roots of a function by linear interpolation."""
    vroots = []
    for i in range(1, len(vx)):
        if vy[i] * vy[i - 1] < 0.:
            root = (vx[i] * vy[i - 1] - vx[i - 1] * vy[i]) / (vy[i - 1] - vy[i])
            vroots.append(root)
    return vroots


def approximate_bound(profile_range, delta_TS, eline=None):
    """
    Estimate 95% CL bound from a ΔTS profile.
    
    Parameters
    ----------
    profile_range : array
        Parameter values (A).
    delta_TS : array
        ΔTS profile values.
    eline : float or None
        Energy value (for logging).
    
    Returns
    -------
    bound : float
        Upper bound (A) where ΔTS crosses 2.71.
        Returns np.nan if no crossing is found.
    """
    x = np.log10(profile_range)
    y = delta_TS - TSth
    vroots = np.power(10., MTroots(x, y))
    
    if len(vroots) > 0:
        return np.max(vroots)
    else:
        if eline is not None:
            print(f"  Warning: no ΔTS=2.71 crossing found for E={eline:.3f} keV")
        else:
            print("  Warning: no ΔTS=2.71 crossing found")
        return np.nan


# ============================================================================
# RESULTS_READ FUNCTIONS
# ============================================================================

def summarize_fit_results(folder, output_csv=None):
    """
    Summarize fit results from all .h5 files in a folder.
    
    Parameters
    ----------
    folder : str
        Path to folder containing fit_*.h5 files.
    output_csv : str or None
        Output CSV filename.
    
    Returns
    -------
    df : pandas.DataFrame
        Summary of all fit results.
    """
    result_rows = []
    files = sorted(glob.glob(os.path.join(folder, "fit_*.h5")))
    
    if not files:
        print(f"No HDF5 files found in {folder}")
        return None
    
    for file in files:
        try:
            match = re.search(r"_E_(\d+\.\d+)\.h5", file)
            Eline = float(match.group(1)) if match else np.nan
            
            with h5py.File(file, "r") as f:
                grp = f[f"E_{Eline:.3f}"]
                dm_grp = grp["dm"]
                
                dm_param_values = dm_grp["params"][:]
                dm_param_values = dm_param_values.astype(float)
                dm_norm = dm_param_values[-1] if dm_param_values.size > 0 else np.nan
                
                row = {
                    "Eline (keV)": Eline,
                    "sigma": grp.attrs.get("sigma", np.nan),
                    "Emin": grp.attrs.get("Emin", np.nan),
                    "Emax": grp.attrs.get("Emax", np.nan),
                    "IB lines": grp.attrs.get("Number_of_IB_lines", -1),
                    "TS_as": grp["astro/TS_astro"][()],
                    "nbins_as": grp["astro/nbins"][()],
                    "nfit_as": grp["astro/nfit_astro"][()],
                    "PhoIndex": grp["astro/PhoIndex"][()],
                    "p-value-null": grp["astro/p-value-nullhyp"][()],
                    "TS_all": dm_grp["TS_all"][()],
                    "ΔTS": dm_grp["deltaTS"][()],
                    "nbins_dm": dm_grp["nbins"][()],
                    "nfit_dm": dm_grp["nfit_dm"][()],
                    "DM norm": float(dm_norm) if dm_norm is not None else np.nan,
                    "DM valid": bool(dm_grp["valid"][()]),
                    "A_95": grp.attrs.get("A_95", np.nan),
                    "Date": grp.attrs.get("Date", "n/a")
                }
                result_rows.append(row)
        
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    df = pd.DataFrame(result_rows).sort_values("Eline (keV)")
    print(df.to_string(index=False))
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSummary saved to {output_csv}")
    
    return df


def combine_profiles(filepaths, n_points=500):
    """
    Combine TS profiles from multiple instruments into a single profile.
    
    Parameters
    ----------
    filepaths : list of str
        Paths to .h5 result files for the same energy but different instruments.
    n_points : int
        Number of points for common A grid.
    
    Returns
    -------
    A_common : np.ndarray
        Common DM normalization grid.
    deltaTS_combined : np.ndarray
        Combined ΔTS profile.
    min_TS : float
        Minimum total TS value.
    TS_at0 : float
        TS at A=0 (sum of all TS_astro).
    """
    profiles = []
    A_min, A_max = np.inf, 0
    
    for filepath in filepaths:
        with h5py.File(filepath, "r") as f:
            match = re.search(r"_E_(\d+\.\d+)\.h5", filepath)
            Eline = float(match.group(1)) if match else None
            grp = f[f"E_{Eline:.3f}"]
            
            profile = grp["profile"]
            A_vals = profile["A_values"][:]
            TS_vals = profile["TS_values"][:]
            
            TS_astro = grp["astro/TS_astro"][()]
            if TS_astro > TS_vals[0]:
                print("Warning: TS_astro inconsistent, adjusting.")
                TS_astro = TS_vals[0]
            
            print(f"  Loaded: TS_astro = {TS_astro:.2f}")
            
            A_vals = np.insert(A_vals, 0, 1e-10)
            TS_vals = np.insert(TS_vals, 0, TS_astro)
            
            profiles.append((A_vals, TS_vals))
            A_min = min(A_min, A_vals.min())
            A_max = max(A_max, A_vals.max())
    
    # Create common grid
    A_common_nonzero = np.logspace(np.log10(max(A_min, 1e-10)), np.log10(A_max), n_points)
    
    # Interpolate and sum TS
    TS_sum = np.zeros_like(A_common_nonzero)
    for A_vals, TS_vals in profiles:
        interp = interp1d(A_vals, TS_vals, kind="linear",
                         bounds_error=False, fill_value=np.nan)
        vals = interp(A_common_nonzero)
        vals = np.nan_to_num(vals, nan=np.max(TS_vals))
        TS_sum += vals
    
    # Add A=0 point
    TS_at0 = sum([TS_vals[0] for _, TS_vals in profiles])
    A_common = np.concatenate(([1e-10], A_common_nonzero))
    TS_sum = np.concatenate(([TS_at0], TS_sum))
    
    # Normalize to ΔTS
    min_TS = np.min(TS_sum)
    deltaTS_combined = TS_sum - min_TS
    
    return A_common, deltaTS_combined, min_TS, TS_at0


def save_combined_profiles(base_folder, outfolder, n_points=200):
    """
    Combine TS profiles from all TM folders and save to .dat files.
    
    Parameters
    ----------
    base_folder : str
        Path containing TM1..TM7 subfolders.
    outfolder : str
        Output folder for combined profiles.
    n_points : int
        Number of grid points.
    """
    os.makedirs(outfolder, exist_ok=True)
    minTS_records = []
    
    # Group files by energy
    groups = {}
    for tm in TELESCOPES:
        tm_path = os.path.join(base_folder, tm)
        files = sorted(glob.glob(os.path.join(tm_path, "fit_*.h5")))
        for f in files:
            match = re.search(r"_E_(\d+\.\d+)\.h5", f)
            if match:
                Eline = float(match.group(1))
                groups.setdefault(Eline, []).append(f)
    
    # Process each energy
    for Eline, filelist in sorted(groups.items()):
        if len(filelist) < 2:
            print(f"Skipping E={Eline:.3f} keV ({len(filelist)} file(s))")
            continue
        
        print(f"Combining {len(filelist)} instruments for E={Eline:.3f} keV...")
        A_common, deltaTS_combined, min_TS, TS_at0 = combine_profiles(filelist, n_points=n_points)
        
        # Save profile
        outfile = os.path.join(outfolder, f"CombinedProfile_E_{Eline:.3f}.dat")
        np.savetxt(outfile, np.column_stack([A_common, deltaTS_combined]),
                  header="A_values  DeltaTS_combined")
        
        bound = approximate_bound(A_common, deltaTS_combined, eline=Eline)
        print(f"  -> Bound: {bound:.6e}")
        
        minTS_records.append({"Eline_keV": Eline, "min_TS": min_TS, "TS_at0": TS_at0})
    
    # Save summary
    if minTS_records:
        df_minTS = pd.DataFrame(minTS_records).sort_values("Eline_keV")
        minTS_file = os.path.join(outfolder, "Combined_minTS.csv")
        df_minTS.to_csv(minTS_file, index=False)
        print(f"Saved min TS summary to {minTS_file}")


# ============================================================================
# COMBINED BOUNDS FUNCTIONS
# ============================================================================

def read_dat_files(folder):
    """
    Read all CombinedProfile_E_xxx.dat files and compute bounds.
    
    Parameters
    ----------
    folder : str
        Folder containing .dat files.
    
    Returns
    -------
    energies : list
    bounds : list
    profiles : dict
    """
    energies, bounds = [], []
    profiles = {}
    
    files = sorted(glob.glob(os.path.join(folder, "CombinedProfile_E_*.dat")))
    for f in files:
        match = re.search(r"_E_(\d+\.\d+)\.dat", f)
        if not match:
            continue
        
        Eline = float(match.group(1))
        data = np.loadtxt(f)
        A_vals, deltaTS = data[:, 0], data[:, 1]
        
        bound = approximate_bound(A_vals, deltaTS)
        energies.append(Eline)
        bounds.append(bound)
        profiles[Eline] = (A_vals, deltaTS)
    
    return energies, bounds, profiles


def save_bounds(energies, bounds, outfile):
    """Save bounds to file."""
    arr = np.column_stack([energies, bounds])
    np.savetxt(outfile, arr, header="Energy_keV   Bound", fmt="%.6f   %.6e")
    print(f"Bounds saved to {outfile}")


# ============================================================================
# PLOTTING FUNCTIONS (SINGLE DATASET)
# ============================================================================

def plot_single_pvalue(csv_file, mints_file, outdir="."):
    """
    Plot p-value for single dataset.
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Read per-TM summary (Combined_ALL.csv) and minTS
    df_prefit_raw = pd.read_csv(csv_file)
    df_mints = pd.read_csv(mints_file)
    if 'Eline_keV' in df_mints.columns:
        df_mints.rename(columns={'Eline_keV': 'Eline (keV)'}, inplace=True)

    # --- Combine the 7 TM rows into one per energy (sum TS and dof) ---
    df_prefit = (
        df_prefit_raw
        .groupby('Eline (keV)', as_index=False)
        .agg({
            'TS_as': 'sum',
            'TS_all': 'sum',
            'nbins_as': 'sum',
            'nfit_as': 'sum',
        })
    )

    # Merge with minTS (TS_at0) on energy
    df_plot = df_prefit.merge(df_mints[['Eline (keV)', 'TS_at0']], on='Eline (keV)', how='inner')

    # Compute p-value with summed dof
    dof_as = df_plot['nbins_as'] - df_plot['nfit_as']
    df_plot['p_value_as'] = chi2.sf(df_plot['TS_at0'], dof_as)

    # --- Plot ---
    fig = plt.figure(figsize=(8, 7))
    ax = plt.subplot()
    
    plt.plot(df_plot['Eline (keV)'], df_plot['p_value_as'], 
             marker='o', linestyle='-', markersize=2, color='blue')
    plt.axhline(y=0.05, color='black', linestyle='--', label='p-value = 0.05')
    
    plt.xlabel(r'E (keV)', fontsize=30)
    plt.ylabel(r'p-value', fontsize=30)
    plt.ylim(1e-3, 1)
    plt.xscale('log')
    plt.yscale('log')
    
    ax.tick_params(which='major', direction='in', width=1, length=10, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='y', direction='in', labelsize=22, width=1, length=7, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='x', direction='in', labelsize=22, width=1, length=7, top=True, right=True, pad=10)
    ax.set_xticks([1, 2])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'p-value.pdf'))
    print(f"✓ Saved: {os.path.join(outdir, 'p-value.pdf')}")


def plot_single_deltaTS(mints_file, outdir="."):
    """
    Plot ΔTS for single dataset.
    """
    os.makedirs(outdir, exist_ok=True)
    
    df = pd.read_csv(mints_file)
    
    if 'Eline_keV' in df.columns:
        df.rename(columns={'Eline_keV': 'Eline (keV)'}, inplace=True)
    
    df['DeltaTS'] = df['TS_at0'] - df['min_TS']
    
    fig = plt.figure(figsize=(8, 7))
    ax = plt.subplot()
    
    plt.plot(df['Eline (keV)'], df['DeltaTS'], 
            marker='o', linestyle='-', markersize=2, color='blue')
    plt.axhline(y=9, color='red', linestyle=':', linewidth=2, label=r'3$\sigma$')
    plt.axhline(y=25, color='red', linestyle='--', linewidth=2, label=r'5$\sigma$')
    
    plt.xlabel(r'E (keV)', fontsize=30)
    plt.ylabel(r'$\Delta$TS', fontsize=30)
    plt.xscale('log')
    plt.ylim(-1, max(df['DeltaTS'].max() * 1.2, 30))
    
    ax.tick_params(which='major', direction='in', width=1, length=10, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='y', direction='in', labelsize=22, width=1, length=7, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='x', direction='in', labelsize=22, width=1, length=7, top=True, right=True, pad=10)
    ax.set_xticks([1, 2])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'DeltaTS.pdf'))
    print(f"✓ Saved: {os.path.join(outdir, 'DeltaTS.pdf')}")


def plot_single_bounds(bounds_file, outdir="."):
    """
    Plot upper bounds for single dataset.
    """
    os.makedirs(outdir, exist_ok=True)
    
    df = np.loadtxt(bounds_file)
    
    fig = plt.figure(figsize=(8, 7))
    ax = plt.subplot()
    
    plt.plot(df[:, 0], df[:, 1], label='Upper Bound', marker='o', linestyle='-', markersize=2, color='blue')
    
    plt.xlabel(r'E (keV)', fontsize=30)
    plt.ylabel(r'Upper Bound (ph cm$^{-2}$ s$^{-1}$)', fontsize=30)
    plt.xscale('log')
    plt.yscale('log')
    
    ax.tick_params(which='major', direction='in', width=1, length=10, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='y', direction='in', labelsize=22, width=1, length=7, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='x', direction='in', labelsize=22, width=1, length=7, top=True, right=True, pad=10)
    ax.set_xticks([1, 2])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'UpperBounds.pdf'))
    print(f"✓ Saved: {os.path.join(outdir, 'UpperBounds.pdf')}")


# ============================================================================
# PLOTTING FUNCTIONS (COMPARISON)
# ============================================================================

def plot_pvalue_comparison(csv_file_1, csv_file_2, mints_1, mints_2, outdir="."):
    """Plot p-value comparison between two datasets."""
    os.makedirs(outdir, exist_ok=True)
    
    df_1 = pd.read_csv(csv_file_1)
    df_2 = pd.read_csv(csv_file_2)
    df_mints_1 = pd.read_csv(mints_1)
    df_mints_2 = pd.read_csv(mints_2)
    
    if 'Eline_keV' in df_mints_1.columns:
        df_mints_1.rename(columns={'Eline_keV': 'Eline (keV)'}, inplace=True)
    if 'Eline_keV' in df_mints_2.columns:
        df_mints_2.rename(columns={'Eline_keV': 'Eline (keV)'}, inplace=True)
    
    df_1['p_value_as'] = chi2.sf(df_mints_1['TS_at0'], df_1['nbins_as'] - df_1['nfit_as'])
    df_2['p_value_as'] = chi2.sf(df_mints_2['TS_at0'], df_2['nbins_as'] - df_2['nfit_as'])
    
    fig = plt.figure(figsize=(8, 7))
    ax = plt.subplot()
    
    plt.plot(df_1['Eline (keV)'], df_1['p_value_as'], 
            marker='o', linestyle='-', markersize=2, label='Dataset 1')
    plt.plot(df_2['Eline (keV)'], df_2['p_value_as'], 
            marker='o', linestyle='-.', markersize=2, label='Dataset 2')
    plt.axhline(y=0.05, color='black', linestyle='--', label='p-value = 0.05')
    
    plt.xlabel(r'E (keV)', fontsize=30)
    plt.ylabel(r'p-value', fontsize=30)
    plt.ylim(1e-3, 1)
    plt.xscale('log')
    plt.yscale('log')
    
    ax.tick_params(which='major', direction='in', width=1, length=10, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='y', direction='in', labelsize=22, width=1, length=7, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='x', direction='in', labelsize=22, width=1, length=7, top=True, right=True, pad=10)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'p-value_comparison.pdf'))
    print(f"✓ Saved: {os.path.join(outdir, 'p-value_comparison.pdf')}")


def plot_deltaTS_comparison(mints_1, mints_2, outdir="."):
    """Plot ΔTS comparison between two datasets."""
    os.makedirs(outdir, exist_ok=True)
    
    df_1 = pd.read_csv(mints_1)
    df_2 = pd.read_csv(mints_2)
    
    if 'Eline_keV' in df_1.columns:
        df_1.rename(columns={'Eline_keV': 'Eline (keV)'}, inplace=True)
    if 'Eline_keV' in df_2.columns:
        df_2.rename(columns={'Eline_keV': 'Eline (keV)'}, inplace=True)
    
    df_1['DeltaTS'] = df_1['TS_at0'] - df_1['min_TS']
    df_2['DeltaTS'] = df_2['TS_at0'] - df_2['min_TS']
    
    fig = plt.figure(figsize=(8, 7))
    ax = plt.subplot()
    
    plt.plot(df_1['Eline (keV)'], df_1['DeltaTS'], 
            marker='o', linestyle='-', markersize=2, label='Dataset 1')
    plt.plot(df_2['Eline (keV)'], df_2['DeltaTS'], 
            marker='o', linestyle='-.', markersize=2, label='Dataset 2')
    plt.axhline(y=9, color='red', linestyle=':', label='3σ')
    plt.axhline(y=25, color='red', linestyle='--', label='5σ')
    
    plt.xlabel(r'E (keV)', fontsize=30)
    plt.ylabel(r'$\Delta$TS', fontsize=30)
    plt.xscale('log')
    plt.ylim(-1, 37)
    
    ax.tick_params(which='major', direction='in', width=1, length=10, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='y', direction='in', labelsize=22, width=1, length=7, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='x', direction='in', labelsize=22, width=1, length=7, top=True, right=True, pad=10)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'DeltaTS_comparison.pdf'))
    print(f"✓ Saved: {os.path.join(outdir, 'DeltaTS_comparison.pdf')}")


def plot_bounds_comparison(bounds_file_1, bounds_file_2, outdir="."):
    """Plot upper bounds comparison."""
    os.makedirs(outdir, exist_ok=True)
    
    df_1 = np.loadtxt(bounds_file_1)
    df_2 = np.loadtxt(bounds_file_2)
    
    fig = plt.figure(figsize=(8, 7))
    ax = plt.subplot()
    
    plt.plot(df_1[:, 0], df_1[:, 1], label='Dataset 1', marker='o', linestyle='-', markersize=2)
    plt.plot(df_2[:, 0], df_2[:, 1], label='Dataset 2', marker='o', linestyle='-.', markersize=2)
    
    plt.xlabel(r'E (keV)', fontsize=30)
    plt.ylabel(r'Upper Bound (ph cm$^{-2}$ s$^{-1}$)', fontsize=30)
    plt.xscale('log')
    plt.yscale('log')
    
    ax.tick_params(which='major', direction='in', width=1, length=10, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='y', direction='in', labelsize=22, width=1, length=7, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='x', direction='in', labelsize=22, width=1, length=7, top=True, right=True, pad=10)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'UpperBounds_comparison.pdf'))
    print(f"✓ Saved: {os.path.join(outdir, 'UpperBounds_comparison.pdf')}")


# ============================================================================
# MAIN PIPELINES
# ============================================================================

def run_single_dataset_pipeline(base_folder, output_dir="."):
    """Run analysis for single dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STEP 1: SUMMARIZE FIT RESULTS")
    print("=" * 60)
    
    # Summarize results for each TM
    df_all = []
    for tm in TELESCOPES:
        tm_path = os.path.join(base_folder, tm)
        if os.path.exists(tm_path):
            print(f"\nProcessing {tm}")
            output_csv = os.path.join(output_dir, f"{tm}_summary.csv")
            df = summarize_fit_results(tm_path, output_csv=output_csv)
            if df is not None:
                df_all.append(df)
    
    # Combine all TM results
    if df_all:
        df_combined = pd.concat(df_all, ignore_index=True)
        df_combined = df_combined.sort_values("Eline (keV)")
        df_combined_path = os.path.join(output_dir, "Combined_ALL.csv")
        df_combined.to_csv(df_combined_path, index=False)
        print(f"\n✓ Saved combined results to {df_combined_path}")
    else:
        print("No results found!")
        return
    
    print("\n" + "=" * 60)
    print("STEP 2: COMBINE PROFILES")
    print("=" * 60)
    
    combined_dir = os.path.join(output_dir, "CombinedProfiles")
    save_combined_profiles(base_folder, combined_dir)
    
    print("\n" + "=" * 60)
    print("STEP 3: COMPUTE BOUNDS")
    print("=" * 60)
    
    energies, bounds, profiles = read_dat_files(combined_dir)
    bounds_file = os.path.join(combined_dir, "CombinedBounds.dat")
    save_bounds(energies, bounds, bounds_file)
    
    print("\n" + "=" * 60)
    print("STEP 4: GENERATE PLOTS")
    print("=" * 60)
    
    plots_dir = os.path.join(output_dir, "Plots")
    mints_file = os.path.join(combined_dir, "Combined_minTS.csv")
    
    print("\nPlotting p-values...")
    plot_single_pvalue(df_combined_path, mints_file, plots_dir)
    
    print("Plotting ΔTS...")
    plot_single_deltaTS(mints_file, plots_dir)
    
    print("Plotting bounds...")
    plot_single_bounds(bounds_file, plots_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")


def run_comparison_pipeline(base_folder_1, base_folder_2, output_dir="."):
    """Run analysis for two datasets with comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STEP 1: SUMMARIZE FIT RESULTS")
    print("=" * 60)
    
    # Dataset 1
    dataset1_dir = os.path.join(output_dir, "Dataset1")
    os.makedirs(dataset1_dir, exist_ok=True)
    
    df1_all = []
    for tm in TELESCOPES:
        tm_path = os.path.join(base_folder_1, tm)
        if os.path.exists(tm_path):
            print(f"\nProcessing Dataset 1 - {tm}")
            output_csv = os.path.join(dataset1_dir, f"{tm}_summary.csv")
            df = summarize_fit_results(tm_path, output_csv=output_csv)
            if df is not None:
                df1_all.append(df)
    
    if df1_all:
        df1_combined = pd.concat(df1_all, ignore_index=True)
        df1_combined = df1_combined.sort_values("Eline (keV)")
        df1_combined_path = os.path.join(dataset1_dir, "Combined_ALL.csv")
        df1_combined.to_csv(df1_combined_path, index=False)
        print(f"\n✓ Saved Dataset 1 to {df1_combined_path}")
    
    # Dataset 2
    dataset2_dir = os.path.join(output_dir, "Dataset2")
    os.makedirs(dataset2_dir, exist_ok=True)
    
    df2_all = []
    for tm in TELESCOPES:
        tm_path = os.path.join(base_folder_2, tm)
        if os.path.exists(tm_path):
            print(f"\nProcessing Dataset 2 - {tm}")
            output_csv = os.path.join(dataset2_dir, f"{tm}_summary.csv")
            df = summarize_fit_results(tm_path, output_csv=output_csv)
            if df is not None:
                df2_all.append(df)
    
    if df2_all:
        df2_combined = pd.concat(df2_all, ignore_index=True)
        df2_combined = df2_combined.sort_values("Eline (keV)")
        df2_combined_path = os.path.join(dataset2_dir, "Combined_ALL.csv")
        df2_combined.to_csv(df2_combined_path, index=False)
        print(f"\n✓ Saved Dataset 2 to {df2_combined_path}")
    
    print("\n" + "=" * 60)
    print("STEP 2: COMBINE PROFILES")
    print("=" * 60)
    
    combined1_dir = os.path.join(dataset1_dir, "CombinedProfiles")
    combined2_dir = os.path.join(dataset2_dir, "CombinedProfiles")
    
    print("\nCombining Dataset 1 profiles...")
    save_combined_profiles(base_folder_1, combined1_dir)
    
    print("\nCombining Dataset 2 profiles...")
    save_combined_profiles(base_folder_2, combined2_dir)
    
    print("\n" + "=" * 60)
    print("STEP 3: COMPUTE BOUNDS")
    print("=" * 60)
    
    energies1, bounds1, _ = read_dat_files(combined1_dir)
    bounds_file1 = os.path.join(combined1_dir, "CombinedBounds.dat")
    save_bounds(energies1, bounds1, bounds_file1)
    
    energies2, bounds2, _ = read_dat_files(combined2_dir)
    bounds_file2 = os.path.join(combined2_dir, "CombinedBounds.dat")
    save_bounds(energies2, bounds2, bounds_file2)
    
    print("\n" + "=" * 60)
    print("STEP 4: GENERATE COMPARISON PLOTS")
    print("=" * 60)
    
    plots_dir = os.path.join(output_dir, "Plots")
    mints1 = os.path.join(combined1_dir, "Combined_minTS.csv")
    mints2 = os.path.join(combined2_dir, "Combined_minTS.csv")
    
    print("\nPlotting p-value comparison...")
    plot_pvalue_comparison(df1_combined_path, df2_combined_path, mints1, mints2, plots_dir)
    
    print("Plotting ΔTS comparison...")
    plot_deltaTS_comparison(mints1, mints2, plots_dir)
    
    print("Plotting bounds comparison...")
    plot_bounds_comparison(bounds_file1, bounds_file2, plots_dir)
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="eROSITA DM search analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single dataset:
    python AnalysisPipeline.py /path/to/base_folder -o ./results
  
  Compare two datasets:
    python AnalysisPipeline.py /path/to/dataset1 /path/to/dataset2 --compare -o ./comparison
        """
    )
    
    parser.add_argument("dataset1", help="Path to first dataset (base folder with TM1..TM7)")
    parser.add_argument("dataset2", nargs='?', help="Path to second dataset (optional)")
    parser.add_argument("-c", "--compare", action='store_true', help="Compare two datasets")
    parser.add_argument("-o", "--output", default="./AnalysisResults", help="Output directory")
    
    args = parser.parse_args()
    
    if args.compare or args.dataset2:
        if not args.dataset2:
            print("Error: --compare requires two dataset paths")
            sys.exit(1)
        run_comparison_pipeline(args.dataset1, args.dataset2, args.output)
    else:
        run_single_dataset_pipeline(args.dataset1, args.output)