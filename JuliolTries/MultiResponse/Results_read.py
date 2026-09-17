import h5py
import numpy as np
import pandas as pd
import glob
import re
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

# 1. View a summary CSV of all results:

# python Results_read.py

# 2. Plot the profile likelihood for a specific result:

# python /home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results_read.py /home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/fit_Minuit_results_E_2.000.h5


def summarize_fit_results(folder, output_csv=None):
    result_rows = []
    files = sorted(glob.glob(os.path.join(folder, "fit_*.h5")))
    if not files:
        print("No HDF5 files found in", folder)
        return

    for file in files:
        try:
            match = re.search(r"_E_(\d+\.\d+)\.h5", file)
            Eline = float(match.group(1)) if match else np.nan

            with h5py.File(file, "r") as f:
                grp = f[f"E_{Eline:.3f}"]
                dm_grp = grp["dm"]

                # Read raw parameter array (no names)
                dm_param_values = dm_grp["params"][:]
                dm_param_values = dm_param_values.astype(float)

                # Heuristic: assume last value is DM norm (common pattern)
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





TSth = 2.71



def MTroots(vx,vy):
    vroots = []
    for i in range(1,len(vx)):
        #print("vx ",vx[i]," ",vy[i])
        #Check whether sign has changed
        if(vy[i]*vy[i-1]<0.):
            root=(vx[i]*vy[i-1]-vx[i-1]*vy[i])/(vy[i-1]-vy[i])
            #print(vx[i-1]," ",vx[i]," ",vy[i-1]," ",vy[i])
            #print("i ", i," ",root," ")
            vroots.append(root)
    return  vroots


def ApproximateBound(profile_range, delta_TS, eline=None):
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

def plot_profile_from_file(filepath):
    """Plot profile for a single file (original functionality)"""
    try:
        match = re.search(r"_E_(\d+\.\d+)\.h5", filepath)
        Eline = float(match.group(1)) if match else None
        if Eline is None:
            print("Could not extract Eline from filename.")
            return
        group_name = f"E_{Eline:.3f}"

        with h5py.File(filepath, "r") as f:
            grp = f[group_name]
            dm_grp = grp["dm"]
            profile = grp["profile"]

            A_vals = profile["A_values"][:]
            TS_vals = profile["TS_values"][:]
            TS_vals_smooth = uniform_filter1d(TS_vals, size=3, mode='nearest')
            TS_all = np.min(TS_vals)

            delta_TS = TS_vals- TS_all
            bound = ApproximateBound(A_vals, delta_TS)
            print(f"Approximate bound for E = {Eline:.3f} keV: {bound:.6f}")
            plt.figure(figsize=(8, 5))
            plt.plot(A_vals, TS_vals, marker='o', linestyle='-')
            plt.axvline(bound, color='black', linestyle='--', label='ΔTS = 0')
            plt.axhline(2.71+min(TS_vals), color='red', linestyle='--', label='95% CL (ΔTS=2.71)')
            plt.xscale("log")
            plt.xlabel("DM Normalization")
            plt.ylabel("TS")
            plt.ylim(min(TS_vals)-1, min(TS_vals)+30)
            plt.title(f"TS Profile for E = {Eline:.3f} keV")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error plotting profile from {filepath}: {e}")


def plot_all_TM_profiles(base_folder, Eline):
    """
    Plot TS profiles for all 7 TMs at a given energy point, including combined profile.
    
    Parameters
    ----------
    base_folder : str
        Path to folder containing TM1..TM7 subfolders
    Eline : float
        Energy point in keV
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    found_any = False
    filepaths = []
    
    for tm_num in range(1, 8):
        tm_folder = os.path.join(base_folder, f"TM{tm_num}")
        filepath = os.path.join(tm_folder, f"fit_results_E_{Eline:.3f}.h5")
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping TM{tm_num}")
            continue
        
        try:
            group_name = f"E_{Eline:.3f}"
            
            with h5py.File(filepath, "r") as f:
                grp = f[group_name]
                profile = grp["profile"]
                
                A_vals = profile["A_values"][:]
                TS_vals = profile["TS_values"][:]
                
                # Get TS_astro (TS at A=0)
                TS_astro = grp["astro/TS_astro"][()]
                if TS_astro > TS_vals[0]:
                    print("Warning: TS_astro inconsistent with profile, adjusting.")
                    TS_astro = TS_vals[0]  # safeguard against inconsistencies
                # Insert A=1e-10 with TS_astro at the beginning
                A_vals = np.insert(A_vals, 0, 1e-10)
                TS_vals = np.insert(TS_vals, 0, TS_astro)
                
                # Compute ΔTS
                TS_min = np.min(TS_vals)
                delta_TS = TS_vals - TS_min
                
                # Plot individual TM
                ax.plot(A_vals, delta_TS, marker='o', linestyle='-', 
                       color=colors[tm_num-1], label=f'TM{tm_num}', 
                       linewidth=1.5, markersize=3, alpha=0.6)
                
                # Compute and print bound (excluding the A=1e-10 point)
                bound = ApproximateBound(A_vals[1:], delta_TS[1:])
                print(f"TM{tm_num}: Bound = {bound:.6e}, TS_astro = {TS_astro:.2f}")
                
                found_any = True
                filepaths.append(filepath)
                
        except Exception as e:
            print(f"Error reading TM{tm_num}: {e}")
    
    if not found_any:
        print(f"No valid profiles found for E = {Eline:.3f} keV")
        plt.close(fig)
        return
    
    # Add combined profile if we have at least 2 TMs
    if len(filepaths) >= 2:
        try:
            print("\nComputing combined profile...")
            A_common, deltaTS_combined, min_TS, TS_at0 = combine_profiles(filepaths)
            
            # Plot combined profile with emphasis
            ax.plot(A_common, deltaTS_combined, 
                   color='black', linewidth=3, linestyle='-', 
                   label='Combined (all TMs)', zorder=10)
            
            # Compute and print combined bound (excluding A=1e-10)
            bound_combined = ApproximateBound(A_common[1:], deltaTS_combined[1:])
            print(f"\nCombined: Bound = {bound_combined:.6e}")
            print(f"Combined: min TS = {min_TS:.2f}, TS at A=0 = {TS_at0:.2f}")
            print(f"Combined: sqrt(ΔTS) at A=1e-10 = {np.sqrt(deltaTS_combined[0]):.2f}")
            
            # Add vertical line for combined bound
            ax.axvline(bound_combined, color='black', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'95% CL Bound ({bound_combined:.2e})')
            
        except Exception as e:
            print(f"Error computing combined profile: {e}")
    
    # Add reference lines
    ax.axhline(2.71, color='red', linestyle='--', linewidth=2, 
              label='95% CL (ΔTS=2.71)', zorder=5)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Format plot
    ax.set_xscale("log")
    ax.set_xlabel('DM Normalization', fontsize=12)
    ax.set_ylabel('ΔTS', fontsize=12)
    ax.set_title(f'TS Profiles for All TMs at E = {Eline:.3f} keV', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, ncol=2)
    
    # Auto-adjust y-limits
    if len(filepaths) >= 2:
        ax.set_ylim(-0.5, min(30, np.percentile(deltaTS_combined, 95) * 1.2))
    else:
        ax.set_ylim(-0.5, 30)
    
    plt.tight_layout()
    plt.show()


def group_files_by_energy(base_folder):
    """
    Traverse subfolders TM1..TM7 under base_folder and group .h5 files by Eline.
    
    Returns
    -------
    dict : {Eline : [file1, file2, ...]}
    """
    groups = {}
    tm_folders = sorted(glob.glob(os.path.join(base_folder, "TM*")))

    for tm in tm_folders:
        files = sorted(glob.glob(os.path.join(tm, "fit_*.h5")))
        for f in files:
            match = re.search(r"_E_(\d+\.\d+)\.h5", f)
            if match:
                Eline = float(match.group(1))
                groups.setdefault(Eline, []).append(f)
    return groups


def save_combined_profiles(base_folder, outfolder, n_points=200):
    """
    Combine TS profiles from all TM folders and save to .dat files.
    Also saves a CSV file with min TS for each energy.
    """
    os.makedirs(outfolder, exist_ok=True)
    groups = group_files_by_energy(base_folder)

    minTS_records = []

    for Eline, filelist in groups.items():
        if len(filelist) < 2:
            print(f"Skipping E={Eline:.3f} keV (only {len(filelist)} file(s) found)")
            continue

        print(f"Combining {len(filelist)} instruments for E={Eline:.3f} keV...")
        A_common, deltaTS_combined, min_TS, TS_at0 = combine_profiles(filelist, n_points=n_points)

        # Save combined profile (ΔTS curve)
        outfile = os.path.join(outfolder, f"CombinedProfile_E_{Eline:.3f}.dat")
        np.savetxt(outfile, np.column_stack([A_common, deltaTS_combined]),
                   header="A_values  DeltaTS_combined")

        # Report approximate bound
        bound = ApproximateBound(A_common, deltaTS_combined, eline=Eline)
        print(f"  -> Bound: {bound:.6e}, saved profile to {outfile}")

        # Collect min TS info
        minTS_records.append({"Eline_keV": Eline, "min_TS": min_TS, "TS_at0": TS_at0})
        
        

    # Save summary CSV with min TS values and at A=0 TS
    if minTS_records:
        df_minTS = pd.DataFrame(minTS_records).sort_values("Eline_keV")
        minTS_file = os.path.join(outfolder, "Combined_minTS.csv")
        df_minTS.to_csv(minTS_file, index=False)
        print(f"Saved min TS summary to {minTS_file}")




def combine_profiles(filepaths, n_points=500):
    """
    Combine TS profiles from multiple instruments into a single, properly normalized profile.
    Works directly with TS values (not ΔTS).
    Explicitly includes the astro-only best fit (A=0, TS_astro) for each instrument.

    Parameters
    ----------
    filepaths : list of str
        Paths to the .h5 result files for the same energy point but different instruments.
    n_points : int
        Number of points for the common A grid.

    Returns
    -------
    A_common : np.ndarray
        Common DM normalization grid (log-spaced, including A=0).
    deltaTS_combined : np.ndarray
        Combined ΔTS profile, normalized so min = 0.
    min_TS : float
        Minimum total TS value (i.e. at the combined best fit).
    """
    profiles = []
    A_min, A_max = np.inf, 0

    # Collect all profiles
    for filepath in filepaths:
        with h5py.File(filepath, "r") as f:
            match = re.search(r"_E_(\d+\.\d+)\.h5", filepath)
            Eline = float(match.group(1)) if match else None
            grp = f[f"E_{Eline:.3f}"]

            profile = grp["profile"]
            A_vals = profile["A_values"][:]
            TS_vals = profile["TS_values"][:]   # raw TS(A)

            # Get TS at A=0 from astro-only fit
            TS_astro = grp["astro/TS_astro"][()]
            if TS_astro > TS_vals[0]:
                print("Warning: TS_astro inconsistent with profile, adjusting.")
                TS_astro = TS_vals[0]  # safeguard against inconsistencies
            print(f"  Loaded {filepath}: TS_astro = {TS_astro:.2f}")
            # Insert A=0, TS=TS_astro at the beginning
            A_vals = np.insert(A_vals, 0, 1e-10)
            TS_vals = np.insert(TS_vals, 0, TS_astro)
            TS_vals_smooth = uniform_filter1d(TS_vals, size=3, mode='nearest')
            profiles.append((A_vals, TS_vals))
            A_min = min(A_min, A_vals.min())
            A_max = max(A_max, A_vals.max())

    # Common log-spaced grid from smallest nonzero A to max
    # We'll treat A=0 separately
    A_common_nonzero = np.logspace(np.log10(max(A_min, 1e-10)), np.log10(A_max), n_points)

    # Interpolate and sum TS values
    TS_sum = np.zeros_like(A_common_nonzero)
    for A_vals, TS_vals in profiles:
        interp = interp1d(A_vals, TS_vals, kind="linear",
                          bounds_error=False, fill_value=np.nan)
        vals = interp(A_common_nonzero)
        vals = np.nan_to_num(vals, nan=np.max(TS_vals))
        TS_sum += vals

    # Add the A=0 point (sum of all TS_astro)
    TS_at0 = sum([TS_vals[0] for _, TS_vals in profiles])
    A_common = np.concatenate(([1e-10], A_common_nonzero))
    TS_sum   = np.concatenate(([TS_at0], TS_sum))

    # Normalize: ΔTS = TS_tot - min(TS_tot)
    min_TS = np.min(TS_sum)
    
    deltaTS_combined = TS_sum - min_TS

    return A_common, deltaTS_combined, min_TS, TS_at0


def plot_combined_profile(filepaths):
    A_common, deltaTS_combined, min_TS, TS_at0 = combine_profiles(filepaths)

    bound = ApproximateBound(A_common, deltaTS_combined)
    print(f"Combined bound: {bound:.6e}")

    plt.figure(figsize=(8, 5))
    plt.plot(A_common, deltaTS_combined, label="Combined ΔTS", color="blue")
    plt.axvline(bound, color='black', linestyle='--', label='95% CL Bound')
    plt.axhline(2.71, color='red', linestyle='--', label='95% CL (ΔTS=2.71)')
    plt.xscale("log")
    plt.xlabel("DM Normalization")
    plt.ylabel("ΔTS (combined)")
    plt.title("Combined TS Profile")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def dm_norm_range_across_TMs(base_folder, output_file=None):
    """
    For each energy point, collect best-fit DM normalizations across TMs,
    and compute min and max.

    Parameters
    ----------
    base_folder : str
        Path containing TM1..TM7 subfolders with .h5 files.
    output_file : str or None
        If given, save results as CSV file.

    Returns
    -------
    df_ranges : pandas.DataFrame
        Table with columns: [Eline (keV), DM_min, DM_max]
    """
    # Collect all fit results across TM folders
    all_rows = []
    tm_folders = sorted(glob.glob(os.path.join(base_folder, "TM*")))
    for tm in tm_folders:
        df = summarize_fit_results(tm)  # your existing function
        if df is not None:
            all_rows.append(df)

    if not all_rows:
        print("No results found.")
        return None

    df_all = pd.concat(all_rows, ignore_index=True)

    # Group by energy and compute min/max of DM norm
    df_ranges = df_all.groupby("Eline (keV)")["DM norm"].agg(["min", "max"]).reset_index()
    df_ranges.rename(columns={"min": "DM_min", "max": "DM_max"}, inplace=True)

    if output_file:
        df_ranges.to_csv(output_file, index=False)
        print(f"Saved DM norm ranges to {output_file}")

    return df_ranges

# === Run ===
if __name__ == "__main__":
    # Check for no arguments FIRST (before accessing sys.argv[1])
    if len(sys.argv) < 2:
        # Default: summarize results for all TM folders
        base_path = "/home/jortecal/GitHub/eRosita/LMC5Deg/ResultsNewData/2to9"
        for tm_num in range(1, 8):  # TM1 through TM7
            data_path = os.path.join(base_path, f"TM{tm_num}")
            
            if not os.path.exists(data_path):
                print(f"Warning: {data_path} does not exist, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing TM{tm_num}")
            print(f"{'='*60}")
            
            output_file = os.path.join(data_path, "fit_summary.csv")
            summarize_fit_results(data_path, output_csv=output_file)
    
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".h5"):
        # Plot a single instrument profile
        plot_profile_from_file(sys.argv[1])
    
    elif sys.argv[1] == "plot_all":
        # Plot all TMs for a given energy
        base_path = sys.argv[2]
        Eline = float(sys.argv[3])
        plot_all_TM_profiles(base_path, Eline)

    elif sys.argv[1] == "combine":
        # Combine all instruments for each energy point
        folder = sys.argv[2]
        outfolder = sys.argv[3]
        save_combined_profiles(folder, outfolder)

    elif sys.argv[1] == "dm_norm":
        # Compute DM normalization ranges across TMs
        base_folder = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        dm_norm_range_across_TMs(base_folder, output_file=output_file)
    
    else:
        print("Unknown command. Usage:")
        print("  python Results_read.py                              # Summarize all TMs")
        print("  python Results_read.py file.h5                      # Plot single profile")
        print("  python Results_read.py plot_all <base_path> <Eline> # Plot all TMs")
        print("  python Results_read.py combine <folder> <outfolder> # Combine profiles")
        print("  python Results_read.py dm_norm <folder> [outfile]   # DM norm ranges")