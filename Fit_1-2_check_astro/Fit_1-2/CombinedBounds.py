import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import sys

# Same TS threshold as before
TSth = 2.71

def MTroots(vx, vy):
    vroots = []
    for i in range(1, len(vx)):
        if vy[i] * vy[i - 1] < 0.:
            root = (vx[i] * vy[i - 1] - vx[i - 1] * vy[i]) / (vy[i - 1] - vy[i])
            vroots.append(root)
    return vroots

def ApproximateBound(profile_range, delta_TS):
    x = np.log10(profile_range)
    y = delta_TS - TSth
    vroots = np.power(10., MTroots(x, y))
    if len(vroots) > 0:
        return np.max(vroots)
    else:
        return np.nan   # flag missing bound

def read_dat_files(folder):
    """
    Reads all CombinedProfile_E_xxx.dat files and computes bounds.
    
    Returns
    -------
    energies : list of float
    bounds : list of float
    profiles : dict {Eline: (A_vals, deltaTS)}
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
        A_vals, deltaTS = data[:,0], data[:,1]

        bound = ApproximateBound(A_vals, deltaTS)
        energies.append(Eline)
        bounds.append(bound)
        profiles[Eline] = (A_vals, deltaTS)

    return energies, bounds, profiles

def save_bounds(energies, bounds, outfile):
    arr = np.column_stack([energies, bounds])
    # Use scientific notation for bounds
    np.savetxt(outfile, arr, header="Energy_keV   Bound", fmt="%.6f   %.6e")
    print(f"Bounds saved to {outfile}")


def plot_bounds_vs_energy(energies, bounds, outfile=None):
    plt.figure(figsize=(7,5))
    plt.plot(energies, bounds, marker="o", linestyle="-", color="blue")
    plt.yscale("log")
    plt.xlabel("Energy [keV]")
    plt.ylabel("95% CL Bound on DM Normalization")
    plt.title("DM Bounds vs Energy")
    plt.grid(True, which="both")
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Bounds plot saved to {outfile}")
    else:
        plt.show()

def plot_profile_for_energy(Eline, profiles):
    if Eline not in profiles:
        print(f"No profile found for E={Eline:.3f} keV")
        return
    A_vals, deltaTS = profiles[Eline]
    bound = ApproximateBound(A_vals, deltaTS)

    plt.figure(figsize=(7,5))
    plt.plot(A_vals, deltaTS, marker="o", linestyle="-", color="black", label="Combined Profile")
    plt.axvline(bound, color='blue', linestyle='--', label='95% CL Bound')
    plt.axhline(2.71, color='red', linestyle='--', label='ΔTS=2.71')
    plt.xscale("log")
    plt.xlabel("DM Normalization")
    plt.ylim(-1, 20)
    plt.ylabel("ΔTS (combined)")
    plt.title(f"Combined Profile at E={Eline:.3f} keV")
    plt.legend()
    plt.grid(True)
    plt.show()



# === Run from CLI ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python AnalyzeCombined.py <dat_folder> [plot_profile_energy_keV]")
        sys.exit(0)
    # If user requested a specific profile

    folder = sys.argv[1]
    energies, bounds, profiles = read_dat_files(folder)

    if len(sys.argv) == 3:
        Ereq = float(sys.argv[2])
        plot_profile_for_energy(Ereq, profiles)
        sys.exit(0)

    # Save bounds to file
    bounds_file = os.path.join(folder, "CombinedBounds.dat")
    save_bounds(energies, bounds, bounds_file)

    # Plot bounds vs energy
    plot_file = os.path.join(folder, "CombinedBounds.png")
    plot_bounds_vs_energy(energies, bounds, outfile=plot_file)
    
    
