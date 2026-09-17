import h5py
import numpy as np
import pandas as pd
import glob
import re
import os
import sys
import matplotlib.pyplot as plt

# 1. View a summary CSV of all results:

# python h5py_reader.py

# 2. Plot the profile likelihood for a specific result:

# python h5py_reader.py /home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/fit_Minuit_results_E_2.000.h5


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
            #print("Test ",Eline)
            with h5py.File(file, "r") as f:
                grp = f[f"E_{Eline:.3f}"]
                dm_grp = grp["dm"]

                dm_param_names = dm_grp["param_names"][:].astype(str)
                dm_param_targets = dm_grp["param_targets"][:].astype(str)
                dm_param_values = dm_grp["params"][:]
                dm_param_values = dm_param_values.astype(float)
                dm_params_dict = {
                    f"{n}.{t}": v
                    for n, t, v in zip(dm_param_names, dm_param_targets, dm_param_values)
                }
                print(dm_params_dict)
                dm_norm = None
                for key in dm_params_dict:
                    if key.endswith(".norm") and "gaussian" in key:
                        try:
                            dm_norm = float(dm_params_dict[key])  # ensure float
                        except Exception:
                            dm_norm = np.nan
                        break
                print(dm_norm)
                row = {
                    "Eline (keV)": Eline,
                    "sigma": grp.attrs.get("sigma", np.nan),
                    "Emin": grp.attrs.get("Emin", np.nan),
                    "Emax": grp.attrs.get("Emax", np.nan),
                    "IB lines": grp.attrs.get("Number_of_IB_lines", -1),
                    "TS_as": grp["astro/TS_astro"][()],
                    "nbins_as": grp["astro/nbins"][()],
                    "nfit_as": grp["astro/nfit_astro"][()],
                    "TS_all": dm_grp["TS_all"][()],
                    "ΔTS": dm_grp["deltaTS"][()],
                    "nbins_dm": dm_grp["nbins"][()],
                    "nfit_dm": dm_grp["nfit_dm"][()],
                    "DM norm": float(dm_norm) if dm_norm is not None else np.nan,
                    "DM valid": bool(dm_grp["valid"][()]),
                    "A_95": grp.attrs.get("A_95", np.nan),
                    "A_95_approx": grp.attrs.get("A_95_approx", np.nan),
                    "normlDMhdata2": grp.attrs.get("normlDMhdata2", np.nan),
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


def plot_profile_from_file(filepath):
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
            TS_all = dm_grp["TS_all"][()]

            delta_TS = TS_vals - TS_all

            plt.figure(figsize=(8, 5))
            plt.plot(A_vals, delta_TS, marker='o', linestyle='-')
            plt.axhline(2.71, color='red', linestyle='--', label='95% CL (ΔTS=2.71)')
            plt.xscale("log")
            plt.xlabel("DM Normalization")
            plt.ylabel("ΔTS")
            plt.title(f"TS Profile for E = {Eline:.3f} keV")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error plotting profile from {filepath}: {e}")



# === Run ===
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith(".h5"):
        plot_profile_from_file(sys.argv[1])
    else:
        # Replace this path with your own if needed
        data_path = "/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_test_3deg/Fitpoly/Fit/XSpec/Model4/TMsingle_rebinned/TM7"
        output_file = os.path.join(data_path, "fit_summary.csv")
        summarize_fit_results(data_path, output_csv=output_file)
