import h5py
import numpy as np
import pandas as pd
import glob
import re
import os

def summarize_fit_results(folder, output_csv=None):
    result_rows = []

    # Match files like "fit_results_E_2.000.h5"
    files = sorted(glob.glob(os.path.join(folder, "fit_*.h5")))

    if not files:
        print("No HDF5 files found in", folder)
        return

    for file in files:
        try:
            match = re.search(r"fit_Minuit_results_E_(\d+\.\d+)\.h5", file) # Change the type of fit depending on the fit type
            Eline = float(match.group(1)) if match else np.nan

            with h5py.File(file, "r") as f:
                grp = f[f"E_{Eline:.3f}"]
                dm_grp = grp["dm"]

                # Load DM param info
                dm_param_names = dm_grp["param_names"][:].astype(str)
                dm_param_targets = dm_grp["param_targets"][:].astype(str)
                dm_param_values = dm_grp["params"][:]

                # Build name.target → value mapping
                dm_params_dict = {
                    f"{n}.{t}": v
                    for n, t, v in zip(dm_param_names, dm_param_targets, dm_param_values)
                }

                # Extract DM norm value (e.g., 'gaussian.norm') if available
                dm_norm = None
                for key in dm_params_dict:
                    if key.endswith(".norm") and "gaussian" in key:
                        dm_norm = dm_params_dict[key]
                        break

                row = {
                    "Eline (keV)": Eline,
                    "sigma": grp.attrs.get("sigma", np.nan),
                    "Emin": grp.attrs.get("Emin", np.nan),
                    "Emax": grp.attrs.get("Emax", np.nan),
                    "IB lines": grp.attrs.get("Number_of_IB_lines", -1),
                    "TS_as": grp["astro/TS_astro"][()],
                    "nbins_as": grp["astro/nbins"][()],
                    "nfit_as": grp["astro/nfit"][()],
                    "TS_all": dm_grp["TS_all"][()],
                    "ΔTS": dm_grp["deltaTS"][()],
                    "nbins_dm": dm_grp["nbins"][()],
                    "nfit_dm": dm_grp["nfit"][()],
                    "DM norm": dm_norm,
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

# === Run ===
if __name__ == "__main__":
    # Replace this with your actual path if needed
    data_path = "/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/Minuit/"
    output_file = os.path.join(data_path, "fit_summary.csv")
    summarize_fit_results(data_path, output_csv=output_file)
