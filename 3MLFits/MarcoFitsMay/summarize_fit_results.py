import h5py
import numpy as np
import pandas as pd

def summarize_hdf5_fit_results(h5_file_path, output_csv=None):
    data_rows = []

    with h5py.File(h5_file_path, "r") as f:
        for group_name in sorted(f.keys()):
            grp = f[group_name]
            try:
                Eline = float(group_name.replace("E_", ""))
                TS_astro = grp["astro/TS"][()]
                TS_dm = grp["dm/TS"][()]
                deltaTS = grp["dm/deltaTS"][()]
                sigma = grp.attrs.get("sigma", np.nan)
                Emin = grp.attrs.get("Emin", np.nan)
                Emax = grp.attrs.get("Emax", np.nan)

                data_rows.append({
                    "Eline (keV)": Eline,
                    "TS_astro": TS_astro,
                    "TS_dm": TS_dm,
                    "ΔTS": deltaTS,
                    "sigma": sigma,
                    "Emin": Emin,
                    "Emax": Emax
                })
            except KeyError as e:
                print(f"Skipping {group_name}: missing {e}")

    df = pd.DataFrame(data_rows).sort_values("Eline (keV)")
    print(df.to_string(index=False))

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved to {output_csv}")

    return df

# === Run the summary ===
if __name__ == "__main__":
    h5_path = "fit_results.h5"
    output_path = "fit_summary.csv"
    summarize_hdf5_fit_results(h5_path, output_csv=output_path)