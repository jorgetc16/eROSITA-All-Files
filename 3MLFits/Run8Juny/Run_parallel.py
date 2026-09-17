import sys
sys.path.append('/home/jortecal/GitHub/eRosita/3MLFits/Run8Juny/')
from Fit_Parallel import run_fit_for_energy, preload_common_data
from multiprocessing import set_start_method
import numpy as np
from joblib import Parallel, delayed

# Recommended on Linux
set_start_method("fork", force=True)

# if __name__ == "__main__":
#     # Energy list (small test)
#     nDM = 3
#     Emin = 2.0
#     Emax = 10.0
#     E_lines = np.logspace(np.log10(Emin), np.log10(Emax), nDM)

#     common_data = preload_common_data()

#     # Serial test
#     for E in E_lines:
#         run_fit_for_energy(E, common_data)

if __name__ == "__main__":
    nDM = 100  # Or however many energy points you want
    Emin = 2
    Emax = 9.0
    E_lines = np.linspace(Emin, Emax, nDM)

    common_data = preload_common_data()

    # Number of parallel workers (tune to match your CPU cores)
    num_jobs = 10  # e.g., for an 8-core machine

    Parallel(n_jobs=num_jobs)(
        delayed(run_fit_for_energy)(E, common_data)
        for E in E_lines
    )

    #Merge per-energy HDF5 files after all runs
    print("\n🔧 Starting merge of result files...")
    input_files = glob.glob("fit_results_E_*.h5")
    master_file = "fit_results.h5"

    # Delete existing master file if exists
    if os.path.exists(master_file):
        os.remove(master_file)

    with h5py.File(master_file, "w") as master_f:
        for input_file in input_files:
            print(f"Merging {input_file}...")
            with h5py.File(input_file, "r") as src_f:
                for group_name in src_f:
                    if group_name in master_f:
                        print(f"⚠ Warning: group {group_name} already exists, skipping!")
                        continue
                    src_f.copy(group_name, master_f)
            print(f"Done with {input_file}")

    print(f"\nAll files merged into {master_file}")