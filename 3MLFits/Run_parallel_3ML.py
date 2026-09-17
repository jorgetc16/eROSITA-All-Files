# Run_parallel_3ML.py
import sys
sys.path.append('/home/jortecal/GitHub/eRosita/3MLFits')

from Fit3ML import run_fit_for_energy, preload_common_data
import numpy as np
from multiprocessing import set_start_method
from joblib import Parallel, delayed

set_start_method("fork", force=True)

if __name__ == "__main__":
    nDM = 10
    Emin = 2.0
    Emax = 10.0
    E_lines = np.logspace(np.log10(Emin), np.log10(Emax), nDM)

    common_data = preload_common_data()
    num_jobs = 1

    Parallel(n_jobs=num_jobs)(
        delayed(run_fit_for_energy)(E, common_data)
        for E in E_lines
    )