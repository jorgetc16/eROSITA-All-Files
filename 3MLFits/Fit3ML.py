from threeML import OGIPLike, Model, PointSource, JointLikelihood
from astromodels import Powerlaw, Gaussian
import numpy as np
import h5py
import os
import time
import traceback

# Path constants
DATA_DIR = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/"
SPECTRUM = DATA_DIR + "srctoolout_120_SourceSpec_00001.fits"
RMF = DATA_DIR + "srctoolout_120_RMF_00001.fits"
ARF = DATA_DIR + "srctoolout_120_ARF_00001.fits"

def preload_common_data():
    from astropy.io import fits
    from scipy import interpolate

    dm_data = np.loadtxt("/home/jortecal/GitHub/eRosita/3MLFits/Energy_width.txt", delimiter=",")
    interpolator = interpolate.interp1d(dm_data[:, 0], dm_data[:, 1], kind="linear", fill_value="extrapolate")

    arf = fits.open(ARF)['SPECRESP']
    arfE = (arf.data['ENERG_LO'] + arf.data['ENERG_HI']) / 2.
    aeff = arf.data['SPECRESP']

    return {
        "interpolator": interpolator,
        "arf_E": arfE,
        "arf_eff": aeff
    }

def select_lines(Eline, sigma, sigmaEnrangeLines=7, AstroEl=0.005):
    Eallmin, Eallmax = 0.3, 9.0
    EminL = max(Eline - sigmaEnrangeLines * sigma, Eallmin)
    EmaxL = min(Eline + sigmaEnrangeLines * sigma, Eallmax)

    # Load IB lines
    IB_data = np.loadtxt("/home/jortecal/GitHub/eRosita/3MLFits/MarcoFitsMay/IBLines.txt")
    IB_energy = IB_data[:, 0]
    IB_min = IB_data[:, 1]
    IB_max = IB_data[:, 2]
    IB_mask = (IB_energy > EminL) & (IB_energy < EmaxL)

    # Load Astro lines
    Astro_data = np.loadtxt("/home/jortecal/GitHub/eRosita/3MLFits/MarcoFitsMay/AstroLines.txt")
    A_energy = Astro_data[:, 0]
    A_min = A_energy - AstroEl
    A_max = A_energy + AstroEl
    A_mask = (A_energy > EminL) & (A_energy < EmaxL)

    # Combine
    IB_lines = IB_energy[IB_mask]
    IB_lines_min = IB_min[IB_mask]
    IB_lines_max = IB_max[IB_mask]

    A_lines = A_energy[A_mask]
    A_lines_min = A_min[A_mask]
    A_lines_max = A_max[A_mask]

    lines = np.concatenate((IB_lines, A_lines))
    lines_min = np.concatenate((IB_lines_min, A_lines_min))
    lines_max = np.concatenate((IB_lines_max, A_lines_max))

    idx = np.argsort(lines)
    return lines[idx], lines_min[idx], lines_max[idx]


def build_model(Eline, sigma, normlDMhdata, fit_dm=True, extra_lines=None):
    cubic = Cubic()
    cubic.norm = normlDMhdata
    cubic.norm.bounds = (1e-10, normlDMhdata * 3.0)
    cubic.norm.free = True

    # Ansatz: start with only constant + linear log(E), freeze c, d
    cubic.a = 1.0
    cubic.a.bounds = (-10, 10)
    cubic.a.free = True

    cubic.b = 0.0
    cubic.b.bounds = (-10, 10)
    cubic.b.free = True

    cubic.c = 0.0
    cubic.c.free = True  # freeze quadratic term

    cubic.d = 0.0
    cubic.d.free = True  # freeze cubic term

    total_spec = cubic


    if fit_dm:
        g_dm = Gaussian()
        g_dm.norm = normlDMhdata * 0.01
        g_dm.norm.bounds = (0.0, normlDMhdata * 3.0)
        g_dm.norm.free = True

        g_dm.mean = Eline
        g_dm.mean.free = False

        g_dm.sigma = sigma
        g_dm.sigma.free = False

        total_spec += g_dm

    if extra_lines is not None:
        lines, lines_min, lines_max = extra_lines
        for i, (E, E_min, E_max) in enumerate(zip(lines, lines_min, lines_max)):
            g = Gaussian()
            g.norm = 0.0
            g.norm.bounds = (0.0, normlDMhdata * 3.0)
            g.norm.free = True

            g.mean = E
            g.mean.bounds = (E_min, E_max)
            g.mean.free = False

            g.sigma = 0.0001
            g.sigma.free = False

            g._instance_name = f"gaussian_line_{i+1}"

            total_spec += g

    source = PointSource("source", 0, 0, spectral_shape=total_spec)
    return Model(source)

def run_fit_for_energy(Eline, common_data):
    try:
        start = time.time()
        print(f"[START] E = {Eline:.3f} keV")

        # Load interpolated data
        interpolator = common_data["interpolator"]
        arfE = common_data["arf_E"]
        aeff = common_data["arf_eff"]

        sigma = interpolator(Eline) / 2.355
        Emin = max(Eline - 5 * sigma, 0.3)
        Emax = min(Eline + 5 * sigma, 9.0)

        plugin = OGIPLike(
            "spectrum",
            observation="/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_120_SourceSpec_00001.fits",
            response="/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_120_RMF_00001.fits",
            arf_file="/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_120_ARF_00001.fits"
        )
        Aefftest = np.interp(Eline, arfE, aeff)
        plugin.set_active_measurements(f"{Emin:.4f}-{Emax:.4f}")
        rates = plugin._observed_spectrum.rates
        # Access the response object
        response = plugin.response

        # Retrieve the energy bin edges
        observed_spectrum = plugin._observed_spectrum

        # Retrieve the energy bin edges
        energy_min = np.array(observed_spectrum.starts)
        energy_max = np.array(observed_spectrum.stops)

        # Compute the energy bin centers
        energy_centers = 0.5 * (energy_min + energy_max)

        Ratetest = np.interp(Eline, energy_centers, rates)
        normlDMhdata = Ratetest / Aefftest
        # Line selection
        extra_lines = select_lines(Eline, sigma)

        # Fit Astro model only
        model = build_model(Eline, sigma, normlDMhdata,  fit_dm=False, extra_lines=extra_lines)
        jl = JointLikelihood(model, data_list=[plugin])
        jl.set_minimizer("minuit", tol=0.01)
        jl.fit()
        TS_astro = -2 * jl.compute_statistic()

        # Fit Astro + DM model
        model = build_model(Eline, sigma, normlDMhdata, fit_dm=True, extra_lines=extra_lines)
        jl = JointLikelihood(model, data_list=[plugin])
        jl.set_minimizer("minuit", tol=0.01)
        results = jl.fit()
        TS_all = -2 * jl.compute_statistic()
        deltaTS = TS_all - TS_astro

        # Save results
        grp_name = "E_{:.3f}".format(Eline).replace('.', '_')
        with h5py.File("fit_results_3ml.h5", "a") as f:
            grp = f.create_group(grp_name)
            grp.attrs["sigma"] = sigma
            grp.attrs["Emin"] = Emin
            grp.attrs["Emax"] = Emax
            grp.attrs["fit_version"] = "3ML_v1"
            grp.attrs["deltaTS"] = deltaTS
            grp.attrs["TS_all"] = TS_all
            grp.attrs["TS_astro"] = TS_astro
            for key, val in results.get_best_fit_parameters().items():
                grp.attrs[key] = val.value

        print(f"[DONE ] E = {Eline:.3f} keV | ∆TS = {deltaTS:.2f} | Time = {time.time() - start:.1f} s")

    except Exception as e:
        print(f"[ERROR] E = {Eline:.3f} keV: {e}")
        traceback.print_exc() 