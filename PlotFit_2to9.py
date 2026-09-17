import os
from xspec import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py
import glob

Xset.addModelString("APECROOT","3.1.3")
Xset.allowPrompting = False
Xset.parallel.steppar = 10

# ============================================================================
# CONFIGURATION
# ============================================================================
base_results_dir = "/home/jortecal/GitHub/eRosita/FINALRESULTS/2to9FixedLines"
output_dir = "/home/jortecal/GitHub/eRosita/plots_simple_2to9"
os.makedirs(output_dir, exist_ok=True)

telescopes = ["TM1", "TM2", "TM3", "TM4", "TM5", "TM6", "TM7"]
selected_lines = 4.58  # "all" or specific line(s)

# ============================================================================
# LOAD COMBINED BOUNDS
# ============================================================================
bounds_file = "/home/jortecal/GitHub/eRosita/FINALRESULTS/2to9FixedLines/Results/CombinedProfiles/CombinedBounds.dat"
bounds_data = np.loadtxt(bounds_file)
bounds_energy = bounds_data[:, 0]
bounds_flux = bounds_data[:, 1]

# ============================================================================
# FUNCTION TO READ FIT RESULTS FROM LINE-SPECIFIC H5 FILE
# ============================================================================
def read_fit_results_for_line(telescope_results_folder, line_energy):
    """
    Read fit results from h5 file for a specific DM line.
    H5 structure:
        E_{Eline:.3f}/
            /astro/
                TS_astro
                nfit_astro
                PhoIndex
                p-value-nullhyp
                params_astro (contains both MW and PB params)
            /dm/
                TS_all
                nfit_dm
                params (contains both MW and PB params)
    """
    # Look for h5 files in the telescope results folder
    h5_files = glob.glob(os.path.join(telescope_results_folder, "*.h5"))
    
    line_group_name = f"E_{line_energy:.3f}"
    
    for h5_file in sorted(h5_files):
        try:
            with h5py.File(h5_file, "r") as f:
                if line_group_name in f:
                    grp = f[line_group_name]
                    
                    # Read astro fit results (background only)
                    if "astro" in grp:
                        astro_grp = grp["astro"]
                        params_astro = astro_grp["params_astro"][:] if "params_astro" in astro_grp else None
                        TS_astro = astro_grp["TS_astro"][()] if "TS_astro" in astro_grp else np.nan
                        pvalue_astro = astro_grp["p-value-nullhyp"][()] if "p-value-nullhyp" in astro_grp else np.nan
                        nfit_astro = astro_grp["nfit_astro"][()] if "nfit_astro" in astro_grp else 0
                    else:
                        params_astro = None
                        TS_astro = np.nan
                        pvalue_astro = np.nan
                        nfit_astro = 0
                    
                    # Read DM fit results
                    if "dm" in grp:
                        dm_grp = grp["dm"]
                        params_dm = dm_grp["params"][:] if "params" in dm_grp else None
                        TS_dm = dm_grp["TS_all"][()] if "TS_all" in dm_grp else np.nan
                        nfit_dm = dm_grp["nfit_dm"][()] if "nfit_dm" in dm_grp else 0
                    else:
                        params_dm = None
                        TS_dm = np.nan
                        nfit_dm = 0
                    
                    # Get attributes
                    sigma = grp.attrs.get("sigma", np.nan)
                    Emin = grp.attrs.get("Emin", np.nan)
                    Emax = grp.attrs.get("Emax", np.nan)
                    
                    return params_astro, params_dm, TS_astro, TS_dm, pvalue_astro, sigma, Emin, Emax
        
        except Exception as e:
            continue
    
    return None, None, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def extract_components(plot_grp, n_ib_lines):
    """
    Extract individual folded model components from XSPEC Plot.
    
    Model structure:
      Source 1 (mw):  powerlaw + gaussian
        addComp 1 → Powerlaw
        addComp 2 → Gaussian (DM)
      Source 2 (mpb): powerlaw + gaussian + ... + gaussian
        addComp 3 → PB Powerlaw
        addComp 4..3+n → IB Lines
    
    Returns dict of component arrays.
    """
    comps = {}
    
    # Source 1 components
    try:
        comps['MW Powerlaw'] = np.asarray(Plot.addComp(1, plot_grp))
    except Exception:
        pass
    try:
        comps['DM Gaussian'] = np.asarray(Plot.addComp(2, plot_grp))
    except Exception:
        pass
    
    # Source 2 components
    try:
        comps['PB Powerlaw'] = np.asarray(Plot.addComp(3, plot_grp))
    except Exception:
        pass
    
    # IB lines — sum into one curve
    ib_sum = None
    for i in range(n_ib_lines):
        try:
            ib_comp = np.asarray(Plot.addComp(4 + i, plot_grp))
            if ib_sum is None:
                ib_sum = ib_comp.copy()
            else:
                ib_sum += ib_comp
        except Exception:
            pass
    if ib_sum is not None:
        comps['IB Lines (sum)'] = ib_sum
    
    return comps


# Component styling
COMP_STYLE = {
    'MW Powerlaw':    {'color': 'orange',     'ls': '--',  'lw': 1.5, 'alpha': 0.8},
    'DM Gaussian':    {'color': 'red',        'ls': ':',   'lw': 2.0, 'alpha': 0.9},
    'PB Powerlaw':    {'color': 'purple',     'ls': '--',  'lw': 1.5, 'alpha': 0.8},
    'IB Lines (sum)': {'color': 'gray',       'ls': ':',   'lw': 1.2, 'alpha': 0.7},
}

# ============================================================================
# LOAD DATA FILES
# ============================================================================
path = "/home/jortecal/GitHub/eRosita/LMC5DegEv/srctoolout_000_SourceProducts_00001_5deg_rebin80_DECEMBER/"

telescope_files = {
    "TM1": (path + "srctoolout_120_SourceSpec_00001.fits",
            path + "srctoolout_120_RMF_00001.fits",
            path + "srctoolout_120_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM1.txt",
            "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM1.csv"),
    "TM2": (path + "srctoolout_220_SourceSpec_00001.fits",
            path + "srctoolout_220_RMF_00001.fits",
            path + "srctoolout_220_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM2.txt",
            "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM2.csv"),
    "TM3": (path + "srctoolout_320_SourceSpec_00001.fits",
            path + "srctoolout_320_RMF_00001.fits",
            path + "srctoolout_320_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM3.txt",
            "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM3.csv"),
    "TM4": (path + "srctoolout_420_SourceSpec_00001.fits",
            path + "srctoolout_420_RMF_00001.fits",
            path + "srctoolout_420_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM4.txt",
            "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM4.csv"),
    "TM5": (path + "srctoolout_520_SourceSpec_00001.fits",
            path + "srctoolout_520_RMF_00001.fits",
            path + "srctoolout_520_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM5.txt",
            "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM5.csv"),
    "TM6": (path + "srctoolout_620_SourceSpec_00001.fits",
            path + "srctoolout_620_RMF_00001.fits",
            path + "srctoolout_620_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM6.txt",
            "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM6.csv"),
    "TM7": (path + "srctoolout_720_SourceSpec_00001.fits",
            path + "srctoolout_720_RMF_00001.fits",
            path + "srctoolout_720_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM7.txt",
            "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM7.csv"),
}

# ============================================================================
# PROCESS EACH TELESCOPE
# ============================================================================
for telescope in telescopes:
    print(f"\n{'='*70}")
    print(f"Processing {telescope}...")
    print(f"{'='*70}")
    
    # Get file paths
    spectrum_file, rmf_file, arf_file, ibline_file, energy_width_file = telescope_files[telescope]
    
    # Load energy resolution data
    ew_data = np.loadtxt(energy_width_file, usecols=(0, 1))
    ew_data = ew_data[np.argsort(ew_data[:, 0])]
    vE = ew_data[:, 0]
    vFWHM = ew_data[:, 1] * 1e-3  # eV -> keV
    vsigma = vFWHM / 2.355
    
    # Load spectrum data
    AllData.clear()
    Spectrum(spectrum_file)
    AllData(1).multiresponse[0] = rmf_file
    AllData(1).multiresponse[0].arf = arf_file
    AllData(1).multiresponse[1] = rmf_file
    
    # Load IB lines
    data = np.loadtxt(ibline_file, skiprows=0)
    IBenergy = data[:, 0]
    IBenergymin = data[:, 0] - 0.025
    IBenergymax = data[:, 0] + 0.025
    
    # ================================================================
    # PLOT: DM line with background best fit (no astro fit plot)
    # ================================================================
    # Filter bounds to selected energy lines
    if selected_lines is not None and selected_lines != "all":
        if isinstance(selected_lines, (int, float)):
            selected_lines_list = [selected_lines]
        else:
            selected_lines_list = list(selected_lines)
        mask = np.array([any(abs(e - s) < 1e-3 for s in selected_lines_list) for e in bounds_energy])
        loop_indices = np.where(mask)[0]
    else:
        loop_indices = np.arange(len(bounds_energy))

    for idx_bound in loop_indices:
        Eline_dm = bounds_energy[idx_bound]
        norm_dm = bounds_flux[idx_bound]

        if np.isnan(norm_dm) or norm_dm <= 0:
            print(f"  Skipping E={Eline_dm:.3f} keV (invalid bound)")
            continue

        # Compute DM line sigma from energy resolution
        sigmaDM = np.interp(Eline_dm, vE, vsigma)
        
        # Set dynamic energy range: Eline ± 5*sigma
        Emin = Eline_dm - 5*sigmaDM
        Emax = Eline_dm + 5*sigmaDM
        
        # Ensure range is within reasonable bounds
        Emin = max(Emin, 2.0)
        Emax = min(Emax, 9.0)
        
        print(f"  E_line={Eline_dm:.3f} keV, sigma={sigmaDM:.4f} keV, range=[{Emin:.3f}, {Emax:.3f}] keV")
        
        # Set energy range
        str_range = f"**-{Emin:.4f},,{Emax:.4f}-**"
        AllData.ignore(str_range)
        
        # Read fit results for this specific line from h5 file
        results_folder = os.path.join(base_results_dir, telescope)
        params_astro, params_dm, TS_astro, TS_dm, pvalue_astro, h5_sigma, h5_Emin, h5_Emax = read_fit_results_for_line(results_folder, Eline_dm)
        
        # params_astro and params_dm contain: [MW params (PhoIndex, norm), PB params (PhoIndex, norm, ...)]
        # We need to split them: first 2 are MW, rest are PB
        if params_astro is not None and len(params_astro) >= 2:
            params_mw = params_astro[:2]  # PhoIndex, norm for powerlaw
            params_pb = params_astro[2:]  # PhoIndex, norm, + gaussian params for IB lines
            print(f"  ✓ Astro fit found: TS={TS_astro:.2f}, p-value={pvalue_astro:.4e}")
        else:
            params_mw = None
            params_pb = None
            print(f"  Warning: No astro fit results found for E={Eline_dm:.3f} keV")
        
        # Build model and set parameters
        AllModels.clear()
        m_mw = Model("powerlaw + gaussian", "mw", 1)
        
        if params_mw is not None:
            # Set MW model parameters from h5 file (background fit)
            m_mw.powerlaw.PhoIndex = params_mw[0]
            m_mw.powerlaw.norm = params_mw[1]
        else:
            # Set defaults
            m_mw.powerlaw.PhoIndex = 2.0
            m_mw.powerlaw.norm = 0.1
        
        # Always set DM line to the limit value
        m_mw.gaussian.LineE = Eline_dm
        m_mw.gaussian.Sigma = sigmaDM
        m_mw.gaussian.norm = norm_dm
        
        # Count IB lines in this energy range
        ib_idx = np.where((IBenergy > Emin) & (IBenergy < Emax))[0]
        n_ib = len(ib_idx)
        
        # Build PB model
        model_string_pb = "powerlaw"
        if n_ib > 0:
            model_string_pb += " + " + " + ".join(["gaussian"] * n_ib)
        
        AllModels += (model_string_pb, "mpb", 2)
        mpb = AllModels(1, "mpb")
        
        if params_pb is not None and len(params_pb) >= 2:
            # Set PB powerlaw parameters
            mpb.powerlaw.PhoIndex = params_pb[0]
            mpb.powerlaw.norm = params_pb[1]
            
            # Set IB line parameters
            param_idx = 2
            for i in range(min(n_ib, (len(params_pb) - 2) // 3)):
                if i == 0:
                    gauss_name = "gaussian"
                else:
                    gauss_name = f"gaussian_{i+2}"
                
                if hasattr(mpb, gauss_name):
                    gauss_obj = getattr(mpb, gauss_name)
                    if param_idx + 2 < len(params_pb):
                        gauss_obj.norm = params_pb[param_idx]
                        gauss_obj.LineE = params_pb[param_idx + 1]
                        gauss_obj.Sigma = params_pb[param_idx + 2]
                        param_idx += 3
        else:
            # Set defaults for PB model
            mpb.powerlaw.PhoIndex = 2.0
            mpb.powerlaw.norm = 0.01
        
        # Plot with DM line included
        Xset.chatter = 0
        Plot.commands = ()
        Plot.add = True
        Plot.xAxis = "keV"
        Plot("ldata")

        energies_dm = np.asarray(Plot.x(1))
        edeltas_dm = np.asarray(Plot.xErr(1))
        rates_dm = np.asarray(Plot.y(1))
        errors_dm = np.asarray(Plot.yErr(1))
        foldedmodel_dm = np.asarray(Plot.model(1))

        # Extract individual components (now with DM norm set)
        comps_dm = extract_components(1, n_ib)

        # Calculate background as sum of all components except DM Gaussian
        background = None
        for comp_name, comp_vals in comps_dm.items():
            if comp_name != 'DM Gaussian':
                if background is None:
                    background = comp_vals.copy()
                else:
                    background += comp_vals

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.errorbar(energies_dm, rates_dm, xerr=edeltas_dm, yerr=errors_dm,
                    fmt='.', markersize=6, label='Data', color='black', alpha=0.7, linewidth=2)
        ax.plot(energies_dm, foldedmodel_dm, label='Total Model',
                color='blue', linewidth=2.5)

        # Plot background (summed components)
        if background is not None:
            ax.plot(energies_dm, background, label='Background',
                    color='red', ls='--', lw=1.5, alpha=0.8)

        # Plot DM Gaussian
        if 'DM Gaussian' in comps_dm:
            style = COMP_STYLE['DM Gaussian']
            ax.plot(energies_dm, comps_dm['DM Gaussian'],
                    label=f'DM Gaussian (A={norm_dm:.2e})',
                    color='green', ls=style['ls'], lw=style['lw'], alpha=style['alpha'])

        ax.set_xlabel('Energy (keV)', fontsize=16)
        ax.set_ylabel('counts/s/keV', fontsize=16)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(2e-3, max(rates_dm)*1.2)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', fontsize=14)
        ax.set_title(f'{telescope} —  DM line at E={Eline_dm:.3f} keV (95% CL bound)',
                     fontsize=14, weight='bold')

        plt.tight_layout()

        dm_output_dir = os.path.join(output_dir, telescope)
        os.makedirs(dm_output_dir, exist_ok=True)
        output_file_dm = os.path.join(dm_output_dir, f"SimpleFit_DM_E_{Eline_dm:.3f}_{telescope}.pdf")
        plt.savefig(output_file_dm, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file_dm}")
        plt.close()

print(f"\n{'='*70}")
print(f"All plots saved to: {output_dir}")
print(f"{'='*70}")