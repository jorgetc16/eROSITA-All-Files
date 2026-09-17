import os
from xspec import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib as mpl
import h5py
import glob
from matplotlib import rc
import matplotlib.patheffects as path_effects

# print an array between 2 and 2.5 with step 0.01


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}' #package mathpazo siunitx
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') #serif
plt.rcParams['axes.linewidth'] = 2

Xset.addModelString("APECROOT","3.1.3")
Xset.allowPrompting = False
Xset.parallel.steppar = 10

# ============================================================================
# CONFIGURATION
# ============================================================================
base_results_dir = "/home/jortecal/GitHub/eROSITA_RegisPC_Script/OUTPUT/1to2_doublepowerlaw_24"
output_dir = "/home/jortecal/GitHub/eRosita/plots_astro_fit_doublepowerlaw_24"
os.makedirs(output_dir, exist_ok=True)

telescopes = ["TM1", "TM2", "TM3", "TM4", "TM5", "TM6", "TM7"]
# Select which DM energy lines to plot. Options:
#   None or "all"  → all energies in the bounds file
#   single float   → e.g. 1.352
#   list of floats → e.g. [1.352, 1.497, 1.800]
selected_lines = 1.206  # e.g. 1.352  or  [1.352, 1.497]
# ============================================================================
# LOAD COMBINED BOUNDS (Energy [keV], Flux limit [ph/cm2/s])
# ============================================================================
bounds_file = "/home/jortecal/GitHub/eRosita/Full_Range_24_dblpwl/Results/CombinedProfiles/CombinedBounds.dat"
bounds_data = np.loadtxt(bounds_file)
bounds_energy = bounds_data[:, 0]
bounds_flux = bounds_data[:, 1]

# ============================================================================
# FUNCTION TO READ ASTRO FIT RESULTS
# ============================================================================
def read_fit_astro(folder):
    """Read astro fit results from h5 file."""
    astro_file = os.path.join(folder, "fit_results_Astro.h5")
    if not os.path.exists(astro_file):
        print(f"Warning: {astro_file} not found")
        return None
    
    with h5py.File(astro_file, "r") as f:
        astro_grp = f["Fit_Astro"]["astro"]
        params_astro_mw = astro_grp["params_astro_mw"][:]
        params_astro_pb = astro_grp["params_astro_pb"][:]
        TS_astro = astro_grp["TS_astro"][()]
        pvalue_astro = astro_grp["p-value-nullhyp"][()]
        dof_astro = astro_grp["dof_astro"][()]
        nfit_astro = astro_grp["nfit_astro"][()]
    
    return params_astro_mw, params_astro_pb, TS_astro, pvalue_astro, dof_astro, nfit_astro


def extract_components(plot_grp, n_ib_lines):
    """
    Extract individual folded model components from XSPEC Plot.
    
    Model structure:
      Source 1 (mw):  tbabs*(apec + powerlaw) + tbabs*(gaussian)
        addComp 1 → TBabs × APEC
        addComp 2 → TBabs × Powerlaw
        addComp 3 → TBabs × Gaussian (DM)
      Source 2 (mpb): powerlaw + gaussian + ... + gaussian
        addComp 4 → PB Powerlaw
        addComp 5..4+n → IB Lines
    
    Returns dict of component arrays.
    """
    comps = {}
    
    # Source 1 components
    try:
        comps['APEC'] = np.asarray(Plot.addComp(1, plot_grp))
    except Exception:
        pass
    try:
        comps['MW Powerlaw'] = np.asarray(Plot.addComp(2, plot_grp))
    except Exception:
        pass
    try:
        comps['DM Gaussian'] = np.asarray(Plot.addComp(3, plot_grp))
    except Exception:
        pass
    
    # Source 2 components
    try:
        comps['PB Powerlaw'] = np.asarray(Plot.addComp(4, plot_grp))
    except Exception:
        pass
    
    # IB lines — sum into one curve
    ib_sum = None
    for i in range(n_ib_lines):
        try:
            ib_comp = np.asarray(Plot.addComp(5 + i, plot_grp))
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
    'APEC':           {'color': 'green',      'ls': '--',  'lw': 1.5, 'alpha': 0.8},
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
    
    # Read fit results
    results_folder = os.path.join(base_results_dir, telescope)
    fit_results = read_fit_astro(results_folder)
    
    if fit_results is None:
        print(f"Skipping {telescope} - no fit results found")
        continue
    
    params_astro_mw, params_astro_pb, TS_astro, pvalue_astro, dof_astro, nfit_astro = fit_results
    
    print(f"TS: {TS_astro:.2f}, p-value: {pvalue_astro:.4f}, DOF: {dof_astro}, nfit: {nfit_astro}")
    
    # Load spectrum data
    AllData.clear()
    Spectrum(spectrum_file)
    AllData(1).multiresponse[0] = rmf_file
    AllData(1).multiresponse[0].arf = arf_file
    AllData(1).multiresponse[1] = rmf_file
    
    # Set energy range
    Emin = 1.0
    Emax = 2.4
    str_range = f"**-{Emin:.4f},,{Emax:.4f}-**"
    AllData.ignore(str_range)
    
    # Load IB lines
    data = np.loadtxt(ibline_file, skiprows=0)
    IBenergy = data[:, 0]
    IBenergymin = data[:, 0] - 0.025
    IBenergymax = data[:, 0] + 0.025
    
    IBidx = np.where((IBenergy > Emin) & (IBenergy < Emax))
    IBALine = IBenergy[IBidx]
    IBALinemin = IBenergymin[IBidx]
    IBALinemax = IBenergymax[IBidx]
    n_ib = len(IBALine)
    
    # Build models
    AllModels.clear()
    m_mw = Model("tbabs*(apec + powerlaw) + tbabs*(gaussian)", "mw", 1)
    
    # Set MW model parameters from h5 file
    m_mw.TBabs.nH = params_astro_mw[0]
    m_mw.TBabs_4.nH = params_astro_mw[1]
    m_mw.apec.kT = params_astro_mw[2]
    m_mw.apec.Abundanc = params_astro_mw[3]
    m_mw.apec.Redshift = params_astro_mw[4]
    m_mw.apec.norm = params_astro_mw[5]
    m_mw.powerlaw.PhoIndex = params_astro_mw[6]
    m_mw.powerlaw.norm = params_astro_mw[7]
    m_mw.gaussian.LineE = params_astro_mw[8]
    m_mw.gaussian.Sigma = params_astro_mw[9]
    m_mw.gaussian.norm = params_astro_mw[10]
    
    # Build PB model
    model_string_pb = "powerlaw"
    if n_ib > 0:
        model_string_pb += " + " + " + ".join(["gaussian"] * n_ib)
    
    AllModels += (model_string_pb, "mpb", 2)
    mpb = AllModels(1, "mpb")
    
    # Set PB powerlaw parameters
    mpb.powerlaw.PhoIndex = params_astro_pb[0]
    mpb.powerlaw.norm = params_astro_pb[1]
    
    # Set IB line parameters
    param_idx = 2  # Start after powerlaw params
    for i, line_e in enumerate(IBALine):
        if i == 0:
            gauss_name = "gaussian"
        else:
            gauss_name = f"gaussian_{i+2}"
        
        if hasattr(mpb, gauss_name):
            gauss_obj = getattr(mpb, gauss_name)
            gauss_obj.norm = params_astro_pb[param_idx]
            gauss_obj.LineE = params_astro_pb[param_idx + 1]
            gauss_obj.Sigma = params_astro_pb[param_idx + 2]
            param_idx += 3
    
    # ================================================================
    # PLOT 1: Astro-only model with components
    # ================================================================
    Xset.chatter = 0
    Plot.commands = ()
    Plot.add = True
    Plot.xAxis = "keV"
    Plot("ldata")
    
    # Extract data and total model
    energies = np.asarray(Plot.x(1))
    edeltas = np.asarray(Plot.xErr(1))
    rates = np.asarray(Plot.y(1))
    errors = np.asarray(Plot.yErr(1))
    foldedmodel = np.asarray(Plot.model(1))
    
    # Extract individual components
    comps = extract_components(1, n_ib)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.errorbar(energies, rates, xerr=edeltas, yerr=errors,
                fmt='.', markersize=6, label='Data', color='black', alpha=0.7, linewidth=2)
    ax.plot(energies, foldedmodel, label='Total Model', color='blue', linewidth=2.5)
    
    # Plot each component
    for comp_name, comp_vals in comps.items():
        if comp_name in ('DM Gaussian'):
            continue
        style = COMP_STYLE.get(comp_name, {'color': 'gray', 'ls': '-.', 'lw': 1.0, 'alpha': 0.6})
        ax.plot(energies, comp_vals, label=comp_name,
                color=style['color'], ls=style['ls'], lw=style['lw'], alpha=style['alpha'])
    
    ax.set_xlabel('Energy (keV)', fontsize=14)
    ax.set_ylabel('counts/s/keV', fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")        
    ax.set_ylim(2e-3, max(rates)*1.2) # Adjust as needed for visibility
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=11)
    ax.set_title(f'Astro Fit - {telescope} (TS={TS_astro:.1f}, p={pvalue_astro:.3e})', 
                 fontsize=14, weight='bold')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"AstroFit_{telescope}.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # ================================================================
    # PLOT 2: Astro model + DM line with components
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
        ew_data = np.loadtxt(energy_width_file, usecols=(0, 1))
        ew_data = ew_data[np.argsort(ew_data[:, 0])]
        vE = ew_data[:, 0]
        vFWHM = ew_data[:, 1] * 1e-3  # eV -> keV
        vsigma = vFWHM / 2.355
        sigmaDM = np.interp(Eline_dm, vE, vsigma)

        # Set the DM gaussian in the MW model to the limit value
        m_mw.gaussian.LineE = Eline_dm
        m_mw.gaussian.Sigma = sigmaDM
        m_mw.gaussian.norm = norm_dm

        # Re-plot with DM line included
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

        fig, ax = plt.subplots(figsize=(8, 7))

        ax.errorbar(energies_dm, rates_dm, xerr=edeltas_dm, yerr=errors_dm,
                    fmt='.', markersize=6, label='Data', color='black', alpha=0.7, linewidth=2)
        ax.plot(energies_dm, foldedmodel_dm, label=f'Total Model',
                color='blue', linewidth=2.5)

        # Plot background (summed components)
        if background is not None:
            ax.plot(energies_dm, background, label='Background',
                    color='red', ls='--', lw=1.5, alpha=0.8)

        # Plot DM Gaussian
        if 'DM Gaussian' in comps_dm:
            style = COMP_STYLE['DM Gaussian']
            ax.plot(energies_dm, comps_dm['DM Gaussian'],
                    # label=f'DM Gaussian (A={norm_dm:.2e})',
                    label=f'DM Line',
                    color='green', ls=style['ls'], lw=style['lw'], alpha=style['alpha'])

        ax.set_xlabel('Energy [keV]', fontsize=24)
        ax.set_ylabel('counts/s/keV', fontsize=24)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
        ax.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
        ax.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        ax.set_ylim(2e-3, max(rates_dm)*1.2) # Adjust as needed for visibility
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', fontsize=20)
        ax.set_title(f'{telescope} — DM line at E={Eline_dm:.3f} keV',
                     fontsize=20, weight='bold')

        ax.set_xticks([1, 1.5, 2, 2.4])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
        plt.tight_layout()

        dm_output_dir = os.path.join(output_dir, "AstroFit_DM", telescope)
        os.makedirs(dm_output_dir, exist_ok=True)
        output_file_dm = os.path.join(dm_output_dir, f"AstroFit_DM_E_{Eline_dm:.3f}_{telescope}.pdf")
        plt.savefig(output_file_dm, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file_dm}")
        plt.close()

    # Reset DM gaussian to zero for next telescope
    m_mw.gaussian.norm = 0.0

print(f"\n{'='*70}")
print(f"All plots saved to: {output_dir}")
print(f"{'='*70}")

