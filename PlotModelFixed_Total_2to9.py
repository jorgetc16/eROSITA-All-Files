import os
from xspec import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
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

Xset.allowPrompting = False
Xset.parallel.steppar = 10

# ============================================================================
# CONFIGURATION - EASILY MODIFIABLE PARAMETERS
# ============================================================================
Eline = 4.325
telescope = 'TM3'

# DM Gaussian parameters (Source 1: mw)
dm_norm = 2.218508e-03
dm_line_e = 4.32500
dm_sigma = 7.20833E-04

# PB Powerlaw parameters (Source 2: mpb)
pb_pho_index = 0.2109822072058005
pb_norm = 0.081763664873960

# PB Gaussian parameters (Source 2: mpb - goes into background)
pb_gauss_line_e = 4.49506
pb_gauss_sigma = 1.00000E-04
pb_gauss_norm = 1.4375012113987395e-07

# ============================================================================
# LOAD DATA
# ============================================================================

AllData.clear()
path="/home/jortecal/GitHub/eRosita/LMC5DegEv/srctoolout_000_SourceProducts_00001_5deg_rebin80_DECEMBER/"
filespectrumTM1=path+"srctoolout_120_SourceSpec_00001.fits"
fileRMFTM1=path+"srctoolout_120_RMF_00001.fits"
fileARFTM1=path+"srctoolout_120_ARF_00001.fits"
Energy_width_TM1 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM1.csv"
#
filespectrumTM2=path+"srctoolout_220_SourceSpec_00001.fits"
fileRMFTM2=path+"srctoolout_220_RMF_00001.fits"
fileARFTM2=path+"srctoolout_220_ARF_00001.fits"
Energy_width_TM2 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM2.csv"
#
filespectrumTM3=path+"srctoolout_320_SourceSpec_00001.fits"
fileRMFTM3=path+"srctoolout_320_RMF_00001.fits"
fileARFTM3=path+"srctoolout_320_ARF_00001.fits"
Energy_width_TM3 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM3.csv"
#
filespectrumTM4=path+"srctoolout_420_SourceSpec_00001.fits"
fileRMFTM4=path+"srctoolout_420_RMF_00001.fits"
fileARFTM4=path+"srctoolout_420_ARF_00001.fits"
Energy_width_TM4 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM4.csv"
#
filespectrumTM5=path+"srctoolout_520_SourceSpec_00001.fits"
fileRMFTM5=path+"srctoolout_520_RMF_00001.fits"
fileARFTM5=path+"srctoolout_520_ARF_00001.fits"
Energy_width_TM5 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM5.csv"
#
filespectrumTM6=path+"srctoolout_620_SourceSpec_00001.fits"
fileRMFTM6=path+"srctoolout_620_RMF_00001.fits"
fileARFTM6=path+"srctoolout_620_ARF_00001.fits"
Energy_width_TM6 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM6.csv"
#
filespectrumTM7=path+"srctoolout_720_SourceSpec_00001.fits"
fileRMFTM7=path+"srctoolout_720_RMF_00001.fits"
fileARFTM7=path+"srctoolout_720_ARF_00001.fits"
Energy_width_TM7 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM7.csv"

# Define telescope file mappings
telescope_files = {
    "TM1": (filespectrumTM1, fileRMFTM1, fileARFTM1, Energy_width_TM1),
    "TM2": (filespectrumTM2, fileRMFTM2, fileARFTM2, Energy_width_TM2),
    "TM3": (filespectrumTM3, fileRMFTM3, fileARFTM3, Energy_width_TM3),
    "TM4": (filespectrumTM4, fileRMFTM4, fileARFTM4, Energy_width_TM4),
    "TM5": (filespectrumTM5, fileRMFTM5, fileARFTM5, Energy_width_TM5),
    "TM6": (filespectrumTM6, fileRMFTM6, fileARFTM6, Energy_width_TM6),
    "TM7": (filespectrumTM7, fileRMFTM7, fileARFTM7, Energy_width_TM7),
}

# Set the file arrays based on telescope selection
if telescope in telescope_files:
    spectrum_file, rmf_file, arf_file, energy_width_file = telescope_files[telescope]
    vfilespectrum = [spectrum_file]
    vfileRMF = [rmf_file]
    vfileARF = [arf_file]
    vEnergyWidth = [energy_width_file]
    print(f"Using telescope: {telescope}")
    print(f"Number of telescopes: {len(vfilespectrum)}")

# Load spectra
AllData.clear()
for plot_grp in range(0, len(vfilespectrum)):
    Spectrum(vfilespectrum[plot_grp])
    AllData(plot_grp+1).multiresponse[0] = vfileRMF[plot_grp]
    AllData(plot_grp+1).multiresponse[0].arf = vfileARF[plot_grp]
    AllData(plot_grp+1).multiresponse[1] = vfileRMF[plot_grp]  # RMF only for src2

AllData.show()

# ============================================================================
# SET ENERGY RANGE (±5σ around Eline, as in Fitpoly_Novembre_Final_MT_2to9.py)
# ============================================================================
data = np.loadtxt(vEnergyWidth[0], usecols=(0, 1))
sorted_data = data[np.argsort(data[:, 0])]
data = sorted_data
vE = data[:, 0]
vFWHM = data[:, 1] * 1e-3  # keV
vsigma = vFWHM / 2.355

sigma = np.interp(Eline, vE, vsigma)
print(f"Eline {Eline}  sigma {sigma}")

Eallmin = 0.3
Eallmax = 9.0
Emin = Eline - 5 * sigma
Emax = Eline + 5 * sigma

if Emin < Eallmin:
    Emin = Eallmin
if Emax > Eallmax:
    Emax = Eallmax

print(f"Emin {Emin}  Emax {Emax}")
str_range = "**-" + str(round(Emin, 4)) + ",," + str(round(Emax, 4)) + "-**"
AllData.ignore(str_range)

# ============================================================================
# BUILD MODELS IN XSPEC (NO FITTING)
# Model: Source 1 (mw) = gaussian (DM line)
#        Source 2 (mpb) = powerlaw + gaussian (PB background + PB line)
# ============================================================================

AllModels.clear()

# Source 1: MW = gaussian (DM line only)
m_mw = Model("gaussian", "mw", 1)

# Set MW Gaussian parameters (DM line)
m_mw.gaussian.LineE = dm_line_e
m_mw.gaussian.Sigma = dm_sigma
m_mw.gaussian.norm = dm_norm
m_mw.gaussian.LineE.frozen = True
m_mw.gaussian.Sigma.frozen = True
m_mw.gaussian.norm.frozen = True

# Source 2: PB = powerlaw + gaussian (background)
AllModels += ("powerlaw + gaussian", "mpb", 2)
mpb = AllModels(1, "mpb")

# Set PB powerlaw parameters
mpb.powerlaw.PhoIndex = pb_pho_index
mpb.powerlaw.norm = pb_norm
mpb.powerlaw.PhoIndex.frozen = True
mpb.powerlaw.norm.frozen = True

# Set PB gaussian parameters
mpb.gaussian.LineE = pb_gauss_line_e
mpb.gaussian.Sigma = pb_gauss_sigma
mpb.gaussian.norm = pb_gauss_norm
mpb.gaussian.LineE.frozen = True
mpb.gaussian.Sigma.frozen = True
mpb.gaussian.norm.frozen = True

# ============================================================================
# PLOT MODEL WITH DATA - MANUAL COMPONENT CALCULATION
# ============================================================================
Xset.chatter = 0

# Plot model with total
Plot.commands = ()
Plot.add = False
Plot.xAxis = "keV"
Plot("ldata")

# Extract data
plot_grp = 1
energies = np.asarray(Plot.x(plot_grp))
edeltas = np.asarray(Plot.xErr(plot_grp))
rates = np.asarray(Plot.y(plot_grp))
errors = np.asarray(Plot.yErr(plot_grp))
total_model = np.asarray(Plot.model(plot_grp))

# Manually calculate components by temporarily setting parameters
# Save original DM norm
original_dm_norm = m_mw.gaussian.norm.values[0]

# Get background: set DM norm to 0 (leaves PB powerlaw + PB gaussian)
m_mw.gaussian.norm = 0.0
Plot.commands = ()
Plot.add = False
Plot.xAxis = "keV"
Plot("ldata")
background = np.asarray(Plot.model(plot_grp))

# Get DM component: total - background
m_mw.gaussian.norm = original_dm_norm
Plot.commands = ()
Plot.add = False
Plot.xAxis = "keV"
Plot("ldata")
total_with_dm = np.asarray(Plot.model(plot_grp))
dm_component = total_with_dm - background

print(f"✓ Components extracted manually")
print(f"  Total model max: {np.max(total_model):.4e}")
print(f"  Background max: {np.max(background):.4e}")
print(f"  DM component max: {np.max(dm_component):.4e}")

# ============================================================================
# CREATE PLOT
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 7))

# Plot data
ax.errorbar(energies, rates, xerr=edeltas, yerr=errors,
            fmt='.', markersize=6, label='Data', color='black', alpha=0.7, linewidth=2)

# Plot total model
ax.plot(energies, total_model, label='Total Model', color='blue', linewidth=2.5)

# Plot background (PB powerlaw + PB gaussian)
ax.plot(energies, background, label='Background',
        color='red', ls='--', lw=1.5, alpha=0.8)

# Plot DM Gaussian
ax.plot(energies, dm_component,
        # label=f'DM Gaussian (A={dm_norm:.2e})',
        label=f'DM Line',
        color='green', ls=':', lw=2.0, alpha=0.9)
ax.set_xlabel('Energy [keV]', fontsize=24)
ax.set_ylabel('counts/s/keV', fontsize=24)
ax.set_xscale("log")
ax.set_yscale("log")
ax.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax.set_ylim(2e-3, max(rates)*1.2) # Adjust as needed for visibility
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='best', fontsize=20)
ax.set_title(f'{telescope} — DM line at E={Eline:.3f} keV',
                fontsize=20, weight='bold')

# Add a major tick at 2 keV
ax.set_xticks([4.1, 4.3, 4.5])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
plt.tight_layout()
# plt.show()
ouput_file_dm = f"PlotModelFixed_Total_{telescope}_Eline_{Eline:.3f}.pdf"
plt.savefig(ouput_file_dm, dpi=300, bbox_inches='tight')

Xset.closeLog()