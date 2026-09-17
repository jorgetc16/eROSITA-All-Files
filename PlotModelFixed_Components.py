import os
from xspec import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

Xset.allowPrompting = False
Xset.parallel.steppar = 10

# ============================================================================
Eline = 1.206
telescope = 'TM6'
# ============================================================================

# ============================================================================
# LOAD DATA
# ============================================================================

AllData.clear()
path="/home/jortecal/GitHub/eRosita/LMC5DegEv/srctoolout_000_SourceProducts_00001_5deg_rebin80_DECEMBER/"
#filespectrum=path+"srctoolout_120_SourceSpec_00001.fits"
#fileRMF=path+"srctoolout_120_RMF_00001.fits"
#fileARF=path+"srctoolout_120_ARF_00001.fits"
filespectrumTM1=path+"srctoolout_120_SourceSpec_00001.fits"
fileRMFTM1=path+"srctoolout_120_RMF_00001.fits"
fileARFTM1=path+"srctoolout_120_ARF_00001.fits"
fileIBLineTM1="/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM1.txt"
Energy_width_TM1 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM1.csv"
#
filespectrumTM2=path+"srctoolout_220_SourceSpec_00001.fits"
fileRMFTM2=path+"srctoolout_220_RMF_00001.fits"
fileARFTM2=path+"srctoolout_220_ARF_00001.fits"
fileIBLineTM2="/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM2.txt"
Energy_width_TM2 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM2.csv"
#
filespectrumTM3=path+"srctoolout_320_SourceSpec_00001.fits"
fileRMFTM3=path+"srctoolout_320_RMF_00001.fits"
fileARFTM3=path+"srctoolout_320_ARF_00001.fits"
fileIBLineTM3="/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM3.txt"
Energy_width_TM3 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM3.csv"
#
filespectrumTM4=path+"srctoolout_420_SourceSpec_00001.fits"
fileRMFTM4=path+"srctoolout_420_RMF_00001.fits"
fileARFTM4=path+"srctoolout_420_ARF_00001.fits"
fileIBLineTM4="/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM4.txt"
Energy_width_TM4 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM4.csv"
#
filespectrumTM5=path+"srctoolout_520_SourceSpec_00001.fits"
fileRMFTM5=path+"srctoolout_520_RMF_00001.fits"
fileARFTM5=path+"srctoolout_520_ARF_00001.fits"
fileIBLineTM5="/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM5.txt"
Energy_width_TM5 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM5.csv"
#
filespectrumTM6=path+"srctoolout_620_SourceSpec_00001.fits"
fileRMFTM6=path+"srctoolout_620_RMF_00001.fits"
fileARFTM6=path+"srctoolout_620_ARF_00001.fits"
fileIBLineTM6="/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM6.txt"
Energy_width_TM6 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM6.csv"
#
filespectrumTM7=path+"srctoolout_720_SourceSpec_00001.fits"
fileRMFTM7=path+"srctoolout_720_RMF_00001.fits"
fileARFTM7=path+"srctoolout_720_ARF_00001.fits"
fileIBLineTM7="/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM7.txt"
Energy_width_TM7 = "/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM7.csv"
#Put here the TM you want to use
#

# Define telescope file mappings
telescope_files = {
    "TM1": (filespectrumTM1, fileRMFTM1, fileARFTM1, fileIBLineTM1, Energy_width_TM1),
    "TM2": (filespectrumTM2, fileRMFTM2, fileARFTM2, fileIBLineTM2, Energy_width_TM2),
    "TM3": (filespectrumTM3, fileRMFTM3, fileARFTM3, fileIBLineTM3, Energy_width_TM3),
    "TM4": (filespectrumTM4, fileRMFTM4, fileARFTM4, fileIBLineTM4, Energy_width_TM4),
    "TM5": (filespectrumTM5, fileRMFTM5, fileARFTM5, fileIBLineTM5, Energy_width_TM5),
    "TM6": (filespectrumTM6, fileRMFTM6, fileARFTM6, fileIBLineTM6, Energy_width_TM6),
    "TM7": (filespectrumTM7, fileRMFTM7, fileARFTM7, fileIBLineTM7, Energy_width_TM7),
    "ALL": ([filespectrumTM1, filespectrumTM2, filespectrumTM3, filespectrumTM4, filespectrumTM5, filespectrumTM6, filespectrumTM7],
            [fileRMFTM1, fileRMFTM2, fileRMFTM3, fileRMFTM4, fileRMFTM5, fileRMFTM6, fileRMFTM7],
            [fileARFTM1, fileARFTM2, fileARFTM3, fileARFTM4, fileARFTM5, fileARFTM6, fileARFTM7],
            [fileIBLineTM1, fileIBLineTM2, fileIBLineTM3, fileIBLineTM4, fileIBLineTM5, fileIBLineTM6, fileIBLineTM7],
            [Energy_width_TM1, Energy_width_TM2, Energy_width_TM3, Energy_width_TM4, Energy_width_TM5, Energy_width_TM6, Energy_width_TM7]),
    "ALL_CLEAN": ([filespectrumTM1, filespectrumTM2, filespectrumTM3, filespectrumTM4, filespectrumTM6],
    [fileRMFTM1, fileRMFTM2, fileRMFTM3, fileRMFTM4, fileRMFTM6],
    [fileARFTM1, fileARFTM2, fileARFTM3, fileARFTM4, fileARFTM6],
    [fileIBLineTM1, fileIBLineTM2, fileIBLineTM3, fileIBLineTM4, fileIBLineTM6],
    [Energy_width_TM1, Energy_width_TM2, Energy_width_TM3, Energy_width_TM4, Energy_width_TM6])
}

# Set the file arrays based on telescope selection
if telescope in telescope_files:
    if telescope == "ALL":
        vfilespectrum, vfileRMF, vfileARF, vfileIBLine, vEnergyWidth = telescope_files[telescope]
    else:
        spectrum_file, rmf_file, arf_file, ibline_file, energy_width_file = telescope_files[telescope]
        vfilespectrum = [spectrum_file]
        vfileRMF = [rmf_file]
        vfileARF = [arf_file]
        vfileIBLine = [ibline_file]
        vEnergyWidth = [energy_width_file]
    print(f"Using telescope: {telescope}")
    print(f"Number of telescopes: {len(vfilespectrum)}")
# Load spectra
AllData.clear()
for plot_grp in range (0,len(vfilespectrum)):
    #print("plot_grp ",plot_grp)
    Spectrum(vfilespectrum[plot_grp])
    AllData(plot_grp+1).multiresponse[0] = vfileRMF[plot_grp]
    AllData(plot_grp+1).multiresponse[0].arf = vfileARF[plot_grp]
    AllData(plot_grp+1).multiresponse[1] = vfileRMF[plot_grp] # RMF only for src2

AllData.show()

# %%
data = np.loadtxt(vEnergyWidth[0], usecols=(0, 1))
sorted_data = data[np.argsort(data[:, 0])]
data = sorted_data
vE = data[:,0]
vFWHM = data[:,1]*1e-3 #keV
vsigma = vFWHM/(2.355)
# ============================================================================
# SET ENERGY RANGE
# ============================================================================

sigma = np.interp(Eline, vE, vsigma)

Eallmin = 0.3
Eallmax = 9.0
Emin = Eline - 5 * sigma
Emax = Eline + 5 * sigma

if Emin < Eallmin:
    Emin = Eallmin
if Emax > Eallmax:
    Emax = Eallmax

str_range = "**-" + str(round(Emin, 4)) + ",," + str(round(Emax, 4)) + "-**"
AllData.ignore(str_range)

# ============================================================================
# LOAD IB LINES
# ============================================================================

EminL = Eline - 7 * sigma
EmaxL = Eline + 7 * sigma
if EminL < Eallmin:
    EminL = Eallmin
if EmaxL > Eallmax:
    EmaxL = Eallmax

data = np.loadtxt(ibline_file, skiprows=0)
IBenergy = data[:, 0]
IBenergymin = data[:, 0] - 0.025
IBenergymax = data[:, 0] + 0.025

IBidx = np.where((IBenergy > EminL) & (IBenergy < EmaxL))
IBLine = IBenergy[IBidx]
IBLinemin = IBenergymin[IBidx]
IBLinemax = IBenergymax[IBidx]

IBALine = IBLine.copy()
IBALinemin = IBLinemin.copy()
IBALinemax = IBLinemax.copy()
idx = np.argsort(IBALine)
IBALine = IBALine[idx]
IBALinemin = IBALinemin[idx]
IBALinemax = IBALinemax[idx]

print(f"Total IB lines: {len(IBALine)}")
print(f"Line energies: {IBALine}")

# ============================================================================
# DEFINE MODEL WITH INITIAL VALUES
# ============================================================================

def pdic(value, lower, upper, key):
    return {"value": value, "lower": lower, "upper": upper, "keyfit": key}


smallnormlower = 1e-13 # if you want to avoid setting lower limit to zero
smallnorm = 1e-10 # if you want to avoid setting norm to zero

normlDMhdata2=0.1
normlDMhdata=0.1
normline_start= normlDMhdata2*0.01 #*0.01
normline_lower = smallnormlower
normline_upper = normlDMhdata2*1000
sigmaDM = Eline*50/3e5 #1e-4
sigma_astrolines = 1e-4
#
#
#power law
startidx = 1.0 #1.0
upperidx = 4.5 #4.5
loweridx = - 3.0 #-3.0
#
# This normalization should be resonable for PL folded through the ARF
startpl = 0.1*normlDMhdata *np.pow(Eline,startidx)
# startpl = 0.05
#upperpl = startpl*1000.0
upperpl = normlDMhdata *np.pow(Eline,upperidx)*100.
lowerpl = smallnormlower
print("startpl ",startpl," upperpl ",upperpl)
#TABS
#from the https://github.com/xyzhang/eRASS-spectrum/blob/main/spectral_analysis_basic.ipynb)
#The foreground Milkyway nH value is cm (https://www.swift.ac.uk/analysis/nhtot/index.php)
# returns 0.3. IN eROSITA LMC paper they have 0.3-0.4 in sources and 8e-2 between SNR
# See also https://ui.adsabs.harvard.edu/abs/2011ApJ...728..159W/abstract (NO arXiv plot is wrong, open published version)
#, LMC is at −32.89, 280.46. I expect NH around few*1e-2
startnH = 6.3e-2 # 6.3e-2 1e22 cm-2
# do not allow too crazy values
uppernH = 0.8 #0.8
lowernH = 1.0e-2 #1e-2
#
# APEC https://cxc.cfa.harvard.edu/sherpa/ahelp/xsapec.html
#from the https://github.com/xyzhang/eRASS-spectrum/blob/main/spectral_analysis_basic.ipynb)
startapec = 0.14 # 0.14 from global fit
upperapec = 2.0 #2.0 80
lowerapec = smallnormlower
startT = 0.23 #keV 0.23
upperT = 3.0 # keV 3.0
lowerT = 0.05 #0.05
startAb = 1.0 # from the https://github.com/xyzhang/eRASS-spectrum/blob/main/spectral_analysis_basic.ipynb)
upperAb = 1.0 #1.0
lowerAb = 1.0 #1.0
#
# TBABS for DM line: do not leave free
# value from https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3nh/w3nh.pl?Entry=LMC&NR=GRB%2FSIMBAD%2BSesame%2FNED&CoordSys=Equatorial&equinox=2000&radius=5.0&usemap=0
nHDM = 0.22
#
smallnormlower = 1e-13
normline_start = 1e-10
normline_lower = smallnormlower
normline_upper = 1e-5
sigmaDM = Eline * 50 / 3e5
sigma_astrolines = 1e-4

# MW model - INITIAL VALUES
model_dic = {
    "TBabs": {
        "TBabs": {
            "nH": pdic(startnH, lowernH, uppernH, True), # (value, lower, upper, key)
            },
        "TBabs_4": {
            "nH": pdic(nHDM, nHDM, nHDM, False), # (value, lower, upper, key)
            }
        },
    "apec": {
        "apec": {
            "kT": pdic(startT, lowerT, upperT, True), # (value, lower, upper, key)
            "Abundanc" : pdic(startAb, lowerAb, upperAb, False),
            "Redshift" : pdic(0., 0., 0.0, False), #0., 0., 0.
            "norm" : pdic(startapec, lowerapec, upperapec, True)
            }
        },
    "powerlaw": {
        "powerlaw": {
            "PhoIndex": pdic(0.0, loweridx, upperidx, False), # (value, lower, upper, key)
            "norm" : pdic(smallnormlower, lowerpl, upperpl, False)
            }
        },
    "gaussian": {
        "gaussian": {
            "LineE" : pdic(Eline, Eline, Eline, False),
            "Sigma" : pdic(sigmaDM, sigmaDM, sigmaDM, False), #sigmaDM, sigmaDM, sigmaDM
            "norm": pdic(normline_start, normline_lower, normline_upper, False), # (value, lower, upper, key)
            }
        }
    }

model_dic["_structure"] = f"tbabs*(apec + powerlaw) + tbabs*(gaussian)"

# PB model - INITIAL VALUES
model_dic_pb = {
    "powerlaw": {
        "powerlaw": {
            "PhoIndex": pdic(-0.371323004057074, -3.0, 4.5, False),
            "norm": pdic(0.0895498456161, 1e-13, 1.0, False),
        }
    },
    "gaussian": {},
}

for i in range(len(IBALine)):
    if i == 0:
        g_key = "gaussian"
    else:
        g_key = f"gaussian_{i+2}"
    model_dic_pb["gaussian"][g_key] = {
        "norm": pdic(0.003446429422432, normline_lower, normline_upper, False),
        "LineE": pdic(IBALine[i], IBALine[i] - 0.025, IBALine[i] + 0.025, False),
        "Sigma": pdic(sigma_astrolines, sigma_astrolines, sigma_astrolines, False),
    }

# ============================================================================
# BUILD MODELS IN XSPEC (NO FITTING)
# ============================================================================
AllModels.clear()
m_mw = Model("tbabs*(apec + powerlaw) + tbabs*(gaussian)", "mw", 1)

# Set MW model with BEST-FIT values
m_mw.TBabs.nH = 0.010052269039695583
m_mw.apec.kT = 0.4842643070700259
m_mw.apec.norm = 0.06963132985602272
m_mw.powerlaw.PhoIndex = 0.825356486567105
m_mw.powerlaw.norm = 0.16038135277432986
m_mw.gaussian.LineE = 1.206
m_mw.gaussian.Sigma = 0.00020099999999999998
m_mw.gaussian.norm = 1e-13

# Freeze parameters according to keyfit values
m_mw.TBabs.nH.frozen = False  # True in keyfit
m_mw.apec.kT.frozen = False  # True in keyfit
m_mw.apec.norm.frozen = False  # True in keyfit
m_mw.apec.Abundanc.frozen = True  # False in keyfit
m_mw.apec.Redshift.frozen = True  # False in keyfit
m_mw.powerlaw.PhoIndex.frozen = False  # True in keyfit
m_mw.powerlaw.norm.frozen = False  # True in keyfit
m_mw.TBabs_4.nH.frozen = True  # False in keyfit
m_mw.gaussian.LineE.frozen = True  # False in keyfit
m_mw.gaussian.Sigma.frozen = True  # False in keyfit
m_mw.gaussian.norm.frozen = True  # False in keyfit

# Build PB model string
n_gauss = len(model_dic_pb["gaussian"])
model_string_pb = "powerlaw"
if n_gauss > 0:
    model_string_pb += " + " + " + ".join(["gaussian"] * n_gauss)

AllModels += (model_string_pb, "mpb", 2)

# Set PB model with BEST-FIT values
mpb = AllModels(1, "mpb")
mpb.powerlaw.PhoIndex = 0.0
mpb.powerlaw.norm = 1e-13

# Set IB line parameters (first gaussian)
if hasattr(mpb, 'gaussian'):
    mpb.gaussian.norm = 0.0006291944896261725
    mpb.gaussian.LineE = 1.5124583151742277
    mpb.gaussian.Sigma = 0.0001
    mpb.gaussian.norm.frozen = False  # True in keyfit
    mpb.gaussian.LineE.frozen = True  # False in keyfit
    mpb.gaussian.Sigma.frozen = True  # False in keyfit

for i in range(1, len(IBALine)):
    gauss_name = f'gaussian_{i+2}'
    if hasattr(mpb, gauss_name):
        gauss_obj = getattr(mpb, gauss_name)
        gauss_obj.norm = 0.003446429422432
        gauss_obj.LineE = IBALine[i]
        gauss_obj.Sigma = sigma_astrolines
        gauss_obj.norm.frozen = True
        gauss_obj.LineE.frozen = True
        gauss_obj.Sigma.frozen = True
# ============================================================================
# PLOT MODEL WITH DATA (NO FITTING)
# ============================================================================

Xset.chatter = 0
Plot.commands = ()
Plot.add = True
Plot.xAxis = "keV"
Plot("ldata")

# Extract data
plot_grp = 1
energies = np.asarray(Plot.x(plot_grp))
edeltas = np.asarray(Plot.xErr(plot_grp))
rates = np.asarray(Plot.y(plot_grp))
errors = np.asarray(Plot.yErr(plot_grp))
foldedmodel = np.asarray(Plot.model(plot_grp))
nAddComps = int(Plot.nAddComps(plot_grp))

print(f"\nNumber of model components: {nAddComps}")

# Extract all components
modelcomp = np.zeros((nAddComps, len(energies)))
for i in range(nAddComps):
    plot_grp_comp = i + 1
    modelcomp[i] = np.asarray(Plot.addComp(plot_grp_comp))

# ============================================================================
# IDENTIFY COMPONENTS BY EXAMINING MAX VALUES
# ============================================================================

print("\nComponent max values from XSPEC Plot:")
for i in range(nAddComps):
    max_val = np.max(modelcomp[i])
    print(f"  Component {i+1}: max = {max_val:.3e}")

# Build component labels based on order and examination
# MW model: TBabs * (apec + powerlaw + gaussian)
# PB model: powerlaw + gaussian(s)
# The first 3 components should be from MW, rest from PB

comp_labels = []

# Components from MW model (first 3)
comp_labels.append('apec (MW-absorbed)')
comp_labels.append('powerlaw (MW-absorbed)')
comp_labels.append('gaussian DM (MW-absorbed)')

# Components from PB model
comp_labels.append('powerlaw (PB-background)')

# IB lines from PB
for i in range(len(IBALine)):
    comp_labels.append(f'IB Line {IBALine[i]:.3f} keV (PB)')

# Ensure we have enough labels
while len(comp_labels) < nAddComps:
    comp_labels.append(f'Component {len(comp_labels)+1}')

print("\nComponent identification:")
for i in range(nAddComps):
    max_val = np.max(modelcomp[i])
    print(f"  {i+1}: {comp_labels[i]} (max={max_val:.2e})")

# ============================================================================
# CREATE PLOT
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Plot data
ax.errorbar(energies, rates, xerr=edeltas, yerr=errors, 
            fmt='.', markersize=6, label='Data', color='black', alpha=0.7, linewidth=2)

# Plot total model
ax.plot(energies, foldedmodel, label='Total Model', color='blue', linewidth=2.5)

# Define colors for components
colors = ['green', 'red', 'darkred', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
linestyles = ['--', '--', '--', '-.', ':', ':', ':', ':', ':', ':']

# Plot each component
for i in range(nAddComps):
    color = colors[i % len(colors)]
    linestyle = linestyles[i % len(linestyles)]
    label = comp_labels[i] if i < len(comp_labels) else f'Component {i+1}'
    
    if np.max(modelcomp[i]) > 1e-15:
        ax.plot(energies, modelcomp[i], label=label, linestyle=linestyle, 
                color=color, linewidth=1.8, alpha=0.85)

ax.set_xlabel('Energy (keV)', fontsize=14)
ax.set_ylabel('counts/s/keV', fontsize=14)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(np.min(rates[rates > 0]) * 0.1, np.max(rates) * 2)
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='best', fontsize=9, ncol=2)
ax.set_title(f'Initial Model at E = {Eline:.3f} keV ({telescope})', fontsize=14, weight='bold')
plt.tight_layout()
# plt.savefig('InitialModel.pdf', dpi=300, bbox_inches='tight')
# print("\n✓ Saved: InitialModel.pdf")
plt.show()

# Print detailed component breakdown
print("\n" + "="*70)
print("MODEL COMPONENT BREAKDOWN (INITIAL VALUES)")
print("="*70)
for i in range(nAddComps):
    label = comp_labels[i] if i < len(comp_labels) else f'Component {i+1}'
    max_val = np.max(modelcomp[i])
    integral = np.trapz(modelcomp[i], energies)
    print(f"{i+1:2d}. {label:45s} | max: {max_val:10.2e} | integral: {integral:10.2e}")
print("="*70)

Xset.closeLog()