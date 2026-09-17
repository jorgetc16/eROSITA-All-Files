# %%
import os
from xspec import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from decimal import Decimal
from scipy import optimize, stats
from iminuit import Minuit
import h5py
from scipy.stats import chi2 as chi2_scipy
import sys  
from scipy.ndimage import uniform_filter1d
from datetime import datetime


Xset.allowPrompting = False # keeps pyxspec from hanging, waiting for a response to a prompt
Xset.parallel.steppar = 10 # number of threads for steppar
Fit.query = "yes"  # automatically answer "yes" to prompts
# # GOAL: to fit spectra in a window around the line. Use DE to fit
#
## Model used to fit ALL DATA from 1 keV to 2 keV NOT in WINDOWS.
# Script fit Only astro (noDM) and save the result in h5 files (7, one for each TM).
# These results can be used by Fitpoly_MT_1to2_DE_DMprofile.py which
# reads the fit astro and then compute the profile likelihood (for a given DM mass)
#
# Model: TBAS*(APEC+PL)+TBAS*Gaussian_DM + (PL_IB + Gaussian)_IB where _IB means not ARF
# Use multiresponse.
# The second APEC has a TABS with nH=0.22e22 cm-2 frozen (see below for the value).
# This is to avoid unphysical values that can affect
# the absorption of the DM line (norm of DM) during the fit.
# Here I use the model setting PL (with ARF) to small values and frozen.
#
# Here I load the data

# %%
AllData.clear()
path="DATA/srctoolout_000_SourceProducts_00001_5deg_rebin80_DECEMBER/"
#filespectrum=path+"srctoolout_120_SourceSpec_00001.fits"
#fileRMF=path+"srctoolout_120_RMF_00001.fits"
#fileARF=path+"srctoolout_120_ARF_00001.fits"
filespectrumTM1=path+"srctoolout_120_SourceSpec_00001.fits"
fileRMFTM1=path+"srctoolout_120_RMF_00001.fits"
fileARFTM1=path+"srctoolout_120_ARF_00001.fits"
fileIBLineTM1="DATA/IBLines/IBLine_TM1.txt"
Energy_width_TM1 = "DATA/Energy_width/Energ_Width_TM1.csv"
#
filespectrumTM2=path+"srctoolout_220_SourceSpec_00001.fits"
fileRMFTM2=path+"srctoolout_220_RMF_00001.fits"
fileARFTM2=path+"srctoolout_220_ARF_00001.fits"
fileIBLineTM2="DATA/IBLines/IBLine_TM2.txt"
Energy_width_TM2 = "DATA/Energy_width/Energ_Width_TM2.csv"
#
filespectrumTM3=path+"srctoolout_320_SourceSpec_00001.fits"
fileRMFTM3=path+"srctoolout_320_RMF_00001.fits"
fileARFTM3=path+"srctoolout_320_ARF_00001.fits"
fileIBLineTM3="DATA/IBLines/IBLine_TM3.txt"
Energy_width_TM3 = "DATA/Energy_width/Energ_Width_TM3.csv"
#
filespectrumTM4=path+"srctoolout_420_SourceSpec_00001.fits"
fileRMFTM4=path+"srctoolout_420_RMF_00001.fits"
fileARFTM4=path+"srctoolout_420_ARF_00001.fits"
fileIBLineTM4="DATA/IBLines/IBLine_TM4.txt"
Energy_width_TM4 = "DATA/Energy_width/Energ_Width_TM4.csv"
#
filespectrumTM5=path+"srctoolout_520_SourceSpec_00001.fits"
fileRMFTM5=path+"srctoolout_520_RMF_00001.fits"
fileARFTM5=path+"srctoolout_520_ARF_00001.fits"
fileIBLineTM5="DATA/IBLines/IBLine_TM5.txt"
Energy_width_TM5 = "DATA/Energy_width/Energ_Width_TM5.csv"
#
filespectrumTM6=path+"srctoolout_620_SourceSpec_00001.fits"
fileRMFTM6=path+"srctoolout_620_RMF_00001.fits"
fileARFTM6=path+"srctoolout_620_ARF_00001.fits"
fileIBLineTM6="DATA/IBLines/IBLine_TM6.txt"
Energy_width_TM6 = "DATA/Energy_width/Energ_Width_TM6.csv"
#
filespectrumTM7=path+"srctoolout_720_SourceSpec_00001.fits"
fileRMFTM7=path+"srctoolout_720_RMF_00001.fits"
fileARFTM7=path+"srctoolout_720_ARF_00001.fits"
fileIBLineTM7="DATA/IBLines/IBLine_TM7.txt"
Energy_width_TM7 = "DATA/Energy_width/Energ_Width_TM7.csv"
#
#
#Put here the TM you want to use
#

outdir = sys.argv[1] # output directory
os.makedirs(outdir, exist_ok=True)
telescope = sys.argv[2] 

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
else:
    print(f"Error: Unknown telescope '{telescope}'. Valid options: TM1, TM2, TM3, TM4, TM5, TM6, TM7, ALL_CLEAN")
    sys.exit(1)
# vfilespectrum = [filespectrumTM4]
# vfileRMF = [fileRMFTM4]
# vfileARF = [fileARFTM4]
#
#LOAD DATA
#

#
#"multiresponse array attribute for assigning multiple detectors (or sources) to a spectrum. The standard 0-based Python array indices corresponding to the 1-based XSPEC
#source numbers"
#
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

# %% [markdown]
# Fix the energy of the DM line and select data

# %%
# I do not consider eROSITA data outside this range
Eallmin = 0.3
Eallmax = 9.


Emin= 1.0
Emax= 2.2
Eline = (Emin + Emax)/2.

sigma = np.interp(Eline,vE,vsigma)
print("Eline ",Eline," sigma ",sigma)

if(Emin<Eallmin):
    Emin=Eallmin
if(Emax>Eallmax):
    Emax=Eallmax
print("Emin ",Emin," Emax ",Emax)
str_range = "**-"+str(round(Emin, 4))+",,"+str(round(Emax, 4))+"-**"
print(str_range)
AllData.ignore(str_range)


Plot.commands = ()
Plot.xAxis = "keV"
Plot("data")
#ax.clear()
# Compute number of E bins, assumig it is the same for all TM
NEbins = len(np.asarray(Plot.x(1)))
print("NEbins ",NEbins)
venergies = np.zeros((len(vfilespectrum), NEbins))
vedeltas = np.zeros((len(vfilespectrum), NEbins))
vrates = np.zeros((len(vfilespectrum), NEbins))
verrors = np.zeros((len(vfilespectrum), NEbins))
for i in range (0,len(vfilespectrum)):
    plot_grp = i + 1 # so that it starts with 1
    energiesTMi = np.asarray(Plot.x(plot_grp))
    edeltasTMi = np.asarray(Plot.xErr(plot_grp))
    ratesTMi = np.asarray(Plot.y(plot_grp))
    errorsTMi = np.asarray(Plot.yErr(plot_grp))
    venergies[i] = energiesTMi 
    vedeltas[i] = edeltasTMi 
    vrates[i] = ratesTMi 
    verrors[i] = errorsTMi 
    print("***************")
    print("Selected values ",np.min(energiesTMi)," ",np.max(energiesTMi)," Nbins ",len(energiesTMi))
    print("***************")
    channelmin = AllData(plot_grp).noticed[0] -1
    channelmax = AllData(plot_grp).noticed[-1]
    #print("channelmin/max ",channelmin," ",channelmax)
    #
    SpectrumOpen = fits.open(vfilespectrum[plot_grp-1])['SPECTRUM']
    counts_TMi = SpectrumOpen.data.field('COUNTS')#
    channel_TMi = SpectrumOpen.data.field('CHANNEL')#
    exposure_TMi=fits.getval(vfilespectrum[plot_grp-1],"exposure", ext=1)
    RMFOpen = fits.open(vfileRMF[plot_grp-1])['EBOUNDS']
    Emin_data_TMi = RMFOpen.data.field('E_MIN')#keV
    Emax_data_TMi = RMFOpen.data.field('E_MAX')#keV
    Eave_data_TMi = (Emin_data_TMi+Emax_data_TMi)/2.
    DeltaE_data_TMi = Emax_data_TMi-Emin_data_TMi
    DeltaE_ana_TMi = DeltaE_data_TMi[channelmin:channelmax]
    Eave_ana_TMi = Eave_data_TMi[channelmin:channelmax]
    channel_ana_TMi = channel_TMi[channelmin:channelmax]
    counts_ana_TMi = counts_TMi[channelmin:channelmax]
    print("Min Max counts ",np.min(counts_ana_TMi)," ",np.max(counts_ana_TMi))



# %%
#ax.clear()
fig, ax = plt.subplots(figsize=(10,6))
for i in range (0,len(vfilespectrum)):
    ax.errorbar(venergies[i], vrates[i], xerr=vedeltas[i],yerr=verrors[i],fmt='.',markersize='5',label='data')
#ax.errorbar(energies, rates, xerr=edeltas,yerr=errors,fmt='.',markersize='5',label='data')
#ax.axhline(y=10)
ax.set_xlabel('Energy (keV)')
ax.set_ylabel(r'counts/s/keV')
ax.set_xscale("log")
ax.set_yscale("log")
#ax.set_ylim((0.001,0.1))
ax.grid()
ax.legend()

# %% [markdown]
# Now I consider the Astro and Internal Background lines falling into the energy range +- 7 sigma. At the end I define arrays containing the position of the lines and the min and max values inside which they can fluctuate in the fit
# 

# %%
#EminL = Eline - 7*sigma
#EmaxL = Eline + 7*sigma
#if(EminL<Eallmin):
#    EminL = Eallmin
#if(EmaxL>Eallmax):
#    EmaxL = Eallmax
    
EminL= Emin
EmaxL= Emax
    

if isinstance(vfileIBLine, list):
    ib_file = vfileIBLine[0]
else:
    ib_file = vfileIBLine

# Load IB lines
data = np.loadtxt(ib_file, skiprows=0)
IBenergy = data[:,0]
#IBenergymin = data[:,2]
#IBenergymax = data[:,5]
IBenergymin = data[:,0]-0.025
IBenergymax = data[:,0]+0.025
#print(Aenergy," ",IBenergy)
#print(len(Aenergy)," ",len(IBenergy))
IBidx = np.where( (IBenergy > EminL ) & (IBenergy < EmaxL ) )
IBLine = IBenergy[IBidx]
IBLinemin = IBenergymin[IBidx]
IBLinemax = IBenergymax[IBidx]





IBn = len(IBLine)
#
# For this model avoid Astro lines
#
IBALine = IBLine.copy()
IBALinemin = IBLinemin.copy()
IBALinemax = IBLinemax.copy()
IBAn = IBn
idx = np.argsort(IBALine)
IBALine = IBALine[idx]
IBALinemin = IBALinemin[idx]
IBALinemax = IBALinemax[idx]
print("Total lines ",IBAn," IB ",IBn)
print(IBALine)
print(IBALinemin," ",IBALinemax)



# %% [markdown]
# FUNCTIONS TO SET LINES TO ZERO AND FIX THEM OR TO FREE THEM

# %%
# CLASS TO DEFINE A FUNCTION TO EVALUATE THE TS specified in fit GIVEN A xspec Model model
# with paramers value params, name (gaussian_3, powerlaw,...) params_name and target (norm, LineE,...) params_target

class MTmodel:
    def __init__(self, model, params_name, params_target, fit):
        self.model = model
        self.params_name = params_name
        self.params_target = params_target
        self.fit = fit

        # Precompile fast setters for just the main value (index 0 of .values)
        # Finds the XSPEC parameter object via getattr(model, name).attr for example: model.powerlaw.PhoIndex
        # Calls _make_fast_setter(param_obj)
        # Stores the resulting function in a list
        self._param_fast_setters = [
            self._make_fast_setter(getattr(getattr(model, name), attr))
            for name, attr in zip(params_name, params_target)
        ]

        # Also keep full parameter objects for set_values_all
        self._param_objects = [
            getattr(getattr(model, name), attr)
            for name, attr in zip(params_name, params_target)
        ]
    
    #It creates a closure that:
    #remembers the XSPEC parameter object
    #updates its value only (fast update)
    def _make_fast_setter(self, param_obj):
        def setter(value, param_obj=param_obj):
            #param_obj.values[0] = value
            # scalar assignment → only sets values[0]
            param_obj.values = value
        return setter

    #set value of parameters
    #if params = [1.2, 3.4, 6.7], this calls: setter_0(1.2) setter_1(3.4) setter_2(6.7)
    def set_values(self, params):
        for setter, value in zip(self._param_fast_setters, params):
            setter(float(value))  # direct, fast assignment

    #set value of parameters including upper/lower limits
    def set_values_all(self, params, lowers, uppers, keyfits):
        """
        Set parameter values and limits.
        keyfits[i] == True  → parameter is free
        keyfits[i] == False → parameter is frozen
        """
        for i, param in enumerate(self._param_objects):
            a = float(params[i])
            b = float(lowers[i])
            c = float(uppers[i])
            step = float(np.abs(1e-2*a))
            # for frozen parameters
            if not keyfits[i]:
                # --- Frozen parameter: safe assignment (no step, identical limits)
                param.values = [a, step, b, b, c, c]
                param.frozen = True
                continue

            param.values = [a, step, b, b, c, c]
            param.frozen = False

    # evaluate TS given paramaters params
    def evaluate_folded(self, params):
        self.set_values(params)
        return self.fit.statistic

    # set a given params_name, params_target to keyfits, i.e. True/False. Relevant for fit with xSpec
    # Careful: my logic is True means is in the fit therefore corresponds to .frozen = False
    def set_frozen(self, keyfits):
        """
        Freeze or unfreeze parameters based on keyfits.
        keyfits[i] == True  → parameter is free (frozen = False)
        keyfits[i] == False → parameter is frozen (frozen = True)
        """
        if len(keyfits) != len(self.params_name):
            raise ValueError("keyfits must be the same length as params_name and params_target")

        for param, keep_free in zip(self._param_objects, keyfits):
            param.frozen = not keep_free  # Invert: True means not frozen

    # return the value of the parameters    
    def get_values(self):
        """Return current values (i.e., param.values[0]) as numpy array."""
        return np.array([param.values[0] for param in self._param_objects])

# %%
# Functions working on dictionary

# Given a dictionary model_dic,
# extract all entries as numpy arrays
def Extract_array_dic(model_dic):
    params = []
    lowers = []
    uppers = []
    keyfits = []
    params_name = []
    params_target = []
    for top_key, sub_models in model_dic.items():
        # Skip non-model entries like _structure
        if not isinstance(sub_models, dict):
            continue
        for sub_key, parameters in sub_models.items():
            for paramtarget, param_info in parameters.items():
                params.append(param_info["value"])
                lowers.append(param_info["lower"])
                uppers.append(param_info["upper"])
                keyfits.append(param_info["keyfit"])
                params_name.append(sub_key)     # second-level key (e.g., powerlaw_2)
                params_target.append(paramtarget)     # third-level key (e.g., PhoIndex)

    # Convert everything to NumPy arrays
    params = np.array(params)
    lowers = np.array(lowers)
    uppers = np.array(uppers)
    params_name = np.array(params_name)
    params_target = np.array(params_target)
    keyfits = np.array(keyfits)
    keyfits = np.array(keyfits)
    return (params,lowers,uppers,params_name,params_target,keyfits)

# %%
# Functions working on numpy arrays

# input arrays defining all the parameters of the model
# output: arrays of parameters entering in the fit (keyfits is True)
def paramfit(params, lowers, uppers , params_name, params_target, keyfits):
    # Create a mask for where keyfits is True
    mask = keyfits == True
    #print(keyfits)
    #print(mask)
    # Apply the mask
    params_out = params[mask]
    lowers_out = lowers[mask]
    uppers_out = uppers[mask]
    params_name_out = params_name[mask]
    params_target_out = params_target[mask]
    keyfits_out = keyfits[mask]
    bounds_out = np.column_stack((lowers_out, uppers_out))
    #print("Values:", params)
    #print("Lowers:", lowers)
    #print("Uppers:", uppers)
    #print("Keyfits:", keyfits)
    #print("Submodel keys:", params_name)
    #print("Param keys:", params_target)
    return (params_out, bounds_out, params_name_out, params_target_out, keyfits_out)

# Select element of params_name, params_target associated to params_name, target_key
# and for that change keyfits to keyfits_value
def change_keyfits(params_name, params_target, keyfits, name_key, target_key, keyfits_value):
    mask = np.logical_and(params_name == name_key, params_target == target_key)
    idx = np.flatnonzero(mask) #True==1
    #print(idx[0]," ",name_key[idx[0]])
    keyfits[idx[0]] = keyfits_value

# For params associated to name_key, target_key change the value to key_value 
def change_param_value(params, params_name, params_target, name_key, target_key, value):
    mask = np.logical_and(params_name == name_key, params_target == target_key)
    idx = np.flatnonzero(mask)
    params[idx[0]] = value

# Return params associated to name_key, target_key
def return_param_value(params, params_name, params_target, name_key, target_key):
    mask = np.logical_and(params_name == name_key, params_target == target_key)
    idx = np.flatnonzero(mask)
    return params[idx[0]]

# Print parameters
def printparam(params, lowers, uppers , params_name, params_target, keyfits):
    for i in range(0,len(params_name)):
        print(params_name[i]," ",params_target[i]," ", params[i]," ", lowers[i]," ", uppers[i]," ", keyfits[i])

TSth = 2.71
def MTroots(vx,vy):
    vroots = []
    for i in range(1,len(vx)):
        #print("vx ",vx[i]," ",vy[i])
        #Check whether sign has changed
        if(vy[i]*vy[i-1]<0.):
            root=(vx[i]*vy[i-1]-vx[i-1]*vy[i])/(vy[i-1]-vy[i])
            #print(vx[i-1]," ",vx[i]," ",vy[i-1]," ",vy[i])
            #print("i ", i," ",root," ")
            vroots.append(root)
    return  vroots

def chi2approx(fullmodel_xp, params_fit, params_name_fit, params_target_fit, A):
    # Change DM normalization
    params_inside = params_fit
    change_param_value(params_inside, params_name_fit, params_target_fit, "gaussian", "norm", A)
    Fit.statMethod = "cstat"  # fitting statistics = Cash-statistics
    mymodel_fit = MTmodel(fullmodel_xp, params_name_fit, params_target_fit, Fit)
    # evaluation
    ### MT: no need to fix DM line norm since here I just do an evaluation, not a fit
    res = mymodel_fit.evaluate_folded(params_inside)
    #print("A ",A," TS ",res)
    #PrintModelXspec()
    #Check the model!
    #Xset.chatter = 10
    #AllModels.show()
    #Xset.chatter = 1
    return res

def ApproximateBound(model_final, DMbf, normlDMhdata2, nstepsDM, TS_all, params_fit, params_name_fit, params_target_fit):
    start = np.max(np.array([DMbf,normlDMhdata2*1e-5]))
    end = normlDMhdata2
    profile_range = np.logspace(np.log10(start), np.log10(end), num = nstepsDM)
    TS_vals = np.zeros_like(profile_range)
    for i, A in enumerate(profile_range):
        TS_vals[i] = chi2approx(model_final, params_fit, params_name_fit, params_target_fit, A)
        #print("A TS ",A," ",TS_vals[i])
    x = np.log10(profile_range)
    y = TS_vals - TS_all - TSth
    vroots = np.pow(10.,MTroots(x,y))
    if(len(vroots)>0):
        bound_approx=np.max(vroots)
    else:
        bound_approx=0.
    return bound_approx

def BoundProfiling(profile_range, delta_TS):
    x = np.log10(profile_range)
    y = delta_TS - TSth
    vroots = np.pow(10.,MTroots(x,y))
    if(len(vroots)>0):
        bound=np.max(vroots)
    else:
        bound=0.
    return bound
    
def PrintModelXspec():
    for _name in m_mw.componentNames:
        _comp = m_mw.__getattribute__(_name)
        for _pname in _comp.parameterNames:
            _par = _comp.__getattribute__(_pname)
            print(_name," ",_pname," ",_par.values)
    print(params_name_pb[0]," ",params_target_pb[0]," ",AllModels(1,"mpb").powerlaw.PhoIndex.values)
    print(params_name_pb[1]," ",params_target_pb[1]," ",AllModels(1,"mpb").powerlaw.norm.values)
    for _name in AllModels(1,"mpb").componentNames:
        if _name == "gaussian" or (_name.startswith("gaussian_") and int(_name.split('_')[-1]) <= 30):
            _comp = AllModels(1,"mpb").__getattribute__(_name)
            for _pname in _comp.parameterNames:
                _par = _comp.__getattribute__(_pname)
                print(_name," ",_pname," ",_par.values)

def freeze_ib_line_energies(freeze=True):
    """Freeze/unfreeze IB Gaussian line energies in mpb model."""
    for name, tgt in zip(params_name_pb, params_target_pb):
        if name.startswith("gaussian") and tgt == "LineE":
            #print("changing")
            change_keyfits(params_name_pb, params_target_pb, keyfits_pb, name, "LineE", not freeze)  # True=free, False=frozen
    mymodel_pb.set_frozen(keyfits_pb)

#def reseed_ib_norms_small():
#    """Re-seed IB Gaussian norms to a small value to avoid bad basins."""
#    for name, tgt in zip(params_name_pb, params_target_pb):
#        if name.startswith("gaussian") and tgt == "norm":
#            change_param_value(params_pb, params_name_pb, params_target_pb, name, "norm", normline_start)
#    mymodel_pb.set_values_all(params_pb, lowers_pb, uppers_pb)

def jitter_params(prev_params, prev_params_lower, prev_params_upper, prev_params_keyfit, rel=0.02):
    """Small multiplicative jitter (±rel) to escape shallow basins."""
    p = np.array(prev_params, dtype=float)
    #eps = (1.0 + rel) ** (2.0*np.random.rand(p.size) - 1.0)
    for i in range(0,len(prev_params)):
        #print(prev_params_keyfit[i]," ",prev_params[i]," ",prev_params_lower[i]," ",prev_params_upper[i])
        if  prev_params_keyfit[i] == True:
            eps = (1.0 + rel) ** (2.0*np.random.rand() - 1.0)
            #print(prev_params_keyfit[i]," ",prev_params[i]," ",prev_params_lower[i]," ",prev_params_upper[i]," ",eps)
            if  (prev_params[i]*eps > prev_params_lower[i])   and   (prev_params[i]*eps < prev_params_upper[i]):
                p[i] = prev_params[i]*eps
            else:
                p[i] = np.random.uniform(prev_params_lower[i], prev_params_upper[i])
                #print("Not to much ")
        else:
            p[i] = prev_params[i]
    return p

def local_predict_ts(logA_prev2, TS_prev2, logA_prev1, TS_prev1, logA_i):
    """Linear extrapolation in log(A) to estimate 'expected' TS."""
    #print("local pred")
    #print(logA_prev2," ",TS_prev2," ",logA_prev1," ",TS_prev1," ",logA_i)
    denom = (logA_prev1 - logA_prev2)
    if denom == 0:
        return TS_prev1
    slope = (TS_prev1 - TS_prev2) / denom
    return TS_prev1 + slope * (logA_i - logA_prev1)
    
def read_fit_astro(folder):
    """Read astro fit results from h5 file."""
    astro_file = os.path.join(folder, "fit_results_Astro.h5")
    with h5py.File(astro_file, "r") as f:
        astro_grp = f["Fit_Astro"]["astro"]
        params_astro_mw = astro_grp["params_astro_mw"][:]
        params_astro_pb = astro_grp["params_astro_pb"][:]
        TS_astro = astro_grp["TS_astro"][()]
        pvalue_astro = astro_grp["p-value-nullhyp"][()]
        dof_astro = astro_grp["dof_astro"][()]
        nfit_astro = astro_grp["nfit_astro"][()]
    return params_astro_mw, params_astro_pb, TS_astro, pvalue_astro, dof_astro, nfit_astro
    
    

# %%
# COMPUTE norm of the line to hit data
# to have an idea of a sensible normalization
# For simplicity choose TM1
arf1test = fits.open(fileARFTM1)['SPECRESP']
arf1ELOW = arf1test.data.field('ENERG_LO')#keV
arf1EHI = arf1test.data.field('ENERG_HI')#keV
arf1Aeff = arf1test.data.field('SPECRESP')#cm2
arf1E = (arf1ELOW + arf1EHI)/2.
Aefftest = np.interp(Eline, arf1E, arf1Aeff)
Ratetest = np.interp(Eline, venergies[0], vrates[0])
normlDMhdata = Ratetest/Aefftest # counts/keV/s/cm2
print("normlDMhdata [counts/keV/s/cm2]",normlDMhdata)
# normlDMhdata2 is the correct normalization to set norm of DM gaussian line to a reasonable value! 
# It has the same units of norm of Gaussian
# See definition of gaussian line in https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node182.html
normlDMhdata2 = normlDMhdata*np.sqrt(2*np.pi)*sigma # counts/s/cm2
# normlDMhdata2 = Ratetest*np.sqrt(2*np.pi)*sigma # counts/s/cm2
print("normlDMhdata2 [counts/s/cm2]",normlDMhdata2)


# %% [markdown]
# Create the dictionary of the model

# %%

# model 6
#This is a model based on TBabsorption and APEC phenomenological models
def pdic(value, lower, upper, key):
    return {"value": value, "lower": lower, "upper": upper, "keyfit": key}

smallnormlower = 1e-13 # if you want to avoid setting lower limit to zero
smallnorm = 1e-10 # if you want to avoid setting norm to zero


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
upperapec = 2.0 #2.0 80.0
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
# I always define DM to be the first gaussian. I set parameters to False (no in the fit)
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
            "norm": pdic(0, 0, 0, False), # (value, lower, upper, key)
            }
        }
    }
# gaussian_components = [f"gaussian" for i in range(len(IBALine))]
# gaussian_sum = " + ".join(gaussian_components)
# if len(IBALine)>0:
#     model_dic["_structure"] = f"gaussian+ {gaussian_sum}" 
# else:
model_dic["_structure"] = f"tbabs*(apec + powerlaw) + tbabs*(gaussian)"
    
#ADD gaussian lines
#
# Count all second-level keys in the entire model_dic
# print(model_dic.values())
start_index_astrol = sum(
    len(sub_models) for sub_models in model_dic.values()
    if isinstance(sub_models, dict)
)
start_index_astrol = start_index_astrol + 1
print("start_index_astrol ",start_index_astrol)


model_dic_pb = {
    
    "powerlaw": {
        "powerlaw": {
            "PhoIndex": pdic(startidx, loweridx, upperidx, True), # (value, lower, upper, key)
            "norm" : pdic(startpl, lowerpl, upperpl, True)
            }
        },
        "gaussian": {}  #
    }
#
# Start numbering from start_index_astrol to avoid conflict with "gaussian"
for i in range(len(IBALine)):
    if i == 0:
        g_key = "gaussian"
    else:
        g_key = f"gaussian_{i+2}"  # XSPEC adds suffixes starting at _2
    model_dic_pb["gaussian"][g_key] = {
        "norm": pdic(normline_start, normline_lower, normline_upper, True),
        "LineE": pdic(IBALine[i], IBALinemin[i], IBALinemax[i], False), #True
        "Sigma": pdic(sigma_astrolines, sigma_astrolines, sigma_astrolines, False)
    }




# %%
print("******")
print(model_dic)
print("******")
print(model_dic_pb)
print("******")

# %% [markdown]
# Define the model

# %%
# Create the XSpec model
if "_structure" in model_dic:
    finalM_xp = model_dic["_structure"]
else:
    baseM_parts = []
    for top_key, sub_models in model_dic.items():
        count = len(sub_models)
        baseM_parts.extend([top_key] * count)
    finalM_xp = '+'.join(baseM_parts)

print("mw ", finalM_xp)
AllModels.clear()
m_mw = Model(finalM_xp, "mw", 1)
#
#Add Particle bkg power law
#
# Build model string: one powerlaw + N gaussian components
n_gauss = len(model_dic_pb["gaussian"])
print("n_gauss ",n_gauss)
model_string_pb = "powerlaw"
if n_gauss > 0:
    model_string_pb += " + " + " + ".join(["gaussian"] * n_gauss)
print("mpb model string:", model_string_pb)
model_dic_pb["_structure"] = model_string_pb
AllModels += (model_string_pb, "mpb", 2)

#
print("MPB model components:", [name for name in dir(AllModels(1, "mpb")) if "gaussian" in name])

# cstat: https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node340.html
#For sufficiently large counts per bin, the C-statistic distribution
#closely approximates a χ² distribution, making χ²-based goodness-of-fit tests a good approximation in the central probability range.
#Kaastra (2017)
Xset.chatter = 0
Fit.statMethod = "cstat"
Fit.nIterations = 100000
#
# Extract the relevant info from the model dictionary
params, lowers, uppers, params_name, params_target, keyfits = Extract_array_dic(model_dic)
#
mymodel_mw = MTmodel(model=m_mw, params_name=params_name, params_target=params_target, fit=Fit)
#
# Extract the relevant info from the model dictionary of PB
params_pb, lowers_pb, uppers_pb, params_name_pb, params_target_pb, keyfits_pb = Extract_array_dic(model_dic_pb)
#
mymodel_pb = MTmodel(model=AllModels(1,"mpb"), params_name=params_name_pb, params_target=params_target_pb, fit=Fit)
#
#
# set the values including the upper and lower limits in Xspec
mymodel_mw.set_values_all(params, lowers, uppers,keyfits)
# set the free parameters in Xspec
mymodel_mw.set_frozen(keyfits)
#
mymodel_pb.set_values_all(params_pb, lowers_pb, uppers_pb, keyfits_pb)
mymodel_pb.set_frozen(keyfits_pb)






# %%
# If not already specified in initial dictionarys set DM to False and norm to 0
change_keyfits(params_name, params_target, keyfits, "gaussian", "norm", False)
change_param_value(params, params_name, params_target, "gaussian", "norm", smallnormlower)
#
# since you have changed norm and key(True/False), update XSpec
mymodel_mw.set_values_all(params, lowers, uppers, keyfits)
mymodel_mw.set_frozen(keyfits)
# %%
# %%
Xset.chatter = 10
AllModels.show()
# Xset.closeLog()
Xset.chatter = 0


# %%
print("****************")
printparam(params, lowers, uppers , params_name, params_target, keyfits)
print("****************")
print("****************")
printparam(params_pb, lowers_pb, uppers_pb , params_name_pb, params_target_pb, keyfits_pb)
print("****************")
PrintModelXspec()
print("****************")
print("****************")


Fit.statMethod = "cstat"  # fitting statistics = Cash-statistics
print("**************")
# %%


# %%
# If not already specified in initial dictionarys set DM to False and norm to 0
change_keyfits(params_name, params_target, keyfits, "gaussian", "norm", False)
change_param_value(params, params_name, params_target, "gaussian", "norm", smallnormlower)
#
# since you have changed norm and key(True/False), update XSpec
mymodel_mw.set_values_all(params, lowers, uppers, keyfits)
mymodel_mw.set_frozen(keyfits)

Xset.chatter = 10
AllModels.show()
Xset.chatter = 1


# Simple tests for ensure_bounds_array
#params_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params, lowers, uppers, params_name, params_target, keyfits)
#print("bounds ",bounds_fit," shape ",bounds_fit.shape)
#print("reshape ", bounds_fit.reshape(-1, 2)," shape",bounds_fit.reshape(-1, 2).shape," shape[0] ",bounds_fit.reshape(-1, 2).shape[0])
#params_fit_pb, bounds_fit_pb, params_name_fit_pb, params_target_fit_pb, keyfits_fit_pb = paramfit(params_pb, lowers_pb, uppers_pb, params_name_pb, params_target_pb, keyfits_pb)
#print("bounds ",bounds_fit_pb," shape ",bounds_fit_pb.shape)
#print("reshape ", bounds_fit_pb.reshape(-1, 2)," shape",bounds_fit_pb.reshape(-1, 2).shape," shape[0] ",bounds_fit_pb.reshape(-1, 2).shape[0])
#zerotest = np.zeros((0, 2), dtype=float)
#print("zerotest ",zerotest," shape ",zerotest.shape," shape[0] ",zerotest.shape[0])



def ensure_bounds_array(bounds):
    b = np.asarray(bounds, dtype=float)
    if b.size == 0:
        return np.zeros((0, 2), dtype=float)
    b = b.reshape(-1, 2)
    return b
    
    
class FastCombinedModel:
    """
    Fastest possible XSPEC objective evaluator:
    - supports zero-length parameter blocks
    - no if-conditions inside evaluate()
    - uses precomputed slices
    """
    def __init__(self, mt_mw, mt_pb, bounds_mw, bounds_pb):

        # Normalize bounds
        bw = ensure_bounds_array(bounds_mw)
        bp = ensure_bounds_array(bounds_pb)

        self.bounds_mw = bw
        self.bounds_pb = bp

        self.n_mw = bw.shape[0]
        self.n_pb = bp.shape[0]

        # Combined bounds for DE
        self.bounds = np.vstack((bw, bp))  # works with 0-row arrays

        # Cache MTmodel references
        self.mt_mw = mt_mw
        self.mt_pb = mt_pb

        # Cached fast setters; will never be called with bad shapes
        self._set_mw = mt_mw.set_values
        self._set_pb = mt_pb.set_values

        # Fit object (same for both models)
        self._fit = mt_mw.fit

        # Precompute slices → avoids branching inside evaluate()
        self.slice_mw = slice(0, self.n_mw)
        self.slice_pb = slice(self.n_mw, self.n_mw + self.n_pb)

    def evaluate(self, params):
        #p = np.asarray(params)

        # These work even when n_mw or n_pb is zero
        self._set_mw(params[self.slice_mw])
        self._set_pb(params[self.slice_pb])
        return self._fit.statistic

#Now let's go back to FastCombineModel. It combines two different models of xspec and crucially it evaluates the chi2 (in a fast way to use then in DE). I want to

def FitInstance():
    try:
        Fit.perform()
        return True
    except Exception as e:
        print("Fit failed with error:", e)
        return False
        
def doFit(mymodel_mw, mymodel_pb,
        params_mw_orig, params_pb_orig,
        params_name_mw, params_target_mw,
        params_name_pb, params_target_pb,
        lowers_mw, uppers_mw, keyfits_mw,
        lowers_pb, uppers_pb, keyfits_pb,
        TypeFit):
        
    #mymodel_mw, mymodel_pb : MTmodels
    #MTmodel instances for MW and PB
    #params_mw_orig, params_pb_orig : np.ndarray of initial full parameter arrays (including frozen values).
    #params_name_mw, params_target_mw, params_name_pb, params_target_pb : arrays for parameters name and target (full, including frozen parameters)
    #lowers_mw, uppers_mw, keyfits_mw : arrays MW parameter bounds and free/frozen flags.
    #lowers_pb, uppers_pb, keyfits_pb : arrays PB parameter bounds and free/frozen flags.
    #
    #Returns
    #-------
    #full_mw, full_pb : np.ndarray Full parameter arrays after fit.
    #TS_out : float Fit statistic after fit.
    #pvalue_out : float Null-hypothesis p-value.
    #dof_out : int Degrees of freedom.
    #nfit_out : int Number of fitted parameters.
                    
    # https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html
    Fit.statMethod = "cstat"  # fitting statistics = Cash-statistics
    #Fit.nIterations=10000
    
    
    if TypeFit == "XSpec":
        mymodel_mw.set_values_all(params_mw_orig, lowers_mw, uppers_mw, keyfits_mw)
        mymodel_mw.set_frozen(keyfits_mw)
        #
        mymodel_pb.set_values_all(params_pb_orig, lowers_pb, uppers_pb, keyfits_pb)
        mymodel_pb.set_frozen(keyfits_pb)
        #
        fail = 0
        Fit.criticalDelta = 1e-3
        Fit.method = "migrad"
        if not FitInstance():
            fail = fail + 1
        
        Fit.method = "leven"
        Fit.criticalDelta = 1e-4
        if not FitInstance():
            fail = fail + 1
         
        Fit.method = "migrad"
        Fit.criticalDelta = 1e-5
        if not FitInstance():
            fail = fail + 1

        #Fit.method = "migrad"
        #Fit.criticalDelta = 1e-3
        #if not FitInstance():
        #    fail = fail + 1
            
        if fail >= 3:
            raise RuntimeError("Fit failed 4 times. Stopping code.")
         
        TS_out = Fit.statistic
        nfit_out = Fit.nVarPars
        dof_out = Fit.dof
        pvalue_out = Fit.nullhyp
        
        # Extract full arrays (including frozen parameters)
        full_mw = mymodel_mw.get_values()
        full_pb = mymodel_pb.get_values()
        
        
    elif TypeFit == "Diffev":
        #print("here",params_mw_orig)
        #out = optimize.differential_evolution(mymodel_fit.evaluate_folded, bounds_fit, polish = False,
        #                              tol = 1e-4, atol = 0, popsize = 60, init = 'sobol',
        #                             maxiter=100000) #maxiter=10000
        #
        # Select the parameters to Fit. Only instrumental background parameters in this fit
        params_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(
            params_mw_orig, lowers_mw, uppers_mw, params_name_mw, params_target_mw, keyfits_mw)
        params_fit_pb, bounds_fit_pb, params_name_fit_pb, params_target_fit_pb, keyfits_fit_pb = paramfit(
            params_pb_orig, lowers_pb, uppers_pb, params_name_pb, params_target_pb, keyfits_pb)
        #
        mymodel_mw_fit  = MTmodel(m_mw, params_name_fit,params_target_fit,Fit)
        mymodel_pb_fit  = MTmodel(AllModels(1,"mpb"), params_name_fit_pb, params_target_fit_pb, Fit)
        #
        # combine
        combined = FastCombinedModel(mymodel_mw_fit, mymodel_pb_fit, bounds_fit, bounds_fit_pb)
        
        # Create custom initial population if h5 values provided
        if TypeFit == "Diffev_custom" and init_values is not None:
            # Extract fitted parameters from h5 solution
            init_fit_mw = np.array([init_values[0][i] for i, fit in enumerate(keyfits_mw) if fit])
            init_fit_pb = np.array([init_values[1][i] for i, fit in enumerate(keyfits_pb) if fit])
            init_combined = np.concatenate([init_fit_mw, init_fit_pb])
            
            # Create population: 50% around h5 solution, 50% random
            popsize = 100
            population = np.zeros((popsize, len(init_combined)))
            
            # First half: jittered h5 solution (±5%)
            for i in range(popsize // 2):
                jitter = 1.0 + 0.05 * (2 * np.random.rand(len(init_combined)) - 1)
                population[i] = np.clip(init_combined * jitter, 
                                       combined.bounds[:, 0], 
                                       combined.bounds[:, 1])
            
            # Second half: Sobol random
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=len(init_combined), scramble=True, seed=42)
            sobol_samples = sampler.random(popsize - popsize // 2)
            for i in range(popsize // 2, popsize):
                population[i] = (combined.bounds[:, 0] + 
                               sobol_samples[i - popsize // 2] * 
                               (combined.bounds[:, 1] - combined.bounds[:, 0]))
            
            init_method = population
        else:
            init_method = 'sobol'
        
        print("="*70)
        print(f"STAGE 1: Coarse global search (init={init_method if isinstance(init_method, str) else 'custom'})")
        print("="*70)
        
        out1 = optimize.differential_evolution(
            combined.evaluate, 
            combined.bounds, 
            polish=False,
            tol=1e-2,
            atol=0,
            popsize=100,
            init=init_method,  # ← Use custom or 'sobol'
            maxiter=3000,
            workers=1,
            seed=42,
            updating='deferred',
            strategy='best1bin'
        )
        
        print(f"Stage 1 result: TS = {out1.fun:.4f}")
        
        print("="*70)
        print("STAGE 2: Fine-tuning from Stage 1 result")
        print("="*70)
        # Stage 2: Refine around Stage 1 result with tighter bounds
        if out1.success or out1.fun < 1e6:  # If Stage 1 found something reasonable
            # Tighten bounds around Stage 1 result (±20% of original range)
            bounds_refined = []
            for i, (lower, upper) in enumerate(combined.bounds):
                range_width = upper - lower
                center = out1.x[i]
                new_lower = max(lower, center - 0.2 * range_width)
                new_upper = min(upper, center + 0.2 * range_width)
                bounds_refined.append([new_lower, new_upper])
            bounds_refined = np.array(bounds_refined)
            
            out2 = optimize.differential_evolution(
                combined.evaluate,
                bounds_refined,
                polish=False,
                tol=1e-3,           # Tighter tolerance
                atol=1e-6,
                popsize=80,
                init='sobol',
                maxiter=5000,
                workers=1,
                seed=42,
                updating='deferred',
                strategy='best1bin'
            )
            print(f"Stage 2 result: TS = {out2.fun:.4f}")
            out = out2
        else:
            out = out1
        
        print("="*70)
        print("STAGE 3: XSPEC polish with Migrad/Leven")
        print("="*70)
        
        params_out = out.x
        TS_out = out.fun
        nfit_out = combined.bounds.shape[0]
        dof_out = len(venergies[0]) * len(vfilespectrum) - nfit_out
        pvalue_out = chi2_scipy.sf(TS_out, dof_out)
        
        print(f"DE result before XSPEC polish: TS = {TS_out:.4f}")
        
        n_mw = combined.n_mw
        n_pb = combined.n_pb
        params_out_mw = params_out[:n_mw]
        params_out_pb = params_out[n_mw:n_mw+n_pb]
        
        # Rebuild full parameter arrays
        full_mw = params_mw_orig.copy()
        full_pb = params_pb_orig.copy()
        
        for val, name, attr in zip(params_out_mw, params_name_fit, params_target_fit):
            change_param_value(full_mw, params_name_mw, params_target_mw, name, attr, val)
        
        for val, name, attr in zip(params_out_pb, params_name_fit_pb, params_target_fit_pb):
            change_param_value(full_pb, params_name_pb, params_target_pb, name, attr, val)
        
        # Update XSPEC models
        mymodel_mw.set_values_all(full_mw, lowers_mw, uppers_mw, keyfits_mw)
        mymodel_mw.set_frozen(keyfits_mw)
        mymodel_pb.set_values_all(full_pb, lowers_pb, uppers_pb, keyfits_pb)
        mymodel_pb.set_frozen(keyfits_pb)
        
        # Polish with XSPEC optimizers
        Fit.method = "leven"
        Fit.criticalDelta = 1e-4
        Fit.nIterations = 10000
        if FitInstance():
            Fit.method = "migrad"
            Fit.criticalDelta = 1e-5
            Fit.nIterations = 10000
            FitInstance()
            
            TS_out = Fit.statistic
            nfit_out = Fit.nVarPars
            dof_out = Fit.dof
            pvalue_out = Fit.nullhyp
            full_mw = mymodel_mw.get_values()
            full_pb = mymodel_pb.get_values()
            
            print(f"After XSPEC polish: TS = {TS_out:.4f}")
    
    else:
        raise ValueError(f"Unknown TypeFit '{TypeFit}'. Must be 'XSpec' or 'Diffev'.")
    
    return (full_mw, full_pb, TS_out, pvalue_out, dof_out, nfit_out)

try:
    params_astro_mw, params_astro_pb, TS_astro, pvalue_astro, dof_astro, nfit_astro = read_fit_astro(outdir)
    print("Successfully loaded initial fit from h5 file")
    print(f"TS_astro from h5: {TS_astro}")
    use_initial_fit = True
except Exception as e:
    print(f"Could not load fit_results_Astro.h5: {e}")
    print("Will perform full fit with loop over photon index")
    use_initial_fit = False

Fit.statMethod = "cstat"  # fitting statistics = Cash-statistics

# Use best-fit values from h5 as initial parameters
if use_initial_fit:
    print("="*70)
    print("USING INITIAL FIT FROM H5 FILE")
    print("="*70)

# Set MW model to best-fit values
mymodel_mw.set_values_all(params_astro_mw, lowers, uppers, keyfits)
mymodel_mw.set_frozen(keyfits)

# Set PB model to best-fit values
mymodel_pb.set_values_all(params_astro_pb, lowers_pb, uppers_pb, keyfits_pb)
mymodel_pb.set_frozen(keyfits_pb)

Xset.chatter = 10
AllModels.show()
Xset.chatter = 1

print("Initial parameters set from h5 file")
print("Ready for refinement fits")

TypeFit = 'Diffev'
#TypeFit = 'XSpec'



params_astro_mw, params_astro_pb,TS_astro,pvalue_astro,dof_astro,nfit_astro = doFit(mymodel_mw, mymodel_pb,
    params, params_pb,
    params_name, params_target,
    params_name_pb, params_target_pb,
    lowers, uppers, keyfits,
    lowers_pb, uppers_pb, keyfits_pb,
    TypeFit)


# all array MW+PB
params_astro = np.concatenate([params_astro_mw, params_astro_pb])

phoindex_bestfit = (return_param_valueparams_astro_pb, params_name_pb, params_target_pb, "powerlaw", "PhoIndex")

print("Test Statistics ",TS_astro," testStatistics ",TS_astro," dof ",dof_astro," nVarPars ",nfit_astro)
print("pvalue ",pvalue_astro)
print("phoindex_bestfit ",phoindex_bestfit)
print("params_astro_mw ",params_astro_mw)
print("params_astro_pb ",params_astro_pb)

#print(params_name_pb)
#print(params_target_pb)

Xset.chatter = 10
AllModels.show()
Xset.chatter = 1

# %%
print("****************")
printparam(params_astro_mw, lowers, uppers , params_name, params_target, keyfits)
print("****************")
print("****************")
printparam(params_astro_pb, lowers_pb, uppers_pb , params_name_pb, params_target_pb, keyfits_pb)
print("****************")
PrintModelXspec()
print("****************")
print("****************")


#filename = os.path.join(outdir, f"fit_results_E_{Eline:.3f}.h5")
filename = os.path.join(outdir, f"fit_results_Astro.h5")
print("filename hp5 ",filename)
with h5py.File(filename, "w") as f:
    grp = f.create_group(f"Fit_Astro")

    # === Global attributes ===
    grp.attrs["Date"] = datetime.now().isoformat()

    # === Astro fit results ===
    grp_astro = grp.create_group("astro")
    grp_astro.create_dataset("TS_astro", data=TS_astro)
    grp_astro.create_dataset("nbins", data=len(venergies[0]) * len(vfilespectrum))
    grp_astro.create_dataset("nfit_astro", data=nfit_astro)
    grp_astro.create_dataset("PhoIndex", data=phoindex_bestfit)
    grp_astro.create_dataset("p-value-nullhyp", data=pvalue_astro)
    grp_astro.create_dataset("params_astro", data=np.asarray(params_astro, dtype=np.float64))
    grp_astro.create_dataset("params_astro_mw", data=np.asarray(params_astro_mw, dtype=np.float64))
    grp_astro.create_dataset("params_astro_pb", data=np.asarray(params_astro_pb, dtype=np.float64))
    grp_astro.create_dataset("dof_astro", data=dof_astro)


