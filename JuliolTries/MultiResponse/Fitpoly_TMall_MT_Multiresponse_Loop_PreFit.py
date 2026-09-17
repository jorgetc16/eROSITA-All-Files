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
# # GOAL: to fit spectra in a window around the line. Use XSpec to fit
# 
# I only use 1 power-law+ gaussians. The goal is to check if this works for E>2 keV

# %% [markdown]
# Here I load the data 

# %%
AllData.clear()
path="/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/"
#filespectrum=path+"srctoolout_120_SourceSpec_00001.fits"
#fileRMF=path+"srctoolout_120_RMF_00001.fits"
#fileARF=path+"srctoolout_120_ARF_00001.fits"
filespectrumTM1=path+"srctoolout_120_SourceSpec_00001.fits"
fileRMFTM1=path+"srctoolout_120_RMF_00001.fits"
fileARFTM1=path+"srctoolout_120_ARF_00001.fits"
#
filespectrumTM2=path+"srctoolout_220_SourceSpec_00001.fits"
fileRMFTM2=path+"srctoolout_220_RMF_00001.fits"
fileARFTM2=path+"srctoolout_220_ARF_00001.fits"
#
filespectrumTM3=path+"srctoolout_320_SourceSpec_00001.fits"
fileRMFTM3=path+"srctoolout_320_RMF_00001.fits"
fileARFTM3=path+"srctoolout_320_ARF_00001.fits"
#
filespectrumTM4=path+"srctoolout_420_SourceSpec_00001.fits"
fileRMFTM4=path+"srctoolout_420_RMF_00001.fits"
fileARFTM4=path+"srctoolout_420_ARF_00001.fits"
#
filespectrumTM5=path+"srctoolout_520_SourceSpec_00001.fits"
fileRMFTM5=path+"srctoolout_520_RMF_00001.fits"
fileARFTM5=path+"srctoolout_520_ARF_00001.fits"
#
filespectrumTM6=path+"srctoolout_620_SourceSpec_00001.fits"
fileRMFTM6=path+"srctoolout_620_RMF_00001.fits"
fileARFTM6=path+"srctoolout_620_ARF_00001.fits"
#
filespectrumTM7=path+"srctoolout_720_SourceSpec_00001.fits"
fileRMFTM7=path+"srctoolout_720_RMF_00001.fits"
fileARFTM7=path+"srctoolout_720_ARF_00001.fits"
#
#Put here the TM you want to use 
#
telescope = sys.argv[3] 

# Define telescope file mappings
telescope_files = {
    "TM1": (filespectrumTM1, fileRMFTM1, fileARFTM1),
    "TM2": (filespectrumTM2, fileRMFTM2, fileARFTM2),
    "TM3": (filespectrumTM3, fileRMFTM3, fileARFTM3),
    "TM4": (filespectrumTM4, fileRMFTM4, fileARFTM4),
    "TM5": (filespectrumTM5, fileRMFTM5, fileARFTM5),
    "TM6": (filespectrumTM6, fileRMFTM6, fileARFTM6),
    "TM7": (filespectrumTM7, fileRMFTM7, fileARFTM7),
    "ALL": ([filespectrumTM1, filespectrumTM2, filespectrumTM3, filespectrumTM4, filespectrumTM5, filespectrumTM6, filespectrumTM7],
            [fileRMFTM1, fileRMFTM2, fileRMFTM3, fileRMFTM4, fileRMFTM5, fileRMFTM6, fileRMFTM7],
            [fileARFTM1, fileARFTM2, fileARFTM3, fileARFTM4, fileARFTM5, fileARFTM6, fileARFTM7]),
    "ALL_CLEAN": ([filespectrumTM1, filespectrumTM2, filespectrumTM3, filespectrumTM4, filespectrumTM6],
    [fileRMFTM1, fileRMFTM2, fileRMFTM3, fileRMFTM4, fileRMFTM6],
    [fileARFTM1, fileARFTM2, fileARFTM3, fileARFTM4, fileARFTM6])
}

# Set the file arrays based on telescope selection
if telescope in telescope_files:
    if telescope == "ALL":
        vfilespectrum, vfileRMF, vfileARF = telescope_files[telescope]
    else:
        spectrum_file, rmf_file, arf_file = telescope_files[telescope]
        vfilespectrum = [spectrum_file]
        vfileRMF = [rmf_file]
        vfileARF = [arf_file]
    
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
AllData.clear()
for plot_grp in range (0,len(vfilespectrum)):
    #print("plot_grp ",plot_grp)
    Spectrum(vfilespectrum[plot_grp])
    AllData(plot_grp+1).multiresponse[0] = vfileRMF[plot_grp]
    AllData(plot_grp+1).multiresponse[0].arf = vfileARF[plot_grp]
    AllData(plot_grp+1).multiresponse[1] = vfileRMF[plot_grp] # RMF only for src2



# %%
data = np.loadtxt("/home/jortecal/GitHub/eRosita/3MLFits/Energy_width.txt",delimiter=",")
vE = data[:,0]
vFWHM = data[:,1]*1e-3 #keV
vsigma = vFWHM/(2.355)

# %% [markdown]
# Fix the energy of the DM line and select data

# %%
# I do not consider eROSITA data outside this range
Eallmin = 0.3
Eallmax = 9.

Eline = sys.argv[1] #keV
Eline = float(Eline)

sigma = np.interp(Eline,vE,vsigma)
print("Eline ",Eline," sigma ",sigma)
Emin= Eline - 5*sigma
Emax= Eline + 5*sigma
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
print("sigma ",sigma)
print("DeltaE ",vedeltas[0]*2)
print("Sigma/DeltaE ",sigma/(vedeltas[0]*2))

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
EminL = Eline - 7*sigma
EmaxL = Eline + 7*sigma
if(EminL<Eallmin):
    EminL = Eallmin
if(EmaxL>Eallmax):
    EmaxL = Eallmax

data = np.loadtxt("/home/jortecal/GitHub/eRosita/3MLFits/MarcoFitsMay/IBLines.txt",skiprows=0)
IBenergy = data[:,0]
IBenergymin = data[:,1]
IBenergymax = data[:,2]
data = np.loadtxt("/home/jortecal/GitHub/eRosita/3MLFits/MarcoFitsMay/AstroLines.txt",skiprows=0)
Aenergy = data[:,0]
Aenergymin = data[:,1]
Aenergymax = data[:,2]
#print(Aenergy," ",IBenergy)
#print(len(Aenergy)," ",len(IBenergy))
IBidx = np.where( (IBenergy > EminL ) & (IBenergy < EmaxL ) )
Aidx = np.where( (Aenergy > EminL ) & (Aenergy < EmaxL ) )
IBLine = IBenergy[IBidx]
IBLinemin = IBenergymin[IBidx]
IBLinemax = IBenergymax[IBidx]


ALine = Aenergy[Aidx]
#ALinemin = Aenergymin[Aidx]
#ALinemax = Aenergymax[Aidx]
# SET HERE VARIATION OF ASTRO LINE
AstroEl = 0.005;
ALinemin = ALine - AstroEl
ALinemax = ALine + AstroEl


IBn = len(IBLine)
An = len(ALine)
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
print("Total lines ",IBAn," IB ",IBn," A ",An)
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
        self._param_fast_setters = [
            self._make_fast_setter(getattr(getattr(model, name), attr))
            for name, attr in zip(params_name, params_target)
        ]

        # Also keep full parameter objects for set_values_all
        self._param_objects = [
            getattr(getattr(model, name), attr)
            for name, attr in zip(params_name, params_target)
        ]

    def _make_fast_setter(self, param_obj):
        def setter(value, param_obj=param_obj):
            #param_obj.values[0] = value
            param_obj.values = value
        return setter

    #set value of parameters
    def set_values(self, params):
        for setter, value in zip(self._param_fast_setters, params):
            setter(float(value))  # direct, fast assignment

    #set value of parameters including upper/lower limits
    def set_values_all(self, params, lowers, uppers):
        for i, param in enumerate(self._param_objects):
            a = float(params[i])
            b = float(lowers[i])
            c = float(uppers[i])
            param.values = [a, 1e-2 * a, b, b, c, c]

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
    res = mymodel_fit.evaluate_folded(params_inside)
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
# See nAPEC_Alll.ipynb
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


normline_start= normlDMhdata2*0.01 #*0.01
normline_lower = 0.0
normline_upper = normlDMhdata2*100
sigmaDM = Eline*220/3e5 #1e-4
sigma_astrolines = 1e-4
#
#
#power law
startidx = 3 #1.4
upperidx = 6.0
loweridx = - 3.0
#
startpl = 0.1*normlDMhdata *np.pow(Eline,startidx)
# startpl = 0.05
upperpl = startpl*1000.0
lowerpl = 0.0

#
# I always define DM to be the first gaussian. I set parameters to False (no in the fit)
model_dic = {
    
    "gaussian": {
        "gaussian": {
            "norm": pdic(normline_start, normline_lower, normline_upper, False), # (value, lower, upper, key)
            "LineE" : pdic(Eline, Eline, Eline, False),
            "Sigma" : pdic(sigmaDM, sigmaDM, sigmaDM, False)
            }
        }
    }
# gaussian_components = [f"gaussian" for i in range(len(IBALine))]
# gaussian_sum = " + ".join(gaussian_components)
# if len(IBALine)>0:
#     model_dic["_structure"] = f"gaussian+ {gaussian_sum}" 
# else:
model_dic["_structure"] = f"gaussian"
    
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
        "gaussian": {}  # This is the DM line
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
        "LineE": pdic(IBALine[i], IBALinemin[i], IBALinemax[i], True),
        "Sigma": pdic(sigma_astrolines, sigma_astrolines, sigma_astrolines, False)
    }




# %%
print(model_dic)
print("******")
print(model_dic_pb)

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
mymodel_mw.set_values_all(params, lowers, uppers)
# set the free parameters in Xspec
mymodel_mw.set_frozen(keyfits)
#
mymodel_pb.set_values_all(params_pb, lowers_pb, uppers_pb)
mymodel_pb.set_frozen(keyfits_pb)

# %%
# %%
# Xset.chatter = 10
# AllModels.show()
# Xset.closeLog()
# Xset.chatter = 0


# %%
printparam(params, lowers, uppers , params_name, params_target, keyfits)
print("****************")
printparam(params_pb, lowers_pb, uppers_pb , params_name_pb, params_target_pb, keyfits_pb)



#PRE-FIT: DOES NOT WORK IN SOME CASES
# if len(IBALine) > 0:
#     for i in range(len(IBALine)):
#         if i == 0:
#             namegauss_a = "gaussian"
#         else:
            
#             namegauss_a ="gaussian_"+str(start_index_astrol+i)
#         # print(namegauss_a)
#         #set astrolines to false and zero
#         change_keyfits(params_name_pb, params_target_pb, keyfits_pb, namegauss_a, "LineE", False)
#         change_keyfits(params_name_pb, params_target_pb, keyfits_pb, namegauss_a, "norm", False)
#         change_param_value(params_pb, params_name_pb, params_target_pb, namegauss_a, "norm", 0.0)
    
#     # change_keyfits(params_name_pb, params_target_pb, keyfits_pb, "powerlaw", "norm", False)
#     print("\n ************** Pre-fit parameters **************")
#     printparam(params_pb, lowers_pb, uppers_pb , params_name_pb, params_target_pb, keyfits_pb)
#     # change_keyfits(params_name_pb, params_target_pb, keyfits_pb, "gaussian", "norm", False)
#     # change_param_value(params_pb, params_name_pb, params_target_pb, "gaussian", "norm", 0.0)
#     # since you have changed norm and key(True/False), update XSpec
#     mymodel_pb.set_values_all(params_pb, lowers_pb, uppers_pb)
#     mymodel_pb.set_frozen(keyfits_pb)
#     # Select the parameters to Fit. You can check that DM is not present
#     params_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params_pb, lowers_pb, uppers_pb, params_name_pb, params_target_pb, keyfits_pb)

#     Fit.perform()
#     Fit.perform()
#     Fit.perform()

#     # Xset.chatter = 10
#     # AllModels.show()
#     # Xset.chatter = 1
    
#     params_initial = mymodel_pb.get_values()
#     params_name_initial = mymodel_pb.params_name
#     params_target_initial = mymodel_pb.params_target
#     phoindex_initial = return_param_value(params_initial, params_name_initial, params_target_initial, "powerlaw", "PhoIndex")
#     norm_initial = return_param_value(params_initial, params_name_initial, params_target_initial, "powerlaw", "norm")
#     print('\n Initial PhoIndex:', phoindex_initial)
#     for i in range(len(IBALine)):
#         if i == 0:
#             namegauss_a = "gaussian"
#         else:
            
#             namegauss_a ="gaussian_"+str(start_index_astrol+i)
#         # print(namegauss_a)
#         #set astrolines to false and zero
#         change_keyfits(params_name_pb, params_target_pb, keyfits_pb, namegauss_a, "LineE", True)
#         change_keyfits(params_name_pb, params_target_pb, keyfits_pb, namegauss_a, "norm", True)
#         change_param_value(params_pb, params_name_pb, params_target_pb, namegauss_a, "norm", normline_start)

#     change_param_value(params_pb, params_name_pb, params_target_pb, "powerlaw", "PhoIndex", phoindex_initial)
#     change_param_value(params_pb, params_name_pb, params_target_pb, "powerlaw", "norm", norm_initial)
#     # change_keyfits(params_name_pb, params_target_pb, keyfits_pb, "powerlaw", "norm", True)


#     mymodel_pb.set_values_all(params_pb, lowers_pb, uppers_pb)
#     mymodel_pb.set_frozen(keyfits_pb)

#     print("\n ************** Regular-fit parameters **************")
#     printparam(params_pb, lowers_pb, uppers_pb , params_name_pb, params_target_pb, keyfits_pb)

Fit.statMethod = "cstat"  # fitting statistics = Cash-statistics

# LOOP TO FIT WITH DIFFERENT INITAL PHOINDEX TO FIND BETTER FIT
TS= 1e6
for phoindex in np.arange(-3.0, 5, 1):
    change_param_value(params_pb, params_name_pb, params_target_pb, "powerlaw", "PhoIndex", phoindex)
    change_param_value(params_pb, params_name_pb, params_target_pb, "powerlaw", "norm", startpl)
    mymodel_pb.set_values_all(params_pb, lowers_pb, uppers_pb)

    Fit.method = "simplex"
    Fit.nIterations = 10000
    Fit.criticalDelta = 1e-3
    Fit.perform()

    Fit.method = "leven"
    Fit.nIterations = 10000
    Fit.criticalDelta = 1e-4
    Fit.perform()

    Fit.method = "migrad"
    Fit.nIterations = 10000
    Fit.criticalDelta = 1e-5
    Fit.perform()
    params_initial = mymodel_pb.get_values()
    params_name_initial = mymodel_pb.params_name
    params_target_initial = mymodel_pb.params_target
    if Fit.statistic < TS:
        TS =  Fit.statistic
        phoindex_initial = return_param_value(params_initial, params_name_initial, params_target_initial, "powerlaw", "PhoIndex")
        norm_initial = return_param_value(params_initial, params_name_initial, params_target_initial, "powerlaw", "norm")

change_param_value(params_pb, params_name_pb, params_target_pb, "powerlaw", "PhoIndex", phoindex_initial)
change_param_value(params_pb, params_name_pb, params_target_pb, "powerlaw", "norm", norm_initial)

mymodel_pb.set_values_all(params_pb, lowers_pb, uppers_pb)


print("**************")
# %%

# Fit.perform()
# Fit.perform()
# Fit.perform()


# %%
# If not already specified in initial dictionarys set DM to False and norm to 0
change_keyfits(params_name, params_target, keyfits, "gaussian", "norm", False)
change_param_value(params, params_name, params_target, "gaussian", "norm", 0.0)

# since you have changed norm and key(True/False), update XSpec
mymodel_mw.set_values_all(params, lowers, uppers)
mymodel_mw.set_frozen(keyfits)


# Select the parameters to Fit. You can check that DM is not present
params_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params, lowers, uppers, params_name, params_target, keyfits)


# %%
# Xset.chatter = 10
# AllModels.show()
# Xset.closeLog()
# Xset.chatter = 1

# %% [markdown]
# Perform Astrofit

# %%
#params_astro,TS_astro, pvalue_astro, dof_astro, nfit_astro = doFit(fullmodel_xp, params_fit, bounds_fit, params_name_fit, params_target_fit,TypeFit)

#print("Test Statistics ",TS_astro," testStatistics ",TS_astro," dof ",dof_astro," nVarPars ",nfit_astro)
#print("pvalue ",pvalue_astro)
#printparam(params_astro, bounds_fit[:,0],bounds_fit[:,1], params_name_fit, params_target_fit, keyfits_fit)


Xset.chatter = 10
AllModels.show()
Xset.chatter = 1



Fit.method = "simplex"
Fit.nIterations = 10000
Fit.criticalDelta = 1e-3
Fit.perform()
Fit.method = "leven"
Fit.nIterations = 10000
Fit.criticalDelta = 1e-4
Fit.perform()

Fit.method = "migrad"
Fit.nIterations = 10000
Fit.criticalDelta = 1e-5
Fit.perform()

# Xset.chatter = 10
# AllModels.show()
# Xset.chatter = 1

TS_astro = Fit.statistic
nfit_astro = Fit.nVarPars
dof_astro = Fit.dof
pvalue_astro = Fit.nullhyp
params_astro = mymodel_pb.get_values()
params_name_astro = mymodel_pb.params_name
params_target_astro = mymodel_pb.params_target
phoindex_bestfit = return_param_value(params_astro, params_name_astro, params_target_astro, "powerlaw", "PhoIndex")


print("Test Statistics ",TS_astro," testStatistics ",TS_astro," dof ",dof_astro," nVarPars ",nfit_astro)
print("pvalue ",pvalue_astro)


# %%


# %%

# NOW FIT ASTRO + DM
# Change DM norm and T/F
change_keyfits(params_name, params_target, keyfits, "gaussian", "norm", True)
change_param_value(params, params_name, params_target, "gaussian", "norm", normline_start)
#
# set the other parameters to their best-fit value of previous fit
for val, name, attr in zip(params_astro, params_name_fit, params_target_fit):
    change_param_value(params, params_name, params_target, name, attr, val)

# since you have changed norm and key(True/False), update XSpec
mymodel_mw.set_values_all(params, lowers, uppers)
mymodel_mw.set_frozen(keyfits)
#

# Select the parameters to Fit. You can check that DM is present
params_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params, lowers, uppers, params_name, params_target, keyfits)


# Xset.chatter = 10
# AllModels.show()
# Xset.closeLog()
# Xset.chatter = 1



Fit.method = "simplex"
Fit.nIterations = 10000
Fit.criticalDelta = 1e-3
Fit.perform()

Fit.method = "leven"
Fit.nIterations = 10000
Fit.criticalDelta = 1e-4
Fit.perform()

Fit.method = "migrad"
Fit.nIterations = 10000
Fit.criticalDelta = 1e-5
Fit.perform()

print("**************** **************** ****************")
# Xset.chatter = 10
# AllModels.show()
# Xset.closeLog()
# Xset.chatter = 1



TS_DM = Fit.statistic
nfit_dm = Fit.nVarPars
dof_DM = Fit.dof
pvalue_DM = Fit.nullhyp
params_DM = mymodel_mw.get_values()


DMbf = return_param_value(params_DM , params_name_fit, params_target_fit, "gaussian", "norm")
print("DM bf ",DMbf)
print("DeltaTS ",TS_astro-TS_DM)

print("Test Statistics DM ",TS_DM," dof ",dof_DM," nVarPars ",nfit_dm)
print("pvalue ",pvalue_DM)

params_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params, lowers, uppers, params_name, params_target, keyfits)


# #### Approximate bound: PASS ARRAYS _all referring to best fit model
nstepsDM = 100 # can be a bit larger since it just a matter of evaluations
bound_approx = ApproximateBound(m_mw, DMbf, normlDMhdata2, nstepsDM, TS_DM, params_fit, params_name_fit, params_target_fit)
print("Approximate bound ",bound_approx)

for val, name, attr in zip(params_DM, params_name_fit, params_target_fit):
    change_param_value(params, params_name, params_target, name, attr, val)

# Change DM  T/F setting to F
change_keyfits(params_name, params_target, keyfits, "gaussian", "norm", False)
#
# since you have changed norm and key(True/False), update XSpec
mymodel_mw.set_values_all(params, lowers, uppers)
mymodel_mw.set_frozen(keyfits)
#

nstepsDM = 100
start = bound_approx/5e5
end = bound_approx*100
profile_range = np.logspace(np.log10(start), np.log10(end), num = nstepsDM)
TS_vals = np.zeros_like(profile_range)
best_vals_per_A = [None] * (len(profile_range)+1)
for i, A in enumerate(profile_range):
    #set DM norm
    m_mw.gaussian.norm = A
    if i > 0 and best_vals_per_A[i-1] is not None:
        mymodel_pb.set_values(best_vals_per_A[i-1])
    else:
        mymodel_pb.set_values(params_astro)  # background-only best fit

    # if i % 3 == 0:
    #     # mymodel_pb.set_values(params_astro)
    #     # Fit.perform()
    #     Fit.method = "simplex"
    #     Fit.nIterations = 10000
    #     Fit.criticalDelta = 1e-3
    #     Fit.perform()
    #     Fit.method = "migrad"
    #     Fit.criticalDelta = 1e-4
    #     Fit.perform()
    

    
    pparams_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params, lowers, uppers, params_name, params_target, keyfits)
    # Fit.method = "simplex"
    # Fit.nIterations = 800
    # Fit.criticalDelta = 0.5

    Fit.method = "leven"
    Fit.nIterations = 10000
    Fit.criticalDelta = 1e-4
    Fit.perform()

    Fit.method = "simplex"
    Fit.nIterations = 10000
    Fit.criticalDelta = 1e-3
    Fit.perform()

    Fit.method = "migrad"
    Fit.nIterations = 10000
    Fit.criticalDelta = 1e-5
    Fit.perform()

    best_vals_per_A[i] = mymodel_pb.get_values()


    TS_inside = Fit.statistic
    TS_vals[i] = TS_inside
    # update params to new best fit if you want
    # otherwise you are always giving the parameters of bf model with DM
    #params_fit = params_inside
    print(i, A, TS_vals[i], TS_vals[i] - TS_DM)
    if  i >= 2 and TS_vals[i] - TS_DM > 50:
        profile_range = profile_range[:i+1]
        TS_vals = TS_vals[:i+1]
        break

deltaTS_vals = TS_vals - TS_DM
#
A_95 = BoundProfiling(profile_range, deltaTS_vals)
#
print("bound_approx", bound_approx)
print("A_95 ",A_95)
#
#
# Check how really close you are to DeltaTS = 2.71
#
m_mw.gaussian.norm = A_95
pparams_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params, lowers, uppers, params_name, params_target, keyfits)



Fit.method = "simplex"
Fit.nIterations = 10000
Fit.criticalDelta = 0.01
Fit.perform()
Fit.method = "leven"
Fit.nIterations = 10000
Fit.criticalDelta = 1e-4
Fit.perform()

Fit.method = "migrad"
Fit.nIterations = 10000
Fit.criticalDelta = 1e-5
Fit.perform()



TS_inside = Fit.statistic
print("Initial A_95 ", A_95, " DeltaTS ", TS_inside-TS_DM)
#
# Refine a bit more
#
factor = 25  # Range: A_95_refined / 2 to A_95_refined * 2
profile_range_fine = np.logspace(
    np.log10(A_95 / factor),
    np.log10(A_95 * factor),
    num=nstepsDM
)
TS_vals_fine = np.zeros_like(profile_range_fine)


best_vals_per_A = [None] * (len(profile_range_fine)+1)
for i, A in enumerate(profile_range_fine):
    m_mw.gaussian.norm = A
    if i > 0 and best_vals_per_A[i-1] is not None:
        mymodel_pb.set_values(best_vals_per_A[i-1])
    else:
        mymodel_pb.set_values(params_astro)  # background-only best fit
    # if i % 3 == 0:
    #     # mymodel_pb.set_values(params_astro)
    #     # Fit.perform()
    #     Fit.method = "simplex"
    #     Fit.nIterations = 10000
    #     Fit.criticalDelta = 0.01
    #     Fit.perform()
    #     Fit.method = "migrad"
    #     Fit.criticalDelta = 1e-4
    #     Fit.perform()
  

    # if i%3 == 0:
    #     TS= 1e6
    #     for phoindex in np.arange(-3.0, 5, 1):
    #         change_param_value(params_pb, params_name_pb, params_target_pb, "powerlaw", "PhoIndex", phoindex)
    #         change_param_value(params_pb, params_name_pb, params_target_pb, "powerlaw", "norm", startpl)
    #         mymodel_pb.set_values_all(params_pb, lowers_pb, uppers_pb)

    #         Fit.method = "leven"
    #         Fit.nIterations = 10000
    #         Fit.criticalDelta = 1e-4
    #         Fit.perform()

    #         Fit.method = "simplex"
    #         Fit.nIterations = 10000
    #         Fit.criticalDelta = 1e-3
    #         Fit.perform()

    #         Fit.method = "migrad"
    #         Fit.nIterations = 10000
    #         Fit.criticalDelta = 1e-5
    #         Fit.perform()
    #         params_initial = mymodel_pb.get_values()
    #         params_name_initial = mymodel_pb.params_name
    #         params_target_initial = mymodel_pb.params_target
    #         if Fit.statistic < TS:
    #             TS =  Fit.statistic
    #             phoindex_initial = return_param_value(params_initial, params_name_initial, params_target_initial, "powerlaw", "PhoIndex")
    #             norm_initial = return_param_value(params_initial, params_name_initial, params_target_initial, "powerlaw", "norm")

    #     change_param_value(params_pb, params_name_pb, params_target_pb, "powerlaw", "PhoIndex", phoindex_initial)
    #     change_param_value(params_pb, params_name_pb, params_target_pb, "powerlaw", "norm", norm_initial)

    #     mymodel_pb.set_values_all(params_pb, lowers_pb, uppers_pb)


    # Select the parameters to Fit. You can check that DM is present
    params_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params, lowers, uppers, params_name, params_target, keyfits)
    # Fit.method = "simplex"
    # Fit.nIterations = 400
    # Fit.criticalDelta = 0.3
    
    Fit.method = "leven"
    Fit.nIterations = 10000
    Fit.criticalDelta = 1e-4
    Fit.perform()

    Fit.method = "simplex"
    Fit.nIterations = 10000
    Fit.criticalDelta = 0.01
    Fit.perform()

    Fit.method = "migrad"
    Fit.nIterations = 10000
    Fit.criticalDelta = 1e-5
    Fit.perform()

    best_vals_per_A[i] = mymodel_pb.get_values()

    TS_inside = Fit.statistic
    TS_vals_fine[i] = TS_inside
    # params_fit = params_inside
    print("[REFINED]", i, A, TS_vals_fine[i], TS_vals_fine[i] - TS_DM)
    if  i >= 2 and TS_vals_fine[i] - TS_DM > 50:
        profile_range_fine = profile_range_fine[:i+1]
        TS_vals_fine = TS_vals_fine[:i+1]
        break

# Combine and sort
profile_range = np.concatenate([profile_range, profile_range_fine])
TS_vals = np.concatenate([TS_vals, TS_vals_fine])
sort_idx = np.argsort(profile_range)
profile_range = profile_range[sort_idx]
TS_vals = TS_vals[sort_idx]
deltaTS_vals = TS_vals - TS_DM
# #
A_95 = BoundProfiling(profile_range, deltaTS_vals)
#
m_mw.gaussian.norm = A_95
params_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params, lowers, uppers, params_name, params_target, keyfits)



Fit.method = "simplex"
Fit.nIterations = 10000
Fit.criticalDelta = 0.01
Fit.perform()
Fit.method = "leven"
Fit.nIterations = 10000
Fit.criticalDelta = 1e-4
Fit.perform()

Fit.method = "migrad"
Fit.nIterations = 10000
Fit.criticalDelta = 1e-5
Fit.perform()



TS_inside = Fit.statistic
print("Final A_95 ", A_95, " DeltaTS ", TS_inside-TS_DM)


outdir = sys.argv[2] # output directory
os.makedirs(outdir, exist_ok=True)
filename = os.path.join(outdir, f"fit_results_E_{Eline:.3f}.h5")
print("filename hp5 ",filename)
with h5py.File(filename, "w") as f:
    grp = f.create_group(f"E_{Eline:.3f}")

    # === Global attributes ===
    grp.attrs["sigma"] = sigma
    grp.attrs["Emin"] = Emin
    grp.attrs["Emax"] = Emax
    grp.attrs["Number_of_IB_lines"] = IBn
    grp.attrs["A_95"] = A_95
    grp.attrs["A_95_approx"] = bound_approx
    grp.attrs["normlDMhdata2"] = normlDMhdata2
    grp.attrs["Date"] = datetime.now().isoformat()

    # === Astro fit results ===
    grp_astro = grp.create_group("astro")
    grp_astro.create_dataset("TS_astro", data=TS_astro)
    grp_astro.create_dataset("nbins", data=len(venergies[0]) * len(vfilespectrum))
    grp_astro.create_dataset("nfit_astro", data=nfit_astro)
    grp_astro.create_dataset("PhoIndex", data=phoindex_bestfit)
    grp_astro.create_dataset("p-value-nullhyp", data=pvalue_astro)

    # === DM fit results ===
    grp_dm = grp.create_group("dm")
    grp_dm.create_dataset("TS_all", data=TS_DM)
    grp_dm.create_dataset("deltaTS", data=TS_astro - TS_DM)
    grp_dm.create_dataset("nbins", data=len(venergies[0]) * len(vfilespectrum))
    grp_dm.create_dataset("nfit_dm", data=nfit_dm)
    grp_dm.create_dataset("params", data=np.asarray(params_DM, dtype=np.float64))
    # grp_dm.create_dataset("param_names", data=params_name_fit_DM.astype("S"))
    # grp_dm.create_dataset("param_targets", data=params_target_fit_DM.astype("S"))
    grp_dm.create_dataset("valid", data=1 if bound_approx > 0 else 0)

        # === Profile likelihood ===
    grp_profile = grp.create_group("profile")
    grp_profile.create_dataset("A_values", data=profile_range)
    grp_profile.create_dataset("TS_values", data=TS_vals)
