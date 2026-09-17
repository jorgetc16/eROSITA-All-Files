# %%
import os
from xspec import *
import xspec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from decimal import Decimal
from scipy import optimize, stats
from iminuit import Minuit
from scipy.stats import chi2 as chi2_scipy
import sys 
import h5py
from datetime import datetime

##### 5 input parameters:
# 1) energy of the DM line in keV
# 2) Type of fitting string: 1=Minuit, 2=Xspec, 3= DiffeV
# 3) Number of points in profile
# 4) Output directory
# 5) Type of astro model: 1 = Powerlaw0+Powerlaw1+Powerlaw2, 2 = bknpowerlaw, 3=...

Xset.allowPrompting = False # keeps pyxspec from hanging, waiting for a response to a prompt


# # GOAL: to fit spectra in a window around the line. Use XSpec to fit


# Here I load the data 

AllData.clear()
path="/Users/marcotaoso/Documents/2024/eROSITA/FromJorge/LMC_test_3deg/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/"
#filespectrumTM1=path+"srctoolout_120_SourceSpec_00001.fits"
filespectrumTM1=path+"Rebining_2/srctoolout_120_SourceSpec_00001_2.grp"
fileRMFTM1=path+"srctoolout_120_RMF_00001.fits"
fileARFTM1=path+"srctoolout_120_ARF_00001.fits"
#
#filespectrumTM2=path+"srctoolout_220_SourceSpec_00001.fits"
filespectrumTM2=path+"Rebining_2/srctoolout_220_SourceSpec_00001_2.grp"
fileRMFTM2=path+"srctoolout_220_RMF_00001.fits"
fileARFTM2=path+"srctoolout_220_ARF_00001.fits"
#
#filespectrumTM3=path+"srctoolout_320_SourceSpec_00001.fits"
filespectrumTM3=path+"Rebining_2/srctoolout_320_SourceSpec_00001_2.grp"
fileRMFTM3=path+"srctoolout_320_RMF_00001.fits"
fileARFTM3=path+"srctoolout_320_ARF_00001.fits"
#
#filespectrumTM4=path+"srctoolout_420_SourceSpec_00001.fits"
filespectrumTM4=path+"Rebining_2/srctoolout_420_SourceSpec_00001_2.grp"
fileRMFTM4=path+"srctoolout_420_RMF_00001.fits"
fileARFTM4=path+"srctoolout_420_ARF_00001.fits"
#
#filespectrumTM5=path+"srctoolout_520_SourceSpec_00001.fits"
filespectrumTM5=path+"Rebining_2/srctoolout_520_SourceSpec_00001_2.grp"
fileRMFTM5=path+"srctoolout_520_RMF_00001.fits"
fileARFTM5=path+"srctoolout_520_ARF_00001.fits"
#
#filespectrumTM6=path+"srctoolout_620_SourceSpec_00001.fits"
filespectrumTM6=path+"Rebining_2/srctoolout_620_SourceSpec_00001_2.grp"
fileRMFTM6=path+"srctoolout_620_RMF_00001.fits"
fileARFTM6=path+"srctoolout_620_ARF_00001.fits"
#
#filespectrumTM7=path+"srctoolout_720_SourceSpec_00001.fits"
filespectrumTM7=path+"Rebining_2/srctoolout_720_SourceSpec_00001_2.grp"
fileRMFTM7=path+"srctoolout_720_RMF_00001.fits"
fileARFTM7=path+"srctoolout_720_ARF_00001.fits"
#
#Put here the TM you want to use
#
vfilespectrum = [filespectrumTM2] #[filespectrumTM1,filespectrumTM2,filespectrumTM3,filespectrumTM4,filespectrumTM6]
vfileRMF = [fileRMFTM2] #[fileRMFTM1,fileRMFTM2,fileRMFTM3,fileRMFTM4,fileRMFTM6]
vfileARF = [fileARFTM2] #[fileARFTM1,fileARFTM2,fileARFTM3,fileARFTM4,fileARFTM6]
#
#LOAD DATA
#
AllData.clear()
for plot_grp in range (0,len(vfilespectrum)):
    #print("plot_grp ",plot_grp)
    Spectrum(vfilespectrum[plot_grp])
    AllData(plot_grp+1).response = vfileRMF[plot_grp]
    AllData(plot_grp+1).response.arf = vfileARF[plot_grp]

AllData.show()



# NOW LOAD the Sigma of EROSITA. At the moment I take the FWHM from https://erosita.mpe.mpg.de/edr/eROSITATechnical/calibration.html
# Computing \sigma=FWHM/2.355 I find that \sigma ranges from [33,72] eV and E/sigma = [34,139] for E ranging from 1 to 10 keV. 
# This is >> sigmav = c/v = 1360 for v=220 km/s.
# Larger than intrinsic/Doppler width of astro lines???

# %%
data = np.loadtxt("/Users/marcotaoso/Documents/2024/eROSITA/FromJorge/sigma/Energy_width.txt",delimiter=",")
vE = data[:,0]
vFWHM = data[:,1]*1e-3 #keV
vsigma = vFWHM/(2.355)


# %% [markdown]
# Fix the energy of the DM line and select data
# I do not consider eROSITA data outside this range
Eallmin = 0.3
Eallmax = 9.
#
Eline = sys.argv[1] #keV
Eline = float(Eline)
#
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
#
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

data = np.loadtxt("/Users/marcotaoso/Documents/2024/eROSITA/IB_FWC/IBLines.txt",skiprows=0)
IBenergy = data[:,0]
IBenergymin = data[:,1]
IBenergymax = data[:,2]
data = np.loadtxt("/Users/marcotaoso/Documents/2024/eROSITA/AstroLines/AstroLines.txt",skiprows=0)
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
#
ALine = Aenergy[Aidx]
#ALinemin = Aenergymin[Aidx]
#ALinemax = Aenergymax[Aidx]
# SET HERE VARIATION OF ASTRO LINE
AstroEl = 0.005;
ALinemin = ALine - AstroEl
ALinemax = ALine + AstroEl
#
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




# %%
# CLASS TO DEFINE A FUNCTION TO EVALUATE THE TS specified in fit GIVEN A xspec Model model
# with paramers value params, name (gaussian_3, powerlaw,...) params_name and target (norm, LineE,...) params_target
#
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
#
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


# Functions working on numpy arrays
#
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


# %% [markdown]
# Normalization factor (can be useful to define limits of the parameters)

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
print("normlDMhdata2 [counts/s/cm2]",normlDMhdata2)






# %% [markdown]
# Create the dictionary of the model

# %%
if sys.argv[5] == "1":
    #This is the first model to fit the background, one with 3 powerlaws, with 0, 1 and 2 as indices (actually there is a minus sign in the index by Xspec definition)
    print("Model 1")
    #
elif sys.argv[5] == "2":
    # This is the second model to fit the background, one with a broken powerlaw
    print("Model 2")
    #
elif sys.argv[5] == "3":
    #This is like the first model to fit the background but changing the sign of the indices, one with 3 powerlaws, with 0, 1 and 2 as indices (actually there is a minus sign in the index by Xspec definition)
    print("Model 3")
    #
elif sys.argv[5] == "4":
    #This is a model based on TBabsorption and APEC phenomenological models
    #
    print("Model 4")
    def pdic(value, lower, upper, key):
        return {"value": value, "lower": lower, "upper": upper, "keyfit": key}
    
    normline_start= normlDMhdata2*0.01 #*0.01
    normline_lower = 0.0
    normline_upper = normlDMhdata2*100
    sigmaDM = Eline*220/3e5 #1e-4
    sigma_astrolines = 1e-4
    #
    #from the https://github.com/xyzhang/eRASS-spectrum/blob/main/spectral_analysis_basic.ipynb)
    #The foreground Milkyway nH value is cm (https://www.swift.ac.uk/analysis/nhtot/index.php)
    startnH = 6.3e-2 #1e22 cm-2
    #uppernH = startnH*100. #*100
    #lowernH = startnH*0.01 #*0.01
    # do not allow too crazy values
    uppernH = 0.7#
    lowernH = startnH*0.1
    #
    # apec https://cxc.cfa.harvard.edu/sherpa/ahelp/xsapec.html
    startapec = 0.17 #from the https://github.com/xyzhang/eRASS-spectrum/blob/main/spectral_analysis_basic.ipynb)
    upperapec = startapec*50.0
    lowerapec = 0.0
    startT = 0.23 #keV
    upperT = 8.0 # keV
    lowerT = startT/10.0
    startAb = 1.0 # from the https://github.com/xyzhang/eRASS-spectrum/blob/main/spectral_analysis_basic.ipynb)
    upperAb = 5.0
    lowerAb = 0.01
    #
    #power law
    startidx = 1.4 #1.4
    upperidx = 5.0
    loweridx = - upperidx
    # norm of PL is #/keV/s/cm2 at 1 keV. normlDMhdata is the appropriate quantity to set norm of PL  using formula below (same units!)
    # norm * (E/1keV)^(-alpha) = normlDMhdata
    #
    #startpl = 0.1*normlDMhdata *np.pow(Eline,startidx)
    startpl = 0.05
    upperpl = startpl*1000.0
    lowerpl = 0.0
    
    # I always define DM to be the first gaussian. I set parameters to False (no in the fit)
    model_dic = {
    
        "TBabs": {
            "TBabs": {
                "nH": pdic(startnH, lowernH, uppernH, True), # (value, lower, upper, key)
                }
            },
        "apec": {
            "apec": {
                "kT": pdic(startT, lowerT, upperT, True), # (value, lower, upper, key)
                "norm" : pdic(startapec, lowerapec, upperapec, True),
                "Abundanc" : pdic(startAb, lowerAb, upperAb, False),
                "Redshift" : pdic(0, 0, 0, False)
                }
            },
        "powerlaw": {
            "powerlaw": {
                "PhoIndex": pdic(startidx, loweridx, upperidx, True), # (value, lower, upper, key)
                "norm" : pdic(startpl, lowerpl, upperpl, True)
                }
            },
        "gaussian": {
            "gaussian": {
                "norm": pdic(normline_start, normline_lower, normline_upper, False), # (value, lower, upper, key)
                "LineE" : pdic(Eline, Eline, Eline, False),
                "Sigma" : pdic(sigmaDM, sigmaDM, sigmaDM, False)
                }
            }
        }
    gaussian_components = [f"gaussian" for i in range(len(IBALine))]
    gaussian_sum = " + ".join(gaussian_components)
    if len(IBALine)>0:
        model_dic["_structure"] = f"tbabs*(apec + powerlaw + gaussian)+ {gaussian_sum}"
    else:
        model_dic["_structure"] = f"tbabs*(apec + powerlaw + gaussian)"
        
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
    #
    # Start numbering from start_index_astrol to avoid conflict with "gaussian"
    for i in range(len(IBALine)):
        g_key = f"gaussian_{start_index_astrol + i}"
        model_dic["gaussian"][g_key] = {
            "norm": pdic(normline_start, normline_lower, normline_upper, True),
            "LineE": pdic(IBALine[i], IBALinemin[i], IBALinemax[i], True),
            "Sigma": pdic(sigma_astrolines, sigma_astrolines, sigma_astrolines, False)
        }
    #
elif sys.argv[5] == "5":
    #This is a model based on TBabsorption and APEC phenomenological models
    print("Model 5")
    #
    # SAVE THE DICTIONARY
    #namefile = path_data + 'Initial' + str(iEDM) + ".npy"
    #np.save(namefile, model_dic,allow_pickle=True)



# %% [markdown]
# Define the model
#
# %%
# Extract the relevant info from the model dictionary
params, lowers, uppers, params_name, params_target, keyfits = Extract_array_dic(model_dic)
#
# Create the XSpec model
if "_structure" in model_dic:
    finalM_xp = model_dic["_structure"]
else:
    baseM_parts = []
    for top_key, sub_models in model_dic.items():
        count = len(sub_models)
        baseM_parts.extend([top_key] * count)
    finalM_xp = '+'.join(baseM_parts)

print("finalM_xp ", finalM_xp)
AllModels.clear()
fullmodel_xp = Model(finalM_xp, "mpoly", 1)
#
Xset.chatter = 0
Fit.statMethod = "cstat"
Fit.nIterations = 10000
print("XSPEC components:")
#
mymodel_all = MTmodel(model=fullmodel_xp, params_name=params_name, params_target=params_target, fit=Fit)
# set the values including the upper and lower limits in Xspec
mymodel_all.set_values_all(params, lowers, uppers)
# set the free parameters in Xspec
mymodel_all.set_frozen(keyfits)
#
#
#
# %%
# If not already specified in initial dictionarys set DM to False and norm to 0
change_keyfits(params_name, params_target, keyfits, "gaussian", "norm", False)
change_param_value(params, params_name, params_target, "gaussian", "norm", 0.0)
#
# since you have changed norm and key(True/False), update XSpec
mymodel_all.set_values_all(params, lowers, uppers)
mymodel_all.set_frozen(keyfits)
#
# Select the parameters to Fit. You can check that DM is not present
params_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params, lowers, uppers, params_name, params_target, keyfits)
#
#
#
# %%
if sys.argv[2] == '1':
    TypeFit = 'Minuit'
elif sys.argv[2] == '2':
    TypeFit = 'XSpec'
elif sys.argv[2] == '3':
    TypeFit = 'Diffev'
else:
    print("Type of fit not recognized. Use 1=Minuit, 2=XSpec, 3=Diffev")
    sys.exit(1)
#
# %%
#
def doFit(fullmodel_xp, params_fit, bounds_fit, params_name_fit, params_target_fit,TypeFit):
    # https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html
    Fit.statMethod = "cstat"  # fitting statistics = Cash-statistics
    Fit.nIterations=100000

    mymodel_fit = MTmodel(fullmodel_xp, params_name_fit, params_target_fit, Fit)

    if TypeFit == 'XSpec':
        mymodel_fit.set_values(params_fit)
        Fit.perform()
        Fit.perform()
        Fit.perform()
        TS_out = Fit.statistic
        nfit = Fit.nVarPars
        dof = Fit.dof
        pvalue = Fit.nullhyp
        params_out = mymodel_fit.get_values()
        #print("Test Statistics ",TS_out," testStatistics ",Fit.testStatistic," dof ",dof," nVarPars ",nfit)
        #print("pvalue ",pvalue)
        #print(params_out)
        #print(params_name_fit)
        #print(params_target_fit)

    if TypeFit == "Diffev":
        #print("here")
        out = optimize.differential_evolution(mymodel_fit.evaluate_folded, bounds_fit, polish = False,
                                      tol = 1e-4, atol = 0, popsize = 60, init = 'sobol',
                                     maxiter=100000) #maxiter=100000
        params_fit = out.x
        m = Minuit(mymodel_fit.evaluate_folded, params_fit)
        m.limits = bounds_fit
        m.errordef = Minuit.LEAST_SQUARES #i.e. 1
        m.migrad()
        params_out = mymodel_fit.get_values()
        nfit = m.nfit
        dof = len(venergies[0])*len(vfilespectrum) - nfit
        TS_out = m.fval
        pvalue = chi2_scipy.sf(TS_out,dof)
        #print("Test Statistics ",TS_out," testStatistics ",TS_out," dof ",dof," nVarPars ",nfit)
        #print("pvalue ",pvalue)
        #print(params_out)
        #print(params_name_fit)
        #print(params_target_fit)


    if TypeFit == "Minuit":
        m = Minuit(mymodel_fit.evaluate_folded, params_fit)
        m.limits = bounds_fit
        m.errordef = Minuit.LEAST_SQUARES #i.e. 1
        m.migrad()
        #params_out = []
        #for p in m.params:
            #params_out.append(p.value)
        #params_out = np.array(params_out)
        params_out = mymodel_fit.get_values()
        nfit = m.nfit
        dof = len(venergies[0])*len(vfilespectrum) - nfit
        TS_out = m.fval
        pvalue = chi2_scipy.sf(TS_out,dof)
        #print("Test Statistics ",TS_out," testStatistics ",TS_out," dof ",dof," nVarPars ",nfit)
        #print("pvalue ",pvalue)
        #print(params_out)
        #print(params_name_fit)
        #print(params_target_fit)
        
    return (params_out,TS_out,pvalue,dof,nfit)
    

# %%
# Perform Astrofit
# %%
params_astro,TS_astro, pvalue_astro, dof_astro, nfit_astro = doFit(fullmodel_xp, params_fit, bounds_fit, params_name_fit, params_target_fit,TypeFit)
#
print("Test Statistics ",TS_astro," testStatistics ",TS_astro," dof ",dof_astro," nVarPars ",nfit_astro)
print("pvalue ",pvalue_astro)
#
#

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
mymodel_all.set_values_all(params, lowers, uppers)
mymodel_all.set_frozen(keyfits)
#
# Select the parameters to Fit. You can check that DM is present
params_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params, lowers, uppers, params_name, params_target, keyfits)



# %%
# Peform fit
params_all,TS_all, pvalue_all, dof_all, nfit_dm = doFit(fullmodel_xp, params_fit, bounds_fit, params_name_fit, params_target_fit, TypeFit)
#
print("Test Statistics ",TS_all," testStatistics ",TS_all," dof ",dof_all," nVarPars ",nfit_dm)
print("pvalue ",pvalue_all)
print(params_all)
print(params_name_fit)
print(params_target_fit)
#
# Save DM normalization
DMbf = return_param_value(params_all, params_name_fit, params_target_fit, "gaussian", "norm")
print("DM bf ",DMbf)
print("DeltaTS ",TS_astro-TS_all)
#
#
params_name_fit_DM = params_name_fit.copy()
params_target_fit_DM = params_target_fit.copy()
#
#
#
#
# %%
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
    
# Just some recap of some relevant quantities along the notebook
print("********************************* RECAP***************")
print("Test Statistics Astro ",TS_astro," testStatistics ",TS_astro," dof ",dof_astro," nVarPars ",nfit_astro," bins ",dof_astro+nfit_astro)
print("pvalue ",pvalue_astro)
print("*********")
print("Test Statistics All ",TS_all," testStatistics ",TS_all," dof ",dof_all," nVarPars ",nfit_dm," bins ",dof_all+nfit_dm)
print("pvalue ",pvalue_all)
print("Delta TS ",TS_astro-TS_all)
#print("*********")
#print("bound_approx", bound_approx)

# %%
#
# set the parameters to their best-fit value of previous fit with DM
for val, name, attr in zip(params_all, params_name_fit, params_target_fit):
    change_param_value(params, params_name, params_target, name, attr, val)
#
# since you have changed norm update XSpec
mymodel_all.set_values_all(params, lowers, uppers)
#
# Select the parameters to Fit. You can check that DM is still present
params_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params, lowers, uppers, params_name, params_target, keyfits)
#
# %%
# #### Approximate bound: PASS ARRAYS _all referring to best fit model
nstepsDM = 100 # can be a bit larger since it just a matter of evaluations
bound_approx = ApproximateBound(fullmodel_xp, DMbf, normlDMhdata2, nstepsDM, TS_all, params_fit, params_name_fit, params_target_fit)
print("Approximate bound ",bound_approx)
#
#
# %% [markdown]
# Proper profiling
#
# %%
# set the parameters to their best-fit value of global fit with DM (also norm DM)
for val, name, attr in zip(params_all, params_name_fit, params_target_fit):
    change_param_value(params, params_name, params_target, name, attr, val)

# Change DM  T/F setting to F
change_keyfits(params_name, params_target, keyfits, "gaussian", "norm", False)
#
# since you have changed norm and key(True/False), update XSpec
mymodel_all.set_values_all(params, lowers, uppers)
mymodel_all.set_frozen(keyfits)
#
# Parameters should be at bf value of fit Astro+DM. Simple check that evaluations gives correct TS
print("TS now ",mymodel_all.evaluate_folded(params)," TS_all ",TS_all)
print("*****************")
#
# Select the parameters to Fit. You can check that DM is not present
params_fit, bounds_fit, params_name_fit, params_target_fit, keyfits_fit = paramfit(params, lowers, uppers, params_name, params_target, keyfits)
#
#
print("bound_approx", bound_approx)
print("DM bf ",DMbf," normlDMhdata2 ",normlDMhdata2)
print("*******************************")
nstepsDM = sys.argv[3] # number of steps for profiling
nstepsDM = int(nstepsDM)
start = bound_approx/10.
end = bound_approx*100
profile_range = np.logspace(np.log10(start), np.log10(end), num = nstepsDM)
TS_vals = np.zeros_like(profile_range)
for i, A in enumerate(profile_range):
        #set DM norm
        fullmodel_xp.gaussian.norm = A
        params_inside,TS_inside, pvalue_inside, dof_inside, nfit_inside = doFit(fullmodel_xp, params_fit, bounds_fit, params_name_fit, params_target_fit, TypeFit)
        TS_vals[i] = TS_inside
        # update params to new best fit if you want
        # otherwise you are always giving the parameters of bf model with DM
        #params_fit = params_inside
        print(i, A, TS_vals[i], TS_vals[i] - TS_all)
        if  i >= 2 and TS_vals[i] - TS_all > 150:
            profile_range = profile_range[:i+1]
            TS_vals = TS_vals[:i+1]
            break

deltaTS_vals = TS_vals - TS_all
#
A_95 = BoundProfiling(profile_range, deltaTS_vals)
#
print("bound_approx", bound_approx)
print("A_95 ",A_95)
#
#
# Check how really close you are to DeltaTS = 2.71
#
fullmodel_xp.gaussian.norm = A_95
params_inside, TS_inside, pvalue_inside, dof_inside, nfit_inside = doFit(
    fullmodel_xp, params_fit, bounds_fit, params_name_fit, params_target_fit, TypeFit
)
print("Initial A_95 ", A_95, " DeltaTS ", TS_inside-TS_all)
#
# Refine a bit more
#
factor = 1.5  # Range: A_95_refined / 2 to A_95_refined * 2
profile_range_fine = np.logspace(
    np.log10(A_95 / factor),
    np.log10(A_95 * factor),
    num=10
)
TS_vals_fine = np.zeros_like(profile_range_fine)
#
# notice that _fit has not been changed and their are set to the astro+DM best fit values
#
for i, A in enumerate(profile_range_fine):
    fullmodel_xp.gaussian.norm = A
    params_inside, TS_inside, pvalue_inside, dof_inside, nfit_inside = doFit(
        fullmodel_xp, params_fit, bounds_fit, params_name_fit, params_target_fit, TypeFit
    )
    TS_vals_fine[i] = TS_inside
    # params_fit = params_inside
    print("[REFINED]", i, A, TS_vals_fine[i], TS_vals_fine[i] - TS_all)

# Combine and sort
profile_range = np.concatenate([profile_range, profile_range_fine])
TS_vals = np.concatenate([TS_vals, TS_vals_fine])
sort_idx = np.argsort(profile_range)
profile_range = profile_range[sort_idx]
TS_vals = TS_vals[sort_idx]
deltaTS_vals = TS_vals - TS_all
#
A_95 = BoundProfiling(profile_range, deltaTS_vals)
#
fullmodel_xp.gaussian.norm = A_95
params_inside, TS_inside, pvalue_inside, dof_inside, nfit_inside = doFit(
    fullmodel_xp, params_fit, bounds_fit, params_name_fit, params_target_fit, TypeFit
)
print("Final A_95 ", A_95, " DeltaTS ", TS_inside-TS_all)
#
#
#
#
#
#
# Add some more points at small DM normalization.
# This might be useful when I combine separate fits of individual TM:
# in this case to find the global TS_dm I need to scan over norm_dm and for each value sum the TS of the different TM
# I need norm_dm also small to find the global TS_dm!
#
profile_range_low = np.logspace(
    np.log10(A_95 *1e-3 ),
    np.log10(A_95 * 0.5),
    num=10
)
TS_vals_low = np.zeros_like(profile_range_low)
#
#
# notice that _fit has not been changed and their are set to the astro+DM best fit values
#
for i, A in enumerate(profile_range_low):
    fullmodel_xp.gaussian.norm = A
    params_inside, TS_inside, pvalue_inside, dof_inside, nfit_inside = doFit(
        fullmodel_xp, params_fit, bounds_fit, params_name_fit, params_target_fit, TypeFit
    )
    TS_vals_low[i] = TS_inside
    # params_fit = params_inside
    print("[LOW]", i, A, TS_vals_low[i], TS_vals_fine[i] - TS_all)
#
# Combine and sort
profile_range = np.concatenate([profile_range, profile_range_low])
TS_vals = np.concatenate([TS_vals, TS_vals_low])
sort_idx = np.argsort(profile_range)
profile_range = profile_range[sort_idx]
TS_vals = TS_vals[sort_idx]
deltaTS_vals = TS_vals - TS_all
#
#
#
#
outdir = sys.argv[4] # output directory
os.makedirs(outdir, exist_ok=True)
filename = os.path.join(outdir, f"fit_{TypeFit}_results_E_{Eline:.3f}.h5")
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

    # === DM fit results ===
    grp_dm = grp.create_group("dm")
    grp_dm.create_dataset("TS_all", data=TS_all)
    grp_dm.create_dataset("deltaTS", data=TS_astro - TS_all)
    grp_dm.create_dataset("nbins", data=len(venergies[0]) * len(vfilespectrum))
    grp_dm.create_dataset("nfit_dm", data=nfit_dm)
    grp_dm.create_dataset("params", data=np.asarray(params_all, dtype=np.float64))
    grp_dm.create_dataset("param_names", data=params_name_fit_DM.astype("S"))
    grp_dm.create_dataset("param_targets", data=params_target_fit_DM.astype("S"))
    grp_dm.create_dataset("valid", data=1 if bound_approx > 0 else 0)

        # === Profile likelihood ===
    grp_profile = grp.create_group("profile")
    grp_profile.create_dataset("A_values", data=profile_range)
    grp_profile.create_dataset("TS_values", data=TS_vals)
    


