import os
from xspec import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
Xset.allowPrompting = False # keeps pyxspec from hanging, waiting for a response to a prompt
import matplotlib as mpl
from scipy import interpolate
#from scipy.optimize import fsolve
from scipy import optimize, stats
from iminuit import Minuit
from scipy.stats import chi2 as chi2_scipy

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}' #package mathpazo
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') #serif #ComputerModern
plt.rcParams['axes.linewidth'] = 2


################ GLOBAL QUANTITIES
Eallmin = 0.3
Eallmax = 9.
sigmaEnrange = 5
sigmaEnrangeLines = 7
starvaluelines = 5 # 1,2,3 are pl, 4 is DM line

# To data
path_data = "/home/jortecal/GitHub/eRosita/3MLFits/MarcoFitsMay/Results_False/"
nameAstro = "Astro_bf_"
nameDM = "DM_bf_"
nameProfile = "Profile_"

#TS threshold for bounds
TSth = 2.71
###################################

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
    
def selectEminEmax(Eline,vEsigma,vsigma):
    sigma = np.interp(Eline,vEsigma,vsigma)
    Emin= Eline - sigmaEnrange*sigma
    Emax= Eline + sigmaEnrange*sigma
    if(Emin<Eallmin):
        Emin=Eallmin
    if(Emax>Eallmax):
        Emax=Eallmax
    return (Emin, Emax, sigma)

# Select the astro and IB lines around Eline (in the range given by sigmaEnrangeLines*sigma)
# return arrays of energy and min/max values where they can fluctuate
def SelectLines(Eline,sigma):
    EminL = Eline - sigmaEnrangeLines*sigma
    EmaxL = Eline + sigmaEnrangeLines*sigma
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
    IBAn = IBn + An
    IBALine = np.concatenate((IBLine, ALine))
    IBALinemin = np.concatenate((IBLinemin, ALinemin))
    IBALinemax = np.concatenate((IBLinemax, ALinemax))
    idx = np.argsort(IBALine)
    IBALine = IBALine[idx]
    IBALinemin = IBALinemin[idx]
    IBALinemax = IBALinemax[idx]
    return (IBALine,IBALinemin,IBALinemax)

# Operates only on Astro lines
def FixLineszero(mpoly,IBALine,IBALinemin,IBALinemax):
    for _name in mpoly.componentNames:
    #print(_name)
        if (_name.startswith("gaussian_") and int(_name.split('_')[-1]) <= 30):
            indexl = int(_name.split('_')[-1]) - 5 # 1,2,3 are pl, 4 is DM line
            _comp = mpoly.__getattribute__(_name)
            _comp.LineE.values = [float(IBALine[indexl]), float(IBALine[indexl])*1e-2, float(IBALinemin[indexl]), float(IBALinemin[indexl]), float(IBALinemax[indexl]), float(IBALinemax[indexl])]
            _comp.LineE.frozen = True
            _comp.Sigma.values = 0.0001
            _comp.Sigma.frozen = True
            _comp.norm.values = 0.0
            _comp.norm.frozen = True
            

                            
# Create Model
# by default Astro lines are at their default energies/min/max and normalization is set to zero. All parameters frozen
# Same for DM. All parameters frozen
def PrepareModel(ElineDM, IBALine, IBALinemin, IBALinemax, chatter=0):
    Xset.chatter = chatter
    baseM = "powerlaw+powerlaw+powerlaw+gaussian"
    linesM = ""
    IBAn = len(IBALine)
    for i in range(0,IBAn):
        linesM= linesM+"+gaussian"
    finalM = baseM + linesM
    #print("Final Model ",finalM)
    AllModels.clear()
    mpoly = Model(finalM,"mpoly",1)
    mpoly.powerlaw.PhoIndex = 0.
    mpoly.powerlaw.PhoIndex.frozen = True
    mpoly.powerlaw_2.PhoIndex = 1.
    mpoly.powerlaw_2.PhoIndex.frozen = True
    mpoly.powerlaw_3.PhoIndex = 2.
    mpoly.powerlaw_3.PhoIndex.frozen = True
    sigmaDM = ElineDM* 220/3e5 #1e-4
    mpoly.gaussian.LineE = ElineDM
    mpoly.gaussian.LineE.frozen = True
    mpoly.gaussian.Sigma = sigmaDM
    mpoly.gaussian.Sigma.frozen = True
    mpoly.gaussian.norm = 0.
    mpoly.gaussian.norm.frozen = True
    FixLineszero(mpoly,IBALine, IBALinemin, IBALinemax)
    return mpoly

# Set initial conditions of the parameters, their bound and the two auxialiary quantities specifying the name of the model and the name of the model parameter
# The parameters include in order the ones of the power law, the ones of the DM line (if FitDM=True), the ones of the lines specified by indexSign
# indexSign contains the line you want to consider and starts from 0, e.g. 0,5,7 where 0 corresponds to gaussian_5
def InitialConditions(Eline, normlDMhdata, IBALine, IBALinemin, IBALinemax, indexSign, FitDM=False):
    #Poly. Power law norm is photons/keV/cm$^2$/s at 1 keV
    # Here I compute the normaziation of norm*E^(alpha) such that at 1 keV it saturates the data for alpha=0,1,2
    startpl = normlDMhdata *np.pow(Eline,0.)
    startpl_2 = normlDMhdata *np.pow(Eline,1.)
    startpl_3 = normlDMhdata *np.pow(Eline,2.)
    #
    paramspoly = np.zeros([3])
    paramspoly[0] = startpl
    paramspoly[1] = startpl_2
    paramspoly[2] = startpl_3
    boundspoly = np.zeros(shape=(3,2))
    boundspoly[0,0] = 0.
    boundspoly[0,1] = startpl*3.0
    boundspoly[1,0] = 0.
    boundspoly[1,1] = startpl_2*3.0
    boundspoly[2,0] = 0.
    boundspoly[2,1] = startpl_3*3.0
    params_name_poly = ["powerlaw","powerlaw_2","powerlaw_3"]
    params_target_poly = ["norm","norm","norm"]

    nparamsline = 2*len(indexSign) #norm and LineE
    paramsline = np.zeros([nparamsline])
    boundsline = np.zeros(shape=(nparamsline,2))
    params_name_line = []
    params_target_line = []
    indexline = 0
    for i in range(0,len(indexSign)):
        paramsline[indexline] = normlDMhdata*0.01#norm
        boundsline[indexline,0] = 0.0
        boundsline[indexline,1] = normlDMhdata*3.
        paramsline[indexline+1] = IBALine[indexSign[i]] #LineE
        boundsline[indexline+1,0] = IBALinemin[indexSign[i]]
        boundsline[indexline+1,1] = IBALinemax[indexSign[i]]
        name = "gaussian_"+str(indexSign[i]+starvaluelines)
        params_name_line.append(name)#for norm
        params_name_line.append(name)#append again for LineE
        params_target_line.append("norm")
        params_target_line.append("LineE")
        indexline = indexline+2
                
    params = np.concatenate((paramspoly, paramsline))
    bounds = np.concatenate((boundspoly, boundsline))
    params_name = np.concatenate((params_name_poly, params_name_line))
    params_target = np.concatenate((params_target_poly, params_target_line))
    

    # DM parameter is only amplitude
    if(FitDM==True):
        paramsDMline = np.array([normlDMhdata*0.01])
        boundsDMline = np.zeros(shape=(1,2))
        boundsDMline[0,0] = 0.0
        boundsDMline[0,1]= normlDMhdata*3.
        params_name_DMline = []
        params_target_DMline = []
        params_name_DMline.append("gaussian")#for norm
        params_target_DMline.append("norm")
        params = np.concatenate((params, paramsDMline))
        bounds = np.concatenate((bounds, boundsDMline))
        params_name = np.concatenate((params_name, params_name_DMline))
        params_target = np.concatenate((params_target, params_target_DMline))
    
    
    return (params, bounds, params_name, params_target)


# CLASS TO DEFINE A FUNCTION TO EVALUATE THE TS specified in fit GIVEN A xspec Model model
# with paramers value params, name (gaussuian_3, powerlaw,...) params_name and target (norm, LineE,...) params_target
class MTmodel:
    def __init__(self, model, params_name, params_target, fit): #params
        self.model = model
        ##self.params = params
        self.params_name = params_name
        self.params_target = params_target
        self.fit = fit

        # Precompile setters to eliminate getattr/setattr overhead during evaluate_folded
        self._param_setters = [
            self._make_setter(getattr(model, name), attr)
            for name, attr in zip(params_name, params_target)
        ]

    def _make_setter(self, obj, attr):
        # Closure that sets obj.attr = value
        def setter(value, obj=obj, attr=attr):
            setattr(obj, attr, value)
        return setter

    def evaluate_folded(self, params):
        for setter, value in zip(self._param_setters, params):
            setter(value)
        return self.fit.statistic

### Perform fit with only astro model and save results in a file
def AstroFit(mpoly, params, bounds, params_name, params_target, iEDM, Eline, nbins, DiffEv=True):
    Fit.statMethod = "cstat"
    Fit.nIterations=10000
    mymodel = MTmodel(model=mpoly,params_name=params_name,params_target=params_target, fit=Fit)
    if DiffEv == True:
        out = optimize.differential_evolution(mymodel.evaluate_folded, bounds, polish = False,
                                      tol = 1e-4, atol = 0, popsize = 60, init = 'sobol',
                                     maxiter=100000)
        params = out.x
    m = Minuit(mymodel.evaluate_folded, params)
    m.limits = bounds
    m.errordef = Minuit.LEAST_SQUARES #i.e. 1
    m.migrad()
    TS_astro = m.fval
    params_astro = []
    for p in m.params:
        params_astro.append(p.value)
    params_astro = np.array(params_astro)
    # SAVE IN A FILE
    namefile = path_data + nameAstro + str(iEDM) + ".dat"
    res = []
    res.append(Eline)
    res.append(TS_astro)
    res.append(m.nfit)
    res.append(nbins)
    res.append(m.valid)
    for p in m.params:
        res.append(p.value)

    np.savetxt(namefile,res,header='Eline fval par Nbins valid par')
    return (TS_astro, params_astro)

### Perform fit with astro model +DM and save results in a file
def DMFit(mpoly, params, bounds, params_name, params_target, iEDM, Eline, nbins, TS_astro, DiffEv=True):
    Fit.statMethod = "cstat"
    Fit.nIterations=10000
    mymodel_all = MTmodel(model=mpoly,params_name=params_name,params_target=params_target, fit=Fit)
    if DiffEv == True:
        out = optimize.differential_evolution(mymodel_all.evaluate_folded, bounds, tol = 1e-4, atol = 0, disp = False, popsize = 60)
        params = out.x
        
    m = Minuit(mymodel_all.evaluate_folded, params)
    m.limits = bounds
    m.errordef = Minuit.LEAST_SQUARES #i.e. 1
    m.migrad()
    TS_all = m.fval
    params_all = []
    for p in m.params:
        params_all.append(p.value)
    # SAVE IN A FILE
    namefile = path_data + nameDM  + str(iEDM) + ".dat"
    res = []
    res.append(Eline)
    res.append(TS_all)
    res.append(m.nfit)
    res.append(nbins)
    res.append(m.valid)
    res.append(TS_astro - TS_all)
    for p in m.params:
        res.append(p.value)

    np.savetxt(namefile,res,header='Eline fval par Nbins valid DeltaTS par')
    return (TS_all, params_all)

# Returns chi^2
def chi2(mpoly, params, bounds, params_name, params_target, A, DiffEv=True):
    Fit.statMethod = "cstat"
    #The fit is performed using the array params as starting point.
    # First I change the normalization of DM (which is not included in params and therefore in the fit.
    # I define new array params_inside in order not to modify params
    mpoly.gaussian.norm = A
    params_inside = params
    mymodel = MTmodel(model=mpoly,params_name=params_name,params_target=params_target, fit=Fit)
    if DiffEv == True:
        out = optimize.differential_evolution(mymodel.evaluate_folded, bounds, atol = 0, popsize = 60)
        params_inside = out.x
        
    
    m = Minuit(mymodel.evaluate_folded, params_inside)
    m.limits = bounds
    m.errordef = Minuit.LEAST_SQUARES #i.e. 1
    m.migrad()
    
    #if return_fit:
    #    return m.fval, m.np_values()
    
    return m.fval

# All astro parameters fixed to the best-fit of global fit with DM. Input: norm of DM output: TS
# To make it work pass parameters (with name and target) of best-fit
def chi2approx(mpoly, params, params_name, params_target, A):
    Fit.statMethod = "cstat"
    # The TS ie evaluated using the array params_all of best fit value and modifying last value
    # which by convention is DM. I define new array params_inside in order not to modify params_all
    params_inside = params
    params_inside[-1] = A
    #print("mpoly.gaussian.norm ",mpoly.gaussian.norm.values[0])
    mymodelprof = MTmodel(model=mpoly,params_name=params_name,params_target=params_target, fit=Fit)
    #mymodelprof = MTmodel(model=mpoly,params_name=params_name,params_target=params_target, fit=Fit)
    res = mymodelprof.evaluate_folded(params_inside)
    #print(A," ",res)
    return res


def ApproximateBound(mpoly, DMbf, normlDMhdata, nstepsDM, TS_all, params_all, params_name_all, params_target_all):
    start = np.max(np.array([DMbf,normlDMhdata*1e-5]))
    end = normlDMhdata
    profile_range = np.logspace(np.log10(start), np.log10(end), num = nstepsDM)
    TS_vals = np.zeros_like(profile_range)
    for i, A in enumerate(profile_range):
        TS_vals[i] = chi2approx(mpoly, params_all, params_name_all, params_target_all, A)
        #print("A TS ",A," ",TS_vals[i])
    x = np.log10(profile_range)
    y = TS_vals - TS_all - TSth
    vroots = np.pow(10.,MTroots(x,y))
    if(len(vroots)>0):
        bound_approx=np.max(vroots)
    else:
        bound_approx=0.
    return bound_approx
    
    
def Profiling(mpoly, bound_approx, nstepsDM, TS_all, params, bounds, params_name, params_target, DiffEv=True):
    
    start = bound_approx/10.
    end = bound_approx*100
    profile_range = np.logspace(np.log10(start), np.log10(end), num = nstepsDM)
    TS_vals = np.zeros_like(profile_range)
    for i, A in enumerate(profile_range):
        TS_vals[i] = chi2(mpoly,params, bounds, params_name, params_target, A, DiffEv)
        print(i, A, TS_vals[i], TS_vals[i] - TS_all)
    ###################
    namefile = path_data + nameProfile  + str(iEDM) + ".dat"
    data = np.column_stack([profile_range, TS_vals])
    np.savetxt(namefile,data)

def DoALLFit(nstepsDM):
    # iEDM, xElineDM, normlDMhdata, Emin, Emax, sigma, energies...
    #are defined in the loop in the main program.
    #print("normlDMhdata ",normlDMhdata)
    nbins = len(energies)
    ############ SELECT LINES
    IBALine,IBALinemin,IBALinemax = SelectLines(xElineDM,sigma)
    IBAn = len(IBALine)
    print(IBALine," ",IBALinemin," ",IBALinemax," ",IBAn)
    #print(xElineDM," ",IBAn)
    ####################################
    ############ PREPARE MODEL
    mpoly = PrepareModel(xElineDM, IBALine, IBALinemin, IBALinemax, chatter=0)
    #Xset.chatter = 10
    #AllModels.show()
    #Xset.chatter = 2
    ####################################
    ####### ALL LINES ASTRO FIT
    # Fill indexSign with all astro lines
    indexSign = np.arange(0,IBAn)
    params, bounds, params_name, params_target = InitialConditions(xElineDM, normlDMhdata, IBALine, IBALinemin, IBALinemax, indexSign, FitDM=False)
    #print(params)
    #print(bounds)
    #print(params_name)
    #print(params_target)
    ####################################
    ###################################
    ####### PERFORM THE FIT WITH ALL ASTRO PARAMETERS AND SAVE IN A FILE
    TS_astro, params_astro = AstroFit(mpoly, params, bounds, params_name, params_target, iEDM, xElineDM, nbins, DiffEv=False)
    print("TS_astro ",TS_astro)
    ######################################
    #############################################################################
    #####PREPARE PARAMETERS WITH DM
    params_all, bounds_all, params_name_all, params_target_all = InitialConditions(xElineDM, normlDMhdata, IBALine, IBALinemin, IBALinemax, indexSign, FitDM=True)
    #############################################################################
    ####### PERFORM THE FIT WITH DM AND SAVE IN A FILE
    TS_all, params_all = DMFit(mpoly, params_all, bounds_all, params_name_all, params_target_all, iEDM, xElineDM, nbins, TS_astro, DiffEv=False)
    print("TS_all ",TS_all," DeltaTS ",TS_astro-TS_all," normDM ",params_all[-1])
    #############################################################################
    #####PROFILING
    DMbf = params_all[-1]
    #### Approximate bound: PASS ARRAYS _all referring to best fit model
    bound_approx = ApproximateBound(mpoly, DMbf, normlDMhdata, nstepsDM, TS_all, params_all, params_name_all, params_target_all)
    print("Approximate bound ",bound_approx)
    # proper profiling,. NB: pass arrays params or params_astro (and corresponding name and target) which DO NOT include DM (you fix it here!)
    Profiling(mpoly, bound_approx, nstepsDM, TS_all, params_astro, bounds, params_name, params_target, DiffEv=False)


############# PATH DATAFILE
path="/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/"
filespectrum=path+"srctoolout_120_SourceSpec_00001.fits"
fileRMF=path+"srctoolout_120_RMF_00001.fits"
fileARF=path+"srctoolout_120_ARF_00001.fits"
arf1test = fits.open(fileARF)['SPECRESP']
arf1ELOW = arf1test.data.field('ENERG_LO')#keV
arf1EHI = arf1test.data.field('ENERG_HI')#keV
arf1Aeff = arf1test.data.field('SPECRESP')#cm2
arf1E = (arf1ELOW + arf1EHI)/2.
########################################




################# SIGMA
vEsigma = np.array([1.,10.])
vFWHM = np.array([0.070,0.165]) #keV
vsigma = vFWHM/(2.355)
########################



############### RELEVANT ARRAYS
nDM= 100
EDMmin = 0.5 #energies[0]
EDMmax = 9.0 #energies[-1]
vEDM = np.logspace(np.log10(EDMmin), np.log10(EDMmax), num = nDM)
nstepsDM = 10
############################################################



def ComputeBound(profile_range, TS_vals, TS_all):
    x=np.log10(profile_range)
    y = TS_vals - TS_all - TSth
    vroots = np.pow(10.,MTroots(x,y))
    bound = 0
    if(len(vroots)>0):
        bound=np.max(vroots)
    else:
        bound=0.
    print("vroots ",vroots)
    print("bound ",bound)
    return bound



# READ ALL DATA
vEline = []
vTS_astro = []
vnfit = []
vnbins = []
vvalid = []
vTS_all = []
vnfitDM = []
vvalidDM = []
vDeltaTS = []
vbound = []
for iEDM in range(0,nDM):
    # ASTRO FIT
    namefile = path_data + nameAstro + str(iEDM) + ".dat"
    data = np.loadtxt(namefile,skiprows=1)
    #print(data[0])
    vEline.append(data[0])
    vTS_astro.append(data[1])
    vnfit.append(data[2])
    vnbins.append(data[3])
    vvalid.append(data[4])
    # DM FIT
    namefile = path_data + nameDM + str(iEDM) + ".dat"
    data = np.loadtxt(namefile,skiprows=1)
    #vEline.append(data[0])
    vTS_all.append(data[1])
    vnfitDM.append(data[2])
    #vnbins.append(data[3])
    vvalidDM.append(data[4])
    vDeltaTS.append(data[5])
    # PROFILING
    namefile = path_data + nameProfile + str(iEDM) + ".dat"
    data = np.loadtxt(namefile,skiprows=1)
    res = ComputeBound(data[:,0], data[:,1], vTS_all[iEDM])
    vbound.append(res)

print(vEline)
print(vbound)
print(vTS_astro)
print(vTS_all)
print(vDeltaTS)

vTS_astro = np.array(vTS_astro)
vnbins = np.array(vnbins)
vnfit = np.array(vnfit)
vdof = vnbins - vnfit
vpdf = 1. - chi2_scipy.cdf(vTS_astro,vdof)

fileout = 'BoundFlux.pdf'
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(vEline, vbound, label='model')
ax.set_xlabel(r'E [keV]',fontsize=24)
ax.set_ylabel(r'$\Phi [cm^{-2}s^{-1}]$',fontsize=24)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax.yaxis.get_ticklocs(minor=True)
ax.minorticks_on()
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
plt.subplots_adjust(hspace=0)
plt.savefig(fileout,bbox_inches="tight")

fileout = 'DeltaTS.pdf'
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(vEline, vDeltaTS, label='model',color='green')
ax.set_xlabel(r'E [keV]',fontsize=24)
ax.set_ylabel(r'$\Delta$TS',fontsize=24)
ax.set_xscale("log")
#ax.set_yscale("log")
ax.set_ylim((-10.,10))
ax.legend()
ax.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax.yaxis.get_ticklocs(minor=True)
ax.minorticks_on()
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
plt.subplots_adjust(hspace=0)
plt.savefig(fileout,bbox_inches="tight")

fileout = 'p-value.pdf'
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(vEline, vpdf, label='model',color='green')
ax.set_xlabel(r'E [keV]',fontsize=24)
ax.set_ylabel(r'p-value',fontsize=24)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax.yaxis.get_ticklocs(minor=True)
ax.minorticks_on()
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
plt.subplots_adjust(hspace=0)
plt.savefig(fileout,bbox_inches="tight")


fileout = 'valid.pdf'
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(vEline, vvalid, label='Astro',color='green')
ax.plot(vEline, vvalidDM, label='DM',color='black')
ax.set_xlabel(r'E [keV]',fontsize=24)
ax.set_ylabel(r'Valid',fontsize=24)
ax.set_xscale("log")
#ax.set_yscale("log")
ax.legend()
ax.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax.yaxis.get_ticklocs(minor=True)
ax.minorticks_on()
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
plt.subplots_adjust(hspace=0)
plt.savefig(fileout,bbox_inches="tight")
