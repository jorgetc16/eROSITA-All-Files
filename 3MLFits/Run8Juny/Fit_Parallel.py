import numpy as np
import matplotlib
matplotlib.use("Agg")  # for non-GUI safe plotting in parallel
import matplotlib.pyplot as plt
import os
from xspec import *
from astropy.io import fits
from scipy import interpolate
from scipy import optimize
from iminuit import Minuit
from numba import njit
import h5py
import time
import csv
import traceback
import datetime
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Constants and paths
Xset.allowPrompting = False
Eallmin = 0.3
Eallmax = 9.
sigmaEnrange = 5
sigmaEnrangeLines = 7
starvaluelines = 5
TSth = 2.71

path_data = "/home/jortecal/GitHub/eRosita/3MLFits/Run8Juny/Results/"
nameAstro = "Astro_bf_"
nameDM = "DM_bf_"
nameProfile = "Profile_"

# ========= Common Data Preload =========
def preload_common_data():
    # Interpolator
    dm_data = np.loadtxt("/home/jortecal/GitHub/eRosita/3MLFits/Run8Juny/Data/DM_Lines_response.txt", delimiter=",")
    interpolator = interpolate.interp1d(dm_data[:, 0], dm_data[:, 1], kind="linear", fill_value="extrapolate")

    # ARF
    fileARF = "/home/jortecal/GitHub/eRosita/3MLFits/Run8Juny/Data/srctoolout_120_ARF_00001.fits"
    arf = fits.open(fileARF)['SPECRESP']
    arfE = (arf.data.field('ENERG_LO') + arf.data.field('ENERG_HI')) / 2.
    aeff = arf.data.field('SPECRESP')

    # Lines
    IB = np.loadtxt("/home/jortecal/GitHub/eRosita/3MLFits/Run8Juny/Data/IBLines.txt")
    Astro = np.loadtxt("/home/jortecal/GitHub/eRosita/3MLFits/Run8Juny/Data/AstroLines.txt")

    return {
        "interpolator": interpolator,
        "arf_E": arfE,
        "arf_eff": aeff,
        "IB": IB,
        "Astro": Astro
    }



@njit
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
    
def selectEminEmax(Eline):
    sigma = FWHM_interpol(Eline)/2.355
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
    data = np.loadtxt("/home/jortecal/GitHub/eRosita/3MLFits/Run8Juny/Data/IBLines.txt",skiprows=0)
    IBenergy = data[:,0]
    IBenergymin = data[:,1]
    IBenergymax = data[:,2]
    data = np.loadtxt("/home/jortecal/GitHub/eRosita/3MLFits/Run8Juny/Data/AstroLines.txt",skiprows=0)
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
    print(f"Selected {len(IBALine)} lines in the range [{EminL:.4f}, {EmaxL:.4f}] keV")
    print(f"The line and the sigma are: {Eline:.4f} keV, sigma = {sigma:.4f} keV")
    for i in range(len(IBALine)):
        print(f"  Line {i}: E = {IBALine[i]:.4f} keV, min = {IBALinemin[i]:.4f}, max = {IBALinemax[i]:.4f}")
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
def PrepareModel(ElineDM, IBALine, IBALinemin, IBALinemax, sigmaDM, chatter=0):
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
    mpoly.gaussian.LineE = ElineDM
    mpoly.gaussian.LineE.frozen = True
    mpoly.gaussian.Sigma = sigmaDM
    mpoly.gaussian.Sigma.frozen = True
    mpoly.gaussian.norm = 0.
    mpoly.gaussian.norm.frozen = True
    FixLineszero(mpoly,IBALine, IBALinemin, IBALinemax)
    return mpoly

# Set initial conditions of the parameters, their bound and the two auxialiary quantities
# specifying the name of the model and the name of the model parameter
# The parameters are in this order: the ones of the power law, the ones of the DM line (if FitDM=True)
#, the ones of the lines specified by indexSign
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
# with paramers value params, name (gaussian_3, powerlaw,...) params_name and target (norm, LineE,...) params_target
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
    t_de_start = time.time()

    print(f"[ASTRO_FIT] Starting differential evolution...")
    if DiffEv == True:
        out = optimize.differential_evolution(mymodel.evaluate_folded, bounds, polish = False,
                                      tol = 1e-4, atol = 0, popsize = 60, init = 'sobol', seed=42,
                                     maxiter=10000
        )
        params = out.x
        t_de_end = time.time()
        print(f"[ASTRO_FIT] DE completed in {t_de_end - t_de_start:.2f} s")
    print(f"[ASTRO_FIT] Starting Minuit...")
    t_minuit_start = time.time()        
    m = Minuit(mymodel.evaluate_folded, params)
    m.limits = bounds
    m.errordef = Minuit.LEAST_SQUARES #i.e. 1
    m.migrad()
    t_minuit_end = time.time()
    print(f"[ASTRO_FIT] Minuit completed in {t_minuit_end - t_minuit_start:.2f} s")
    TS_astro = m.fval
    params_astro = []
    for p in m.params:
        params_astro.append(p.value)
    params_astro = np.array(params_astro)
    result = {
    "Eline": Eline,
    "TS_astro": TS_astro,
    "params": [p.value for p in m.params],
    "params_names": params_name,
    "params_targets": params_target,
    "nfit": m.nfit,
    "valid": m.valid,
    "nbins": nbins
    }
    # SAVE IN A FILE
    # namefile = path_data + nameAstro + str(iEDM) + ".dat"
    # header = ['Eline', 'TS_astro', 'nfit', 'Nbins', 'valid'] + [f'par{i+1}' for i in range(len(m.params))]
    # row = [Eline, TS_astro, m.nfit, nbins, m.valid] + [p.value for p in m.params]
    # header_str = ' '.join(header)
    # row_str = ' '.join(str(x) for x in row)
    # with open(namefile, 'w') as f:
    #     f.write(header_str + '\n')
    #     f.write(row_str + '\n')
    return result

### Perform fit with astro model +DM and save results in a file
def DMFit(mpoly, params, bounds, params_name, params_target, iEDM, Eline, nbins, TS_astro, DiffEv=True):
    Fit.statMethod = "cstat"
    Fit.nIterations=10000
    mymodel_all = MTmodel(model=mpoly,params_name=params_name,params_target=params_target, fit=Fit)
    t_de_start = time.time()

    print(f"[DM_FIT] Starting differential evolution...")
    if DiffEv == True:
        out = optimize.differential_evolution(mymodel_all.evaluate_folded, bounds, tol = 1e-4, atol = 0, disp = False, popsize = 60, seed=42, maxiter=10000)
        params = out.x
        t_de_end = time.time()
        print(f"[DM_FIT] DE completed in {t_de_end - t_de_start:.2f} s")
    
    print(f"[DM_FIT] Starting Minuit...")
    t_minuit_start = time.time()        
    m = Minuit(mymodel_all.evaluate_folded, params)
    m.limits = bounds
    m.errordef = Minuit.LEAST_SQUARES #i.e. 1
    m.migrad()
    t_minuit_end = time.time()
    print(f"[DM_FIT] Minuit completed in {t_minuit_end - t_minuit_start:.2f} s")
    deltaTS = TS_astro - m.fval
    TS_all = m.fval
    params_all = []
    for p in m.params:
        params_all.append(p.value)
    result = {
    "Eline": Eline,
    "TS_all": TS_all,
    "params": [p.value for p in m.params],
    "params_names": params_name,
    "params_targets": params_target,
    "nfit": m.nfit,
    "valid": m.valid,
    "nbins": nbins,
    "deltaTS": deltaTS
    }
    # SAVE IN A FILE
    # namefile = path_data + nameDM + str(iEDM) + ".dat"
    # header = ['Eline', 'TS_all', 'nfit', 'Nbins', 'valid'] + [f'par{i+1}' for i in range(len(m.params))]
    # row = [Eline, TS_all, m.nfit, nbins, m.valid] + [p.value for p in m.params]
    # header_str = ' '.join(header)
    # row_str = ' '.join(str(x) for x in row)
    # with open(namefile, 'w') as f:
    #     f.write(header_str + '\n')
    #     f.write(row_str + '\n')
    return result

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
        out = optimize.differential_evolution(mymodel.evaluate_folded, bounds, atol = 0, seed=42,    popsize = 50)
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
    
    
def Profiling(mpoly, iEDM, bound_approx, nstepsDM, TS_all, params, bounds, params_name, params_target, DiffEv=True):
    
    start = bound_approx/10.
    end = bound_approx*100
    profile_range = np.logspace(np.log10(start), np.log10(end), num = nstepsDM)
    TS_vals = np.zeros_like(profile_range)
    for i, A in enumerate(profile_range):
        TS_vals[i] = chi2(mpoly,params, bounds, params_name, params_target, A, DiffEv)
        # print(i, A, TS_vals[i], TS_vals[i] - TS_all)
    ###################
    namefile = path_data + nameProfile  + str(iEDM) + ".dat"
    data = np.column_stack([profile_range, TS_vals])
    np.savetxt(namefile,data)
    

def delta_chi2(A, mpoly, params, bounds, params_name, params_target, TS_min):
    TS = chi2(mpoly, params, bounds, params_name, params_target, A, DiffEv=True)
    return TS - TS_min - TSth  # Want this to be zero

def compute_upper_limit(
    mpoly, TS_min, params, bounds, params_name, params_target,
    A_min=1e-6, A_max=1.0, tol=1e-6,
    save_plot=False, Eline=None, outdir="plots",
    max_grid_points=30, refine_near_min=True
):
    def safe_chi2(A):
        if not np.isfinite(A) or A <= 0:
            return np.inf
        try:
            return chi2(mpoly, params, bounds, params_name, params_target, A, DiffEv=False)
        except Exception as e:
            print(f"[ERROR] E = {Eline:.3f} keV | χ² failed at A = {A:.3e}: {e}")
            return np.inf

    A_min = max(A_min, 1e-10)
    A_max = max(A_max, A_min * 1000)

    try:
        y0 = safe_chi2(A_min) - TS_min - TSth
        y1 = safe_chi2(A_max) - TS_min - TSth
        if y0 * y1 > 0:
            raise ValueError("No sign change in brentq bounds.")
        A_95 = brentq(lambda A: safe_chi2(A) - TS_min - TSth, A_min, A_max, xtol=tol)
        return A_95
    except Exception as e:
        print(f"[WARN] E = {Eline:.3f} keV | brentq failed: {e}. Using fallback...")

    # Coarse scan
    A_vals = np.logspace(np.log10(A_min), np.log10(A_max), max_grid_points)
    TS_vals = np.array([safe_chi2(A) for A in A_vals])
    delta_TS = TS_vals - TS_min

    # Refine around ΔTS = 2.71 ± ε
    refine_idxs = np.where((delta_TS > 1.5) & (delta_TS < 4.0))[0]
    A_refined = []
    for i in refine_idxs:
        A1 = A_vals[max(0, i-1)]
        A2 = A_vals[min(len(A_vals)-1, i+1)]
        A_refined.extend(np.logspace(np.log10(A1), np.log10(A2), 10))
    A_refined = np.unique(A_refined)
    TS_refined = np.array([safe_chi2(A) for A in A_refined])
    delta_refined = TS_refined - TS_min

    A_vals = np.concatenate((A_vals, A_refined))
    delta_TS = np.concatenate((delta_TS, delta_refined))
    idx_sort = np.argsort(A_vals)
    A_vals = A_vals[idx_sort]
    delta_TS = delta_TS[idx_sort]

    # Precise interpolation near ΔTS = 2.71
    A_upper = None
    for i in range(1, len(delta_TS)):
        if delta_TS[i-1] < TSth < delta_TS[i]:
            logA_lo = np.log10(A_vals[i-1])
            logA_hi = np.log10(A_vals[i])
            interp = interp1d([delta_TS[i-1], delta_TS[i]], [logA_lo, logA_hi])
            A_upper = 10**interp(TSth)
            break

    if A_upper is None:
        print(f"[WARN] E = {Eline:.3f} keV | No ΔTS > 2.71 found. Extending scan...")
        A_upper = A_vals[-1]  # fallback
        # optionally: return None

    # Plot
    if save_plot and Eline is not None:
        print(f"[PLOT ] E = {Eline:.3f} keV | Saving profile plot to {outdir}/profile_E_{Eline:.3f}.png")
        os.makedirs(outdir, exist_ok=True)
        plt.figure()
        plt.plot(A_vals, delta_TS, label="ΔTS(A)")
        plt.axhline(TSth, color="red", linestyle="--", label="2.71 threshold")
        plt.axvline(A_upper, color="green", linestyle="--", label="Upper limit")
        plt.xscale("log")
        plt.xlabel("DM norm A")
        plt.ylabel("ΔTS")
        plt.title(f"Profile at E = {Eline:.3f} keV")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{outdir}/profile_E_{Eline:.3f}.png")
        plt.close()
        np.savetxt(f"{outdir}/profile_E_{Eline:.3f}.dat",
                   np.column_stack((A_vals, delta_TS)),
                   header="A_DM_norm  Delta_TS")

    return A_upper
#index  0,1,2.. index of line
#def drop_test(mpoly,index, TS_astro, params, bounds, params_name, params_target, DiffEv=False):
#    params_guess  = params
#    params_guess[index] = 0.0
#    Fit.statMethod = "cstat"
#    mymodel_guess = MTmodel(model=mpoly,params_name=params_name,params_target=params_target, fit=Fit)
#    #
#    m = Minuit(mymodel_guess.evaluate_folded, params_guess)
#    m.limits = bounds
#    m.errordef = Minuit.LEAST_SQUARES #i.e. 1
#    index_normline  = starvaluelines + index*2
#    index_lineE  = starvaluelines + index*2+1
#    m.fixed('x' + str(index_normline)) = True
#    m.fixed('x' + str(index_lineE)) = True
#    m.migrad()
#    return m.fval
    
#def dropping(mpoly,TS_astro, params, bounds, params_name, params_target, DiffEv=False):
#        params_guess  = params
    
    

def DoALLFit(nstepsDM, xElineDM, sigma, normlDMhdata, energies, iEDM, Emin, Emax, sigmaDM):
    # iEDM, xElineDM, normlDMhdata, Emin, Emax, sigma, energies...
    #are defined in the loop in the main program.
    #print("normlDMhdata ",normlDMhdata)
    nbins = len(energies)
    ############ SELECT LINES
    t_lines_start = time.time()
    IBALine,IBALinemin,IBALinemax = SelectLines(xElineDM,sigma)
    IBAn = len(IBALine)
    t_lines_end = time.time()
    print(f"[TIME ] E = {xElineDM:.3f} keV | Lines selection: {t_lines_end - t_lines_start:.2f} s")
    # print(IBALine," ",IBALinemin," ",IBALinemax," ",IBAn)
    #print(xElineDM," ",IBAn)
    ####################################
    ############ PREPARE MODEL
    mpoly = PrepareModel(xElineDM, IBALine, IBALinemin, IBALinemax, sigmaDM, chatter=0)
    #Xset.chatter = 10
    #AllModels.show()
    #Xset.chatter = 2
    ####################################
    ####### ALL LINES ASTRO FIT
    # Fill indexSign with all astro lines
    t_astro_start = time.time()
    indexSign = np.arange(0,IBAn)
    params, bounds, params_name, params_target = InitialConditions(xElineDM, normlDMhdata, IBALine, IBALinemin, IBALinemax, indexSign, FitDM=False)
    #print(params)
    #print(bounds)
    #print(params_name)
    #print(params_target)
    ####################################
    ###################################
    ####### PERFORM THE FIT WITH ALL ASTRO PARAMETERS AND SAVE IN A FILE
    astro_result = AstroFit(mpoly, params, bounds, params_name, params_target, iEDM, xElineDM, nbins, DiffEv=True)
    t_astro_end = time.time()
    print(f"[FIT  ] E = {xElineDM:.3f} keV | Astro fit: {t_astro_end - t_astro_start:.2f} s")

    TS_astro = astro_result["TS_astro"]
    params_astro = astro_result["params"]
    # print("TS_astro ",TS_astro)
    ######################################
    #############################################################################
    #####PREPARE PARAMETERS WITH DM
    params_all, bounds_all, params_name_all, params_target_all = InitialConditions(xElineDM, normlDMhdata, IBALine, IBALinemin, IBALinemax, indexSign, FitDM=True)
    #############################################################################
    ####### PERFORM THE FIT WITH DM AND SAVE IN A FILE
    t_dm_start = time.time()
    DM_result =  DMFit(mpoly, params_all, bounds_all, params_name_all, params_target_all, iEDM, xElineDM, nbins, TS_astro, DiffEv=True)
    t_dm_end = time.time()
    print(f"[FIT  ] E = {xElineDM:.3f} keV | DM fit: {t_dm_end - t_dm_start:.2f} s")

    TS_all = DM_result["TS_all"]
    params_all = DM_result["params"]
    # print("TS_all ",TS_all," DeltaTS ",TS_astro-TS_all," normDM ",params_all[-1])
    #############################################################################
    #####PROFILING
    # best fit of DM normalization
    DMbf = params_all[-1]
    #### Approximate bound: PASS ARRAYS _all referring to best fit model

    bound_approx = ApproximateBound(mpoly, DMbf, normlDMhdata, nstepsDM, TS_all, params_all, params_name_all, params_target_all)
    # print("Approximate bound ",bound_approx)
        # proper profiling,. NB: pass arrays params or params_astro (and corresponding name and target) which DO NOT include DM (you fix it here!)
    t_profile_start = time.time()
    A_95 = compute_upper_limit(
        mpoly, TS_all, params_astro, bounds, params_name, params_target,
        A_min=max(bound_approx / 100, 1e-10),
        A_max=bound_approx * 100,
        save_plot=True, Eline=xElineDM
    )
    # Profiling(mpoly, iEDM, bound_approx, nstepsDM, TS_all, params_astro, bounds, params_name, params_target, DiffEv=False)
    t_profile_end = time.time()
    print(f"[FIT  ] E = {xElineDM:.3f} keV | Bound: {t_profile_end - t_profile_start:.2f} s")
    print("Approximate 95% CL upper limit on DM norm:", bound_approx)
    print("95% CL upper limit on DM norm:", A_95)


    t_save_start = time.time()
    with h5py.File(f"{path_data}fit_results_E_{xElineDM:.3f}.h5", "w") as f:
        grp = f.create_group(f"E_{xElineDM:.3f}")
        # Add metadata as attributes
        grp.attrs["sigma"] = sigma
        grp.attrs["Emin"] = Emin
        grp.attrs["Emax"] = Emax
        grp.attrs["Number_of_IB_lines"] = IBAn
        grp.attrs["Date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        astro_grp = grp.create_group("astro")
        dm_grp = grp.create_group("dm")

        # Save Astro datasets
        for key in ["TS_astro", "params", "nfit", "nbins", "valid"]:
            astro_grp.create_dataset(key, data=astro_result[key])
        astro_grp.create_dataset("param_names", data=np.array(astro_result["params_names"], dtype='S'))
        astro_grp.create_dataset("param_targets", data=np.array(astro_result["params_targets"], dtype='S'))

        # Save DM datasets
        for key in ["TS_all", "params", "nfit", "nbins", "valid", "deltaTS"]:
            dm_grp.create_dataset(key, data=DM_result[key])
        dm_grp.create_dataset("param_names", data=np.array(DM_result["params_names"], dtype='S'))
        dm_grp.create_dataset("param_targets", data=np.array(DM_result["params_targets"], dtype='S'))
        grp.attrs["A_95"] = A_95


    t_save_end = time.time()
    print(f"[SAVE ] E = {xElineDM:.3f} keV | Save results: {t_save_end - t_save_start:.2f} s")



# === Entry point ===
def run_fit_for_energy(xElineDM, common_data):
    start_time = time.time()
    print(f"[START] E = {xElineDM:.3f} keV")
    try:
        # Unpack shared data
        interpolator = common_data["interpolator"]
        arfE = common_data["arf_E"]
        aeff = common_data["arf_eff"]
        IB = common_data["IB"]
        Astro = common_data["Astro"]

        # File paths (only used in this worker)
        path = "/home/jortecal/GitHub/eRosita/3MLFits/Run8Juny/Data/"
        filespectrum = path + "srctoolout_120_SourceSpec_00001.fits"
        fileRMF = path + "srctoolout_120_RMF_00001.fits"
        fileARF = path + "srctoolout_120_ARF_00001.fits"

        t_load_start = time.time()
        

        # Load data and apply energy cut
        AllData.clear()
        Xset.chatter = 0
        s1 = Spectrum(filespectrum)
        s1.response = fileRMF
        s1.response.arf = fileARF

        # Compute sigma and energy window
        sigma = interpolator(xElineDM)*1e-3 / 2.355
        
        sigmaDM = np.sqrt((interpolator(xElineDM)*1e-3 / 2.355)**2 + (xElineDM* 220/3e5)**2)

        Emin = max(xElineDM - sigmaEnrange * sigma, Eallmin)
        Emax = min(xElineDM + sigmaEnrange * sigma, Eallmax)
        str_range = f"**-{Emin:.4f},,{Emax:.4f}-**"
        AllData.ignore(str_range)
        t_load_end = time.time()
        print(f"[TIME ] E = {xElineDM:.3f} keV | Data load: {t_load_end - t_load_start:.2f} s")


        t_plot_start = time.time()
        Plot.commands = ()
        Plot.xAxis = "keV"
        Plot("data")
        energies = np.asarray(Plot.x())
        rates = np.asarray(Plot.y())
        t_plot_end = time.time()
        print(f"[TIME ] E = {xElineDM:.3f} keV | Plot extract: {t_plot_end - t_plot_start:.2f} s")



        Aefftest = np.interp(xElineDM, arfE, aeff)
        Ratetest = np.interp(xElineDM, energies, rates)
        normlDMhdata = Ratetest / Aefftest

        nstepsDM = 10
        iEDM = str(xElineDM).replace('.', '_')

        t_fit_start = time.time()
        DoALLFit(nstepsDM, xElineDM, sigma, normlDMhdata, energies, iEDM, Emin, Emax, sigmaDM)
        t_fit_end = time.time()
        print(f"[TIME ] E = {xElineDM:.3f} keV | Fitting: {t_fit_end - t_fit_start:.2f} s")


        total_time = time.time() - start_time
        print(f"[DONE ] E = {xElineDM:.3f} keV | Total time: {total_time:.2f} s")
    
    except Exception as e:
        print(f"[ERROR] E = {xElineDM:.3f} keV: {e}")
        traceback.print_exc() 

# if __name__ == "__main__":
#     import warnings
#     warnings.filterwarnings("ignore")  # Optional: suppress matplotlib and optimization warnings

#     # Energy to test
#     test_Eline = 2.404  # in keV

#     # Load reusable data
#     print(f"\n[TEST] Running test fit at {test_Eline} keV...")
#     common_data = preload_common_data()

#     # Run the full pipeline for one energy
#     run_fit_for_energy(test_Eline, common_data)

#     print(f"[TEST DONE] Finished test fit at {test_Eline} keV.")