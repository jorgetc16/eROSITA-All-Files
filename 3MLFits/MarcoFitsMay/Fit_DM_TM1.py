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
from scipy.optimize import fsolve

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}' #package mathpazo
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') #serif #ComputerModern
plt.rcParams['axes.linewidth'] = 2

'''
def PlotFit(fileout, vTM=[1,2,3,4,5,6,7], FitModel=True):
    #if ax:
    #    ax.clear()
    #fig, ax = plt.subplots(figsize=(10,6))
    if(FitModel==True):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(10,6))
    Plot.commands = ()
    Plot.xAxis = "keV"
    Plot("data")
    for plot_grp in vTM:
        energies = np.asarray(Plot.x(plot_grp))
        edeltas = np.asarray(Plot.xErr(plot_grp))
        rates = np.asarray(Plot.y(plot_grp))
        errors = np.asarray(Plot.yErr(plot_grp))
        ax1.errorbar(energies, rates, xerr=edeltas,yerr=errors,fmt='.',markersize='5',label='data')
        if(FitModel==True):
            foldedmodel = np.asarray(Plot.model(plot_grp))
            ax1.plot(energies, foldedmodel,label='model')
    #ax1.set_xlabel('Energy (keV)',)
    ax1.set_ylabel(r'counts/s/keV',fontsize=24)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
    ax1.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True,pad=10)
    #plt.xticks(fontsize=22)
    #plt.yticks(fontsize=22)
    ax1.tick_params(axis='y', labelsize=22)
    ax1.tick_params(axis='x', labelsize=22)
    ax1.yaxis.get_ticklocs(minor=True)
    ax1.minorticks_on()
    ax1.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if(FitModel==False):
        ax1.set_xlabel('Energy (keV)',fontsize=24)
    if(FitModel==True):
        ax1.set_xticklabels([])  # Remove x-axis labels for the first subplot
        ax2.plot(energies,(rates-foldedmodel)/errors,label='residuals')
        ax2.set_xlabel('Energy (keV)',fontsize=24)
        ax2.set_ylabel(r' (data-model)/error',fontsize=24)
        ax2.set_xscale("log")
        ax2.legend()
        ax2.tick_params(which='major', direction='in', width=1, length=10, top=True, right=True, pad=10)
        ax2.tick_params(which='minor', direction='in', width=1, length=7, top=True, right=True, pad=10)
        #plt.xticks(fontsize=22)
        #plt.yticks(fontsize=22)
        ax2.tick_params(axis='y', labelsize=22)
        ax2.tick_params(axis='x', labelsize=22)
        ax2.yaxis.get_ticklocs(minor=True)
        ax2.minorticks_on()
    plt.subplots_adjust(hspace=0)
    plt.savefig(fileout,bbox_inches="tight")
    return
 
 
Xset.chatter = 1
'''

##### GLOBAL QUANTITIES
Eallmin = 0.3
Eallmax = 9.
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
    Emin= Eline - 5*sigma
    Emax= Eline + 5*sigma
    if(Emin<Eallmin):
        Emin=Eallmin
    if(Emax>Eallmax):
        Emax=Eallmax
    return (Emin, Emax, sigma)
    
def SelectLines(Eline,sigma):
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
    IBidx = np.where( (IBenergy > EminL ) & (IBenergy < EmaxL ) )
    Aidx = np.where( (Aenergy > EminL ) & (Aenergy < EmaxL ) )
    IBLine = IBenergy[IBidx]
    IBLinemin = IBenergymin[IBidx]
    IBLinemax = IBenergymax[IBidx]
    ALine = Aenergy[Aidx]
    ALinemin = Aenergymin[Aidx]
    ALinemax = Aenergymax[Aidx]
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
    return (IBALine,IBALinemin,IBALinemax,IBAn)
    
def PrepareModel(ElineDM, IBAn, chatter=0):
    Xset.chatter = chatter
    baseM = "powerlaw+powerlaw+powerlaw+gaussian"
    linesM = ""
    for i in range(0,IBAn):
        linesM= linesM+"+gaussian"
    finalM = baseM + linesM
    print("Final Model ",finalM)
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
    for _name in mpoly.componentNames:
        #print(_name)
        if (_name.startswith("gaussian_") and int(_name.split('_')[-1]) <= 30):
            print(_name)
            indexl = int(_name.split('_')[-1]) - 5 # 1,2,3 are pl, 4 is DM line
            print(indexl)
            _comp = mpoly.__getattribute__(_name)
            #print(_comp)
            _comp.LineE.values = [float(IBALine[indexl]), float(IBALine[indexl])*1e-2, float(IBALinemin[indexl]), float(IBALinemin[indexl]), float(IBALinemax[indexl]), float(IBALinemax[indexl])]
            #print(_comp.LineE.values)
            _comp.LineE.frozen = True
            _comp.Sigma.values = 0.0001
            _comp.Sigma.frozen = True
            _comp.norm.values = 0.0
            _comp.norm.frozen = True
    return mpoly

def DoFit(mpoly,loopfit,thresholdfit):
    Fit.statMethod = "cstat"
    Fit.nIterations=10000
    Fit.perform()
    Fit.perform()
    Fit.perform()
    Fit.perform()
    TSbase2 = Fit.statistic
    for i in range(0,loopfit):
        Fit.perform()
        Fit.perform()
        Fit.perform()
        Fit.perform()
        Fit.perform()
        TSnew = Fit.statistic
        if(TSbase2 - TSnew <thresholdfit):
            break
    return (TSnew, Fit.testStatistic, Fit.dof)

def SelectRelevantLines(mpoly):
    # FIT WITH NO LINES
    Fit.statMethod = "cstat"
    Fit.nIterations=10000
    Fit.perform()
    Fit.perform()
    Fit.perform()
    Fit.perform()
    TSnolines = Fit.statistic
    #dofnolines = Fit.dof
    #print("Test Statistics ",TSnolines," testStatistics ",Fit.testStatistic," dof ",dofnolines," nVarPars ",Fit.nVarPars)
    loopfit = 100
    thresholdfit =0.01
    # INTRODUCE ASTRO LINES one by one
    THretainlines = 4
    IBAnSign = 0
    indexSign = []
    TSbase = TSnolines
    for _name in mpoly.componentNames:
        if (_name.startswith("gaussian_") and int(_name.split('_')[-1]) <= 30):
            indexl = int(_name.split('_')[-1]) - 5 # 1,2,3 are pl, 4 is DM line
            _comp = mpoly.__getattribute__(_name)
            _comp.norm.frozen = False
            _comp.LineE.frozen = False
            #Fit.statMethod = "cstat"
            #Fit.nIterations=10000
            #Fit.perform()
            #Fit.perform()
            #Fit.perform()
            #Fit.perform()
            #TSbase2 = Fit.statistic
            #dofnew = Fit.dof
            #for i in range(0,loopfit):
                #Fit.perform()
                #Fit.perform()
                #Fit.perform()
                #Fit.perform()
                #Fit.perform()
                #TSnew = Fit.statistic
                #if(TSbase2 - TSnew <thresholdfit):
                #    break
            (TSnew, res, res2) = DoFit(mpoly,loopfit,thresholdfit)
            DeltaTS = TSbase - TSnew
            if( DeltaTS > 4):
                IBAnSign = IBAnSign + 1
                indexSign.append(int(_name.split('_')[-1]))
            TSbase = TSnew
    ####
    loopfit = 1000
    thresholdfit =0.001
    # At this stage all fines are free
    IBAnSign = 0
    Tfinal = TSbase
    indexSign = []
    for _name in mpoly.componentNames:
        if (_name.startswith("gaussian_") and int(_name.split('_')[-1]) <= 30):
            print(_name)
            indexl = int(_name.split('_')[-1]) - 5 # 1,2,3 are pl, 4 is DM line
            #print(indexl)
            _comp = mpoly.__getattribute__(_name)
            #set to zero and freeze
            _comp.norm = 0.0
            _comp.norm.frozen = True
            _comp.LineE.frozen = True
            #Fit.statMethod = "cstat"
            #Fit.nIterations=10000
            #Fit.perform()
            #Fit.perform()
            #Fit.perform()
            #Fit.perform()
            #TSbase2 = Fit.statistic
            #dofnew = Fit.dof
            #for i in range(0,loopfit):
                #print(i)
                #Fit.perform()
                #Fit.perform()
                #Fit.perform()
                #Fit.perform()
                #Fit.perform()
                #TSnew = Fit.statistic
                #dofnoDM = Fit.dof
                #if(TSbase2 - TSnew <thresholdfit):
                    #break
            (TSnew, res, dofnew) = DoFit(mpoly,loopfit,thresholdfit)
            print(TSnew," ",dofnew)
            DeltaTS = TSnew - Tfinal
            if( DeltaTS > 4):
                IBAnSign = IBAnSign + 1
                indexSign.append(int(_name.split('_')[-1]))
                #allow to fluctuate again
                _comp.norm.frozen = False
                _comp.LineE.frozen = False
    return indexSign

def FitNODM(mpoly,indexSign):
    loopfit = 20000
    thresholdfit =0.001
    # DO AGAIN, free only astro which are significant
    for _name in mpoly.componentNames:
        if (_name.startswith("gaussian_") and int(_name.split('_')[-1]) <= 30):
            #print(_name)
            indexl = int(_name.split('_')[-1]) - 5 # 1,2,3 are pl, 4 is DM line
            #print(indexl)
            _comp = mpoly.__getattribute__(_name)
            #print(_comp)
            if int(_name.split('_')[-1]) in indexSign:
                _comp.norm.frozen = False
                _comp.LineE.frozen = False
            else:
                _comp.norm.values = 0.0
                _comp.norm.frozen = True
                _comp.LineE.frozen = True
    loopfit = 20000
    thresholdfit =0.001
    # DO AGAIN, free only astro which are significant
    for _name in mpoly.componentNames:
        if (_name.startswith("gaussian_") and int(_name.split('_')[-1]) <= 30):
            #print(_name)
            indexl = int(_name.split('_')[-1]) - 5 # 1,2,3 are pl, 4 is DM line
            #print(indexl)
            _comp = mpoly.__getattribute__(_name)
            #print(_comp)
            if int(_name.split('_')[-1]) in indexSign:
                _comp.norm.frozen = False
                _comp.LineE.frozen = False
            else:
                _comp.norm.values = 0.0
                _comp.norm.frozen = True
                _comp.LineE.frozen = True
    #Fit.statMethod = "cstat"
    #Fit.nIterations=10000
    #Fit.perform()
    #Fit.perform()
    #Fit.perform()
    #Fit.perform()
    #Fit.perform()
    #TSnoDM = Fit.statistic
    #dofnoDM = Fit.dof
    #print("Test Statistics ",TSnoDM," testStatistics ",Fit.testStatistic," dof ",dofnoDM," nVarPars ",Fit.nVarPars)
    # PUT LOOP OF FIT TO CHECK CONVERGENCE
    #TSbase = TSnoDM
    #for i in range(0,loopfit):
        #print(i)
        #Fit.perform()
        #Fit.perform()
        #Fit.perform()
        #Fit.perform()
        #Fit.perform()
        #TSnoDM = Fit.statistic
        #dofnoDM = Fit.dof
        #print("Test Statistics ",TSnoDM," testStatistics ",Fit.testStatistic," dof ",dofnolines," nVarPars ",Fit.nVarPars)
        #if(TSbase - TSnoDM <thresholdfit):
        #    break
        #TSbase = TSnoDM
    TSnoDM, testStatisticnoDM, dofnoDM = DoFit(mpoly,loopfit,thresholdfit)
    #print("iterations in the loop ",i," loopfit ",loopfit)
    #TSnoDM = Fit.statistic
    #dofnoDM = Fit.dof
    print("Test Statistics ",TSnoDM," testStatistics ",testStatisticnoDM," dof ",dofnoDM," nVarPars ",Fit.nVarPars)
    Xset.chatter=10
    AllModels.show()
    Fit.show()
    Xset.chatter = 1
    return (TSnoDM, testStatisticnoDM, dofnoDM)

# Some of the lines which are free have best-fit values really small
# To help the fit I fix these values
def FixLines(mpoly,indexSign):
    Maxnormline = 0
    for _name in mpoly.componentNames:
        if (_name.startswith("gaussian_") and int(_name.split('_')[-1]) <= 30):
            indexl = int(_name.split('_')[-1]) - 5 # 1,2,3 are pl, 4 is DM line
            _comp = mpoly.__getattribute__(_name)
            if int(_name.split('_')[-1]) in indexSign:
                if(Maxnormline < _comp.norm.values[0]):
                    Maxnormline = _comp.norm.values[0]
    #print("Maxnormline ", Maxnormline)
    IBAnSignsmall = 0
    for _name in mpoly.componentNames:
        if (_name.startswith("gaussian_") and int(_name.split('_')[-1]) <= 30):
            indexl = int(_name.split('_')[-1]) - 5 # 1,2,3 are pl, 4 is DM line
            _comp = mpoly.__getattribute__(_name)
            if int(_name.split('_')[-1]) in indexSign:
                if(_comp.norm.values[0] < Maxnormline*1e-3):
                    IBAnSignsmall = IBAnSignsmall +1
                    #print("exclude ",_name)
                    _comp.norm.frozen = True
                    _comp.LineE.frozen = True
    IBAnSign = len(indexSign)
    IBAnSign2 = IBAnSign - IBAnSignsmall
    return  (IBAnSign2,Maxnormline)

def DoDMfit(mpoly):
    loopfit = 20000
    thresholdfit =0.001
    mpoly.gaussian.norm.frozen = False
    TSDMline, testStatisticDMline, dofDMline = DoFit(mpoly,loopfit,thresholdfit)
    normDMbf = mpoly.gaussian.norm.values[0]
    return (TSDMline, testStatisticDMline, dofDMline, normDMbf)

def doDMscan(mpoly,normDMbf, normlDMhdata, Maxnormline, TSDMline):
    if normDMbf <= Maxnormline*1e-2 or normDMbf <= normlDMhdata*1e-4:
    #print("NO BF")
        if(Maxnormline >0):
            normmin = Maxnormline*1e-2
            normmax = Maxnormline*1.
        else:
            normmin = normlDMhdata*1e-3
            normmax = normlDMhdata*0.2
    else:
        #print("BF")
        normmin = normDMbf
        normmax = normDMbf*10.0
    vnormDM = np.logspace(np.log10(normmin), np.log10(normmax), num = nstepsDM)
    vTSDM = np.zeros(nstepsDM)
    ibrealk = 0
    loopfit = 20000
    thresholdfit =0.01 #0.001
    for inorm in range(0,len(vnormDM)):
        print("inorm ",inorm)
        norm = vnormDM[inorm]
        mpoly.gaussian.norm.values = norm
        mpoly.gaussian.norm.frozen = True
        TSDMlinetest, testStatisticDMlinetest, dofDMlinetest = DoFit(mpoly,loopfit,thresholdfit)
        vTSDM[inorm] = TSDMlinetest
        if inorm > 1 and (vTSDM[inorm] - vTSDM[inorm-1] > 0.) and vTSDM[inorm] - TSDMline > 10.:
            ibrealk = inorm
            break
    if ibrealk > 0:
        for inorm in range (ibrealk,len(vnormDM)):
            vTSDM[inorm] = vTSDM[ibrealk]
    return (vnormDM,vTSDM)
        
def DoALLFit(mpoly, normlDMhdata, nstepsDM):
    indexSign = SelectRelevantLines(mpoly)
    print(indexSign)
    TSnoDM, testStatisticnoDM, dofnoDM = FitNODM(mpoly,indexSign)
    print("Test Statistics ",TSnoDM," testStatistics ",testStatisticnoDM," dof ",dofnoDM)
    # Fix lines
    IBAnSign2, Maxnormline = FixLines(mpoly,indexSign)
    print("IBAnSign2 ",IBAnSign2)
    # Fit again
    TSnoDM, testStatisticnoDM, dofnoDM = FitNODM(mpoly,indexSign)
    # DM dest-fit
    TSDMline, testStatisticDMline, dofDMline, normDMbf = DoDMfit(mpoly)
    # SAVE INFO IN ARRAY
    vinfo = np.zeros(20)
    vinfo[0] = len(indexSign)
    vinfo[1] = IBAnSign2
    vinfo[2] = TSnoDM
    vinfo[3] = testStatisticnoDM
    vinfo[4] = dofnoDM
    vinfo[5] = TSDMline
    vinfo[6] = testStatisticDMline
    vinfo[7] = dofDMline
    vinfo[8] = TSnoDM - TSDMline
    print("*****************************")
    print("NOW SCAN")
    print("*****************************")
    # SCAN
    vnormDM, vTSDM = doDMscan(mpoly,normDMbf, normlDMhdata, Maxnormline, TSDMline)
    return (vinfo, vnormDM, vTSDM)


#def SaveonFile(namefile,vinfo, vnormDM, vTSDM):


############# PATH DATAFILE
path="/Users/marcotaoso/Documents/2024/eROSITA/FromJorge/LMC_test_3deg/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/"
filespectrum=path+"srctoolout_120_SourceSpec_00001.fits"
fileRMF=path+"srctoolout_120_RMF_00001.fits"
fileARF=path+"srctoolout_120_ARF_00001.fits"
arf1test = fits.open(fileARF)['SPECRESP']
arf1ELOW = arf1test.data.field('ENERG_LO')#keV
arf1EHI = arf1test.data.field('ENERG_HI')#keV
arf1Aeff = arf1test.data.field('SPECRESP')#cm2
arf1E = (arf1ELOW + arf1EHI)/2.
########################################

#### PATH TO SAVE DAT#############################
pathOut = "/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_test_3deg/Fitpoly/TM1/LowRes/data/"
NameInfo = "Info_"
NameScan = "Scan_"
########################


################# SIGMA
vEsigma = np.array([1.,10.])
vFWHM = np.array([0.070,0.165]) #keV
vsigma = vFWHM/(2.355)
########################



############### RELEVANT ARRAYS
nDM= 10
EDMmin = 0.5 #energies[0]
EDMmax = 9.0 #energies[-1]
vEDM = np.logspace(np.log10(EDMmin), np.log10(EDMmax), num = nDM)
nstepsDM = 10
############################################################


for iEDM in range(0,1):
    xElineDM = vEDM[iEDM]
    Emin, Emax, sigma = selectEminEmax(xElineDM,vEsigma,vsigma)
    print("Emin ",Emin," Emax ",Emax," sigma ",sigma)
    str_range = "**-"+str(round(Emin, 4))+",,"+str(round(Emax, 4))+"-**"
    #########SELECT DATA
    AllData.clear()
    s1 = Spectrum(filespectrum)
    s1.response=fileRMF
    s1.response.arf=fileARF
    AllData.ignore(str_range)
    Plot.commands = ()
    Plot.xAxis = "keV"
    Plot("data")
    energies = np.asarray(Plot.x())
    edeltas = np.asarray(Plot.xErr())
    rates = np.asarray(Plot.y())
    errors = np.asarray(Plot.yErr())
    print("Selected values ",np.min(energies)," ",np.max(energies)," Nbins ",len(energies))
    ############ SELECT LINES
    IBALine,IBALinemin,IBALinemax, IBAn = SelectLines(xElineDM,sigma)
    print(IBALine," ",IBALinemin," ",IBALinemax," ",IBAn)
    ####################################
    ############ COMPUTE normlDMhdata
    Aefftest = np.interp(xElineDM, arf1E, arf1Aeff)
    Ratetest = np.interp(xElineDM, energies, rates)
    normlDMhdata = Ratetest/Aefftest
    print("normlDMhdata ",normlDMhdata)
    ####################################
    ############ PREPARE MODEL
    mpoly = PrepareModel(xElineDM, IBAn, chatter=0)
    Xset.chatter = 10
    AllModels.show()
    Xset.chatter = 2
    ############################# FILTER LINES
    vinfo, vnormDM, vTSDM = DoALLFit(mpoly, normlDMhdata, nstepsDM)
    #vinfo = np.zeros(20)
    arrInfo = np.array([[vinfo[0],vinfo[1]], [vinfo[2],vinfo[3]], [vinfo[4],vinfo[5]],[vinfo[6],vinfo[7]],[vinfo[8],xElineDM]])
    namefile = pathOut + NameInfo + str(iEDM) +".dat"
    np.savetxt(namefile,arrInfo, header="IBAnSign IBAnSign2 TSnoDM IBAnStestStatisticnoDMign2 dofnoDM TSDMline testStatisticDMline dofDMline DTS Eline")
    namefile = pathOut + NameScan + str(iEDM) +".dat"
    data = np.column_stack([vnormDM, vTSDM])
    np.savetxt(namefile,data)
    


