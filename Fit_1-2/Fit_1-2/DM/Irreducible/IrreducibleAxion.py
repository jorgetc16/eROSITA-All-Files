import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root_scalar
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}' #package mathpazo siunitx
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') #serif
plt.rcParams['axes.linewidth'] = 2

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


'''
# Path the LMC limits
LMC_file = "/home/jortecal/GitHub/eRosita/Bounds_ALP_LMCDR1.txt"
LMC_data = np.loadtxt(LMC_file)
ma_LMC = LMC_data[:, 0]          # mass [keV]
g_lim_LMC = LMC_data[:, 1]         # original limits on g under DM assumption [GeV^-1]

# Path to limits from literature
Literature_file = "/home/jortecal/GitHub/eRosita/Bounds_ALP_Literature.txt"
Literature_data = np.loadtxt(Literature_file)
ma_Literature = Literature_data[:, 0]          # mass [keV]
g_lim_Literature = Literature_data[:, 1]         # original limits on g under DM assumption [GeV^-1]
'''

# Path to irreducible abundance data
irred_data_path = "/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_windows/DM/Irreducible/PhotoPhillic_TRH=5MeV.csv"


ma, ga, Fa = np.transpose(np.loadtxt(irred_data_path,delimiter=','))

# Unique grid points
ma_vec = np.unique(ma)
ga_vec = np.unique(ga)
Fa_arr = np.transpose(np.split(Fa, ga_vec.shape[0]))

log_ma = np.log10(ma_vec)
log_ga = np.log10(ga_vec)
log_Fa = np.log10(Fa_arr)

Fa_interp = RegularGridInterpolator(
    (log_ga, log_ma),   # grid
    log_Fa,             # values on the grid
    bounds_error=False, # allows extrapolation
    fill_value=None
)

# ma in keV and ga in GeV-1
# returns fractional abundance as in Fig.S2 of https://arxiv.org/pdf/2209.06216
def Fa(ma, ga):
    """
    Wrapper for fractional abundance interpolation.
    """
    pts = np.array([[np.log10(ga), np.log10(ma)]]) #careful with order!
    fa_val = 10**Fa_interp(pts)[0]
    # If outside range, clip
    return max(fa_val, 0.0)



year = 365*24*60*60
tU = 13.8e9*year

# ma in keV and gagg in GeV-1
def TauALP(ma,gagg):
    eVtosec = 1.51927e15 #s-1
    ma_eV = ma*1e3
    Gamma_eV = ma_eV**3/(64*np.pi)*((gagg*1e-9)**2)
    #print("Gamma GeV ",Gamma_eV*1e-9)
    #print("ma ",ma)
    tau = 1./(Gamma_eV*eVtosec)
    return tau

Literature_file ='/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_windows/DM/Bound_tau_Literature.txt'
Our_file = '/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_windows/DM/Bound_tau_OURS.txt'


# CAREFUL! Bounds for DM -> gamma X while I have ALP -> gamma gamma
data = np.loadtxt(Our_file,delimiter=' ')
ma_eROSITA = data[:,0]
tau_eROSITA= data[:,1]*2 # 2photons instead of 1
#print("ma ",ma_eROSITA)
#print("Tau ",Tau)
Tau_eROSITA_interp_log = interp1d(np.log10(ma_eROSITA), np.log10(tau_eROSITA))
# ma in keV, return my bound on lifetime in seconds
def Tau_eROSITA_interp(ma):
    return 10**Tau_eROSITA_interp_log(np.log10(ma))

#print("test ",Tau_eROSITA_interp(4.27))
#print("test ",TauALP(4.3,1e-15))

# CAREFUL! Bounds for DM -> gamma X while I have ALP -> gamma gamma
data_lit = np.loadtxt(Literature_file,delimiter=' ')
ma_lit = data_lit[:,0]
tau_lit= data_lit[:,1]*2 # 2photons instead of 1
#print("ma ",ma_eROSITA)
#print("Tau ",Tau)
Tau_lit_interp_log = interp1d(np.log10(ma_lit), np.log10(tau_lit))
# ma in keV, return my bound on lifetime in seconds
def Tau_lit_interp(ma):
    return 10**Tau_lit_interp_log(np.log10(ma))




npoints_ma=10


# ama: array of masses in keV where to compute the bound
# FTauBound: interpolating function returning the bound of lifetime of ALP in sec as a function of mass in keV
# Return: ma_low/up,ga_plot_low/up: arrays contaning the mass and the value of gagg corresponding to the limit.
# Here there is an upper and lower line therefore 2 couples of ma,gagg are returned
# For some masses there is no limit therefore ma_low/up have different lenghts than ama
def Fgabound(ama,FTauBound, npoints_ga=200):
    ga_plot_low = []
    ga_plot_up = []
    ama_low = []
    ama_up = []
    for ma in ama:
        aga = np.logspace(-15., -9, num=npoints_ga)
        Fa_values = np.array([Fa(ma, g) for g in aga])
        tau_values = np.array([TauALP(ma, g) for g in aga]) #lifetime
        tauDM = FTauBound(ma) # bound for ALP=DM
        x = np.log10(aga)
        y = np.log(Fa_values) - (np.log(tau_values)-np.log(tauDM) + tU/tau_values)
        vroots = np.power(10., MTroots(x, y))
        #print("ma ",ma)
        #print("vroots ",vroots)
        #print(y)
        # Append only if the root exists
        if len(vroots) >= 1:
            ga_plot_low.append(vroots[0])
            ama_low.append(ma)
        if len(vroots) >= 2:
            ga_plot_up.append(vroots[1])
            ama_up.append(ma)
    ga_plot_low = np.array(ga_plot_low)
    ga_plot_up = np.array(ga_plot_up)
    #print("ga_plot_low ",ga_plot_low)
    return (
        np.array(ama_low), np.array(ga_plot_low),
        np.array(ama_up),  np.array(ga_plot_up)
    )


def fill_between_interpolated(ax, x_low, y_low, x_up, y_up, **kwargs):
    """
    Fill between two curves even if x arrays are different.
    
    ax      : matplotlib Axes
    x_low   : x values for lower curve
    y_low   : y values for lower curve
    x_up    : x values for upper curve
    y_up    : y values for upper curve
    kwargs  : passed to ax.fill_between
    """
    # Find the overlapping x-range
    x_min = max(x_low.min(), x_up.min())
    x_max = min(x_low.max(), x_up.max())

    if x_min >= x_max:
        print("No overlapping x-range, cannot fill")
        return

    # Create a fine grid in the overlapping range
    x_common = np.linspace(x_min, x_max, 500)

    # Interpolate both curves onto this grid
    y_low_interp = np.interp(x_common, x_low, y_low)
    y_up_interp  = np.interp(x_common, x_up, y_up)

    # Fill the region
    ax.fill_between(x_common, y_low_interp, y_up_interp, **kwargs)

    return x_common, y_low_interp, y_up_interp


npoints_ma = 300
ama = np.logspace(np.log10(ma_eROSITA[0]), np.log10(ma_eROSITA[-1]), num=npoints_ma)
ama_eROSITA_low, ga_bound_eROSITA_low, ama_eROSITA_up, ga_bound_eROSITA_up = Fgabound(ama,Tau_eROSITA_interp, npoints_ga=300)




npoints_ma = 300
ama = np.logspace(np.log10(ma_lit[0]), np.log10(ma_lit[-1]), num=npoints_ma)
ama_lit_low, ga_bound_lit_low, ama_lit_up, ga_bound_lit_up = Fgabound(ama,Tau_lit_interp, npoints_ga=300)

#print(ga_bound_eROSITA_low)

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()
    





#ax1.plot(ama_lit_low, ga_bound_lit_low, color='gray')
#ax1.plot(ama_lit_up, ga_bound_lit_up, color='gray')

# Fill the region between curves
fill_between_interpolated(ax1,
    ama_lit_low, ga_bound_lit_low,
    ama_lit_up,  ga_bound_lit_up,
    color='gray',
    alpha=0.3
)

plt.fill_between(ama_lit_up, ga_bound_lit_up,y2=4e-11,facecolor='gray',zorder=-1)

#plt.fill_between(ama_lit_low, ga_bound_lit_low,y2=4e-11,facecolor='lightgray',zorder=-1)


ax1.plot(ama_eROSITA_low, ga_bound_eROSITA_low, color='mediumseagreen')
ax1.plot(ama_eROSITA_up, ga_bound_eROSITA_up, color='mediumseagreen')

# Fill the region between curves
fill_between_interpolated(ax1,
    ama_eROSITA_low, ga_bound_eROSITA_low,
    ama_eROSITA_up,  ga_bound_eROSITA_up,
    color='mediumseagreen'
    #alpha=0.3
)

 
ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$m_{a}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$g_{a\gamma} [{\rm GeV}^{-1}]$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')


# Add a major tick at 2 keV
ax1.set_xticks([2, 3, 5, 10, 20])
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())


ax1.set_ylim(2e-14,3e-11)
ax1.set_xlim(1.9,20.)
#plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Bounds_IrreducibleAxion.pdf')



'''
####### CHECK: reproduce Fig.S2 of https://arxiv.org/pdf/2209.06216 ##########

#######  ##########

ma_test = 1e-3    # example ma
ga_test = 1e-12   # example ga
print("Interpolated Fa =", Fa(ma_test,ga_test))


npoints_ma=100
npoints_ga=200
ama = np.logspace(-3., 5., num=npoints_ma)

def GaFunc(ama,Fa_plot):
    ga_plot = []
    for ma in ama:
        aga = np.logspace(-19., -5, num=npoints_ga)
        Fa_values = np.array([Fa(ma, g) for g in aga])
        x = np.log10(aga)
        y = Fa_values - Fa_plot
        vroots = np.power(10., MTroots(x, y))
        res=np.min(vroots)
        ga_plot.append(res)
    ga_plot = np.array(ga_plot)
    #print(ga_plot)
    return ga_plot


ga_12 =GaFunc(ama,1e-12)
ga_8 =GaFunc(ama,1e-8)
ga_4 =GaFunc(ama,1e-4)
ga_1 =GaFunc(ama,1e-1)

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()
ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$m_a$ [keV]', size=24)
ax1.set_ylabel(r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]", size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlim(1e-3,4.e6)
ax1.set_ylim(1e-15,1.e-7)
    
ax1.plot(ama, ga_12, color='lightgray', linestyle='-')
ax1.plot(ama, ga_8, color='lightgray', linestyle='-')
ax1.plot(ama, ga_4, color='lightgray', linestyle='-')
ax1.plot(ama, ga_1, color='lightgray', linestyle='-')

ax1.text(1.5e-3,2.e-15,r'{\it Photophilic}',fontsize=28,bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1',alpha=0.95,zorder=4))

plt.tight_layout()
#plt.show()
plt.savefig('Test_Fa.pdf')



'''
