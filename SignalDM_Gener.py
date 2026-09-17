import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
import sys
import pandas as pd
#import matplotlib as mp
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion, RectangleSkyRegion
from matplotlib.colors import LogNorm
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

from matplotlib import rc
import matplotlib.patheffects as path_effects
import numpy.ma as ma

import time
from datetime import timedelta

from gammapy.maps import Map
profiles_path = "/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge"
if profiles_path not in sys.path:
    sys.path.append(profiles_path)

import profiles
from MTJFactory import MTJFactory



from gammapy.astro.darkmatter import (
    #DarkMatterAnnihilationSpectralModel,
    #DarkMatterDecaySpectralModel,
    JFactory,
    #PrimaryFlux,
    #profiles,
)

from gammapy.maps import WcsGeom, WcsNDMap






prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}' #package mathpazo siunitx
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') #serif
plt.rcParams['axes.linewidth'] = 2



Dfactor_wmask = 4.04e20 #GeV cm-2

t_Universe = 4.35e17 #s


# D [GeV cm-2]
# M [GeV]
# tau [s]
#Out: cm-2 s-1
def FluxLine(Mdm, Dfactor, tau, nphoton=1):
    #print("nphoton FluxLine ",nphoton)
    return Dfactor/(4*(np.pi)*Mdm*tau)*nphoton

# Eline in keV Fluxlimit in photons/s/cm2
# returns Mass [keV] and tau [s]
def bound_tau(Eline, Fluxlim, Dfactor, nphoton=1):
    #Mass = Elin2*2
    vMdm_kev = Eline*2
    vMdm_gev = vMdm_kev*1e-6 # GeV
    xtau=1.0 #s
    # 1 photon for sterile nu
    #print("nphoton ",nphoton)
    vFluxDM = FluxLine(vMdm_gev, Dfactor, xtau, nphoton)
    # bound on lifetime
    vbound_tau = xtau*vFluxDM/Fluxlim # in sec
    return (vMdm_kev, vbound_tau)


# Eline in keV Fluxlimit in photons/s/cm2
# returns Mass [keV] and sin^2(2*theta)
def FSinsq2theta(Eline,Fluxlim,Dfactor):
    #
    vMdm_kev, vbound_tau = bound_tau(Eline, Fluxlim, Dfactor, nphoton=1)
    # see eq.1 of https://arxiv.org/pdf/1911.09120
    # and eq. 4.13 of https://arxiv.org/pdf/1602.04816.
    # and eq.10 of https : // arxiv . org/pdf/2207.04572
    #See NFW.nb in this folder
    vSinsq2theta = 7.2e21*pow(vbound_tau,-1.0)*pow(vMdm_kev,-5.0)
    return (vMdm_kev, vSinsq2theta)
    
# Eline in keV Fluxlimit in photons/s/cm2
# returns Mass [keV] and gagg [GeV-1]
def Fgagg(Eline,Fluxlim,Dfactor):
    #
    vMdm_kev, vbound_tau = bound_tau(Eline, Fluxlim, Dfactor, nphoton=2)
    eVtosec = 1.51927e15 #s-1
    vMdm_gev = vMdm_kev*1e-6 # GeV
    vGamma_GeV = 1.0/vbound_tau*1/eVtosec*1e-9 # GeV
    vgagg = np.sqrt(64*np.pi*vGamma_GeV/(vMdm_gev**3))
    return (vMdm_kev, vgagg)
    
# Transform bound of sterile nu ( on sin2(2theta) ) to bound on ALP (on g_agg in GeV-1)
def FSinsq2theta_gagg(vMdm_kev, vSinsq2theta):
    # mass is the same, always 1->2 decay
    vMdm_gev = vMdm_kev*1e-6
    # bound on lifetime
    vbound_tau = 7.2e21*pow(vSinsq2theta,-1.0)*pow(vMdm_kev,-5.0) #s
    # bound on lifetime for ALP is a factor 2 more stringent because 2 photons in the decay vs 1
    vbound_tau = vbound_tau*2.0
    eVtosec = 1.51927e15 #s-1
    vGamma_GeV = 1.0/vbound_tau*1/eVtosec*1e-9 # GeV
    vgagg = np.sqrt(64*np.pi*vGamma_GeV/(vMdm_gev**3))
    return  (vMdm_kev, vgagg)

# Transform bound of sterile nu ( on sin2(2theta) ) to bound on lifetime (in sec)
def FSinsq2theta_tau(vMdm_kev, vSinsq2theta):
    # mass is the same, always 1->2 decay
    vMdm_gev = vMdm_kev*1e-6
    # bound on lifetime
    vbound_tau = 7.2e21*pow(vSinsq2theta,-1.0)*pow(vMdm_kev,-5.0) #s
    return  (vMdm_kev, vbound_tau)
    
def IrreducibleAxionBackground(m_a, tau_lim, T_RH=5):
    g_agamma_irr = 1e8*(3e-6 * t_Universe * 5 * (T_RH/5)**-1 / tau_lim)**(1/4) / m_a
    return g_agamma_irr

# -------------------------------------------------------------------------
# TBABS nH correction factor:
# The flux limits were obtained with TBABS nH=0.22 x 10^22 cm-2.
# The corrected value is nH=0.145. The file below contains the
# energy-dependent ratio: TBABS(nH=0.22) / TBABS(nH=0.145).
# We multiply old flux limits by this ratio to get corrected limits.
# See RefereeChecks/nH/Marco_nH/Checks_EffectTBABSonLine.ipynb
# -------------------------------------------------------------------------
_corr_data = np.loadtxt("/home/jortecal/GitHub/eRosita/RefereeChecks/nH/Marco_nH/CorrectionFactor.txt")
corr_factor = interp1d(_corr_data[:, 0], _corr_data[:, 1],
                        bounds_error=False, fill_value=1.0)

File_fluxlimit1 = np.loadtxt("/home/jortecal/GitHub/eRosita/2to9FixedLines/Results/CombinedProfiles/CombinedBounds.dat",skiprows=1)
vEline1 = File_fluxlimit1[:,0]
vFluxlim1 = File_fluxlimit1[:,1]


Energy_Change = 2.28
Energy_min = 1.088
mask1 = vEline1 >= Energy_Change
vEline1 = vEline1[mask1]
vFluxlim1 = vFluxlim1[mask1]


#if you want to exclude the first masses
start=0
File_fluxlimit2 = np.loadtxt("/home/jortecal/GitHub/eRosita/Full_Range_24_dblpwl/Results/CombinedProfiles/CombinedBounds.dat",skiprows=1)
vEline2 = File_fluxlimit2[:,0]
vFluxlim2 = File_fluxlimit2[:,1]
vEline2 = vEline2[start:]
vFluxlim2 = vFluxlim2[start:]

mask2 = vEline2 < Energy_Change
vEline2 = vEline2[mask2]
vFluxlim2 = vFluxlim2[mask2]

mask3 = vEline2 >= Energy_min
vEline2 = vEline2[mask3]
vFluxlim2 = vFluxlim2[mask3]

# -------------------------------------------------------------------------
# Apply TBABS nH correction: F_new = F_old * ratio(E)
# ratio = TBABS(0.22)/TBABS(0.145), so F_new (nH=0.145) = F_old * ratio
# -------------------------------------------------------------------------
vFluxlim1 = vFluxlim1 * corr_factor(vEline1)
vFluxlim2 = vFluxlim2 * corr_factor(vEline2)

vEline = np.concatenate([vEline2, vEline1])
vFluxlim = np.concatenate([vFluxlim2, vFluxlim1])
sort_idx = np.argsort(vEline)
vEline = vEline[sort_idx]
vFluxlim = vFluxlim[sort_idx]

# Sterile nu
vMdm_kev_1, vSinsq2theta_1 = FSinsq2theta(vEline1,vFluxlim1,Dfactor_wmask)
vMdm_kev_2, vSinsq2theta_2 = FSinsq2theta(vEline2,vFluxlim2,Dfactor_wmask)
vMdm_kev, vSinsq2theta = FSinsq2theta(vEline,vFluxlim,Dfactor_wmask)

# ALP
vMdm_kev_alp_1, vgagg_1 = Fgagg(vEline1,vFluxlim1,Dfactor_wmask)
vMdm_kev_alp_2, vgagg_2 = Fgagg(vEline2,vFluxlim2,Dfactor_wmask)
vMdm_kev_alp, vgagg = Fgagg(vEline,vFluxlim,Dfactor_wmask)

# Tau
vMdm_kev_tau, vbound_tau = bound_tau(vEline, vFluxlim, Dfactor_wmask, nphoton=1)

############################################
########### PLOT STERILE NU################
#######################################################

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()

data = np.loadtxt("/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/2102.02207_Fig3top.txt",skiprows=0,delimiter=",")
XMM = interp1d(np.log10(data[:,0]), np.log10(data[:,1]), bounds_error=False, fill_value=-1)
# Above 5 keV
#ax1.plot(data[:,0], data[:,1], color='orange', linestyle='-')

# subdominant for E<9 keV wrt XMM
data = np.loadtxt("/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/2207.04572_Fig6_blue+red.txt",skiprows=0,delimiter=",")
# Above 6 keV
Nustar = interp1d(np.log10(data[:,0]), np.log10(data[:,1]), bounds_error=False, fill_value=-1)
#ax1.plot(data[:,0], data[:,1], color='blue', linestyle='-')

# Horiuchi et al. Chandra M31
data = np.loadtxt("/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/1311.0282_Fig4.txt",skiprows=0,delimiter=",")
# Above \sim 1 keV. At 4 keV bound on sinsq2theta is 1e-9
M31 = interp1d(np.log10(data[:,0]), np.log10(data[:,1]), bounds_error=False, fill_value=-1)
#ax1.plot(data[:,0], data[:,1], color='magenta', linestyle='-')

#Loewenstein, Kusenko, Biermann Suzaku URSA MINOR dwarf galaxy
data = np.loadtxt("/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/0812.2710_Fig8+9.dat",skiprows=0,delimiter=",")
# Above \sim 1 keV. At 4 keV bound less stringent than M31
Suzaku = interp1d(np.log10(data[:,0]), np.log10(data[:,1]), bounds_error=False, fill_value=-1)
#ax1.plot(data[:,0], data[:,1], color='red', linestyle='-')
data = np.loadtxt('/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/2405.17861_Fig1.txt',skiprows=0,delimiter=" ")
Nustar2025 = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]), bounds_error=False, fill_value=-1)

data = np.loadtxt("/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/2401.16747_Fig6_yellow_new.csv")
eFEDS = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]), bounds_error=False, fill_value=-1)



def totalXrays(x):
    # Evaluate all interpolation functions at x
    # Make sure x is a NumPy array to handle vectorization
    x = np.atleast_1d(x)
    values = np.vstack([
        XMM(x),
        Nustar(x),
        M31(x),
        Suzaku(x),
        Nustar2025(x),
    ])
    # Take the minimum along the first axis (i.e., across the functions)
    min_values = np.min(values, axis=0)
    
    # Return scalar if input was scalar
    return min_values[0] if np.isscalar(x) else min_values

Mass_bound_Sinsq2theta = np.logspace(np.log10(1.1), np.log10(20.), 300)
bound_Sinsq2theta = np.pow(10.,totalXrays(np.log10(Mass_bound_Sinsq2theta)))
#print(y)
ax1.plot(Mass_bound_Sinsq2theta, bound_Sinsq2theta, color='lightgray', linestyle='-')
plt.fill_between(Mass_bound_Sinsq2theta,bound_Sinsq2theta,y2=1e-5,facecolor='lightgray',zorder=-1)

ax1.plot(data[:, 0], data[:, 1], color='mediumseagreen', linestyle='-', label='eFEDS', linewidth=1)
ax1.plot(vMdm_kev, vSinsq2theta, color='darkviolet', label='LMC DR1', linewidth=2)
#ax1.plot(vMdm_kev_1, vSinsq2theta_1, color='black',linestyle='dashed')
#ax1.plot(vMdm_kev_2, vSinsq2theta_2, color='gray',linestyle='dotted')

ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$m_{s}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$\sin^2(2\theta)$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')

# Add a major tick at 2 keV
ax1.set_xticks([2, 3, 5, 10, 20])
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())

ax1.set_ylim(2e-12,1e-7)
ax1.set_xlim(1.9,20.)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Bounds_SterileNu_corrected.pdf')
#######################################################
    
############################################
########### PLOT STERILE ALP################
#######################################################

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()
    

Mass_bound_Sinsq2theta, bound_gagg = FSinsq2theta_gagg(Mass_bound_Sinsq2theta, bound_Sinsq2theta)
ax1.plot(Mass_bound_Sinsq2theta, bound_gagg, color='lightgray', linestyle='-')
plt.fill_between(Mass_bound_Sinsq2theta,bound_gagg,y2=1e-5,facecolor='lightgray',zorder=-1)
 
#vMdm_kev_alp, vgagg
ax1.plot(vMdm_kev_alp, vgagg, color='darkviolet', label='LMC DR1', linewidth=2)
#ax1.plot(vMdm_kev_alp_1, vgagg_1, color='black',linestyle='dashed')
#ax1.plot(vMdm_kev_alp_2, vgagg_2, color='gray',linestyle='dotted')
 
 
ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$m_{a}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$g_{a\gamma}\, [{\rm GeV}^{-1}]$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')


# Add a major tick at 2 keV
ax1.set_xticks([2, 3, 5, 10, 20])
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())


ax1.set_ylim(1e-18,1e-16)
ax1.set_xlim(1.9,20.)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Bounds_ALP_corrected.pdf')
#######################################################
    

############################################
########### PLOT FLUX################
#######################################################

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()


ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'Energy\, [keV]', size=30)
ax1.set_ylabel(r'$\phi\, [{\rm photons\,}{\rm cm}^{-2} {\rm s}^{-1}]$', size=30)
ax1.set_yscale('log')
ax1.set_xscale('log')


ax1.plot(vEline, vFluxlim, color='darkviolet', label='LMC DR1', linewidth=2)

#x = np.array(vEline)
#y = np.array(vFluxlim)
## Create dense x grid
#x_smooth = np.linspace(x.min(), x.max(), 500)
## Cubic spline
#spline = make_interp_spline(x, y, k=3)
#y_smooth = spline(x_smooth)
#ax1.plot(x_smooth, y_smooth, color='black')

#y_smooth = savgol_filter(np.log10(y), window_length=11, polyorder=3)
#ax1.plot(x, 10.**y_smooth, color='red')



# Add a major tick at 2 keV
#ax1.set_xticks([2, 3, 5, 10, 20])
#ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
#ax1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())


ax1.set_xticks([1, 2, 4, 6, 9])
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
ax1.set_ylim(1e-5,1e-1)
ax1.set_xlim(0.9,10.)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Bounds_flux_corrected.pdf')
#######################################################


############################################
########### PLOT LIFETIME ################
#######################################################

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()


ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$m_{\rm DM}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$\tau_{\rm DM}\, [{\rm s}]$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')



Mass_bound_Sinsq2theta, bound_tau = FSinsq2theta_tau(Mass_bound_Sinsq2theta, bound_Sinsq2theta)
ax1.plot(Mass_bound_Sinsq2theta, bound_tau, color='lightgray', linestyle='-')
plt.fill_between(Mass_bound_Sinsq2theta,bound_tau,y2=1e-30,facecolor='lightgray',zorder=-1)

ax1.plot(vMdm_kev_tau, vbound_tau, color='darkviolet', label='LMC DR1', linewidth=2)


# Add a major tick at 2 keV
ax1.set_xticks([2, 3, 5, 10, 20])
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())


ax1.set_ylim(1e25,4e29)
ax1.set_xlim(1.9,20.)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Bounds_tau_corrected.pdf')
#######################################################

'''
Here we save the limits in txt files for sterile nu, ALP and lifetime
'''
# # Sterile nu
output_data = np.column_stack((vMdm_kev, vSinsq2theta))
np.savetxt('Bounds_SterileNu_LMCDR1_corrected.txt', output_data, header='Mass_keV   sin2(2theta)', fmt=['%.6e', '%.6e'])
# # ALP
output_data = np.column_stack((vMdm_kev_alp, vgagg))
np.savetxt('Bounds_ALP_LMCDR1_corrected.txt', output_data, header='Mass_keV gagg_GeV-1', fmt=['%.6e', '%.6e'])
# # ALP Literature
output_data = np.column_stack((Mass_bound_Sinsq2theta, bound_gagg))
np.savetxt('Bounds_ALP_Literature_corrected.txt', output_data, header='Mass_keV   gagg_GeV-1', fmt=['%.6e', '%.6e'])
# # Lifetime
output_data = np.column_stack((vMdm_kev_tau, vbound_tau))
np.savetxt('Bounds_tau_LMCDR1_corrected.txt', output_data, header='Mass_keV   tau_s', fmt=['%.6e', '%.6e'])
