import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
#import sys
#import matplotlib as mp
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion, RectangleSkyRegion
from matplotlib.colors import LogNorm
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

import numpy.ma as ma

import time
from datetime import timedelta

from gammapy.maps import Map

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

'''

print('#######################')
print(' Welcome')
print('#######################')
#https://docs.gammapy.org/dev/api/gammapy.astro.darkmatter.JFactory.html
#https://docs.gammapy.org/dev/tutorials/api/astro_dark_matter.html#j-factors

# FROM JORGE, coordinates center of LMC map 80.89417, -69.75611
LMC_RA, LMC_DEC = 80.26, -69.26

# DM density profile from
#https://arxiv.org/pdf/2106.08025.pdf
#profile = profiles.NFWProfile(r_s=9.8 * u.kpc)
#profiles.DMProfile.DISTANCE_GC = 50. * u.kpc
#profiles.DMProfile.LOCAL_DENSITY = 0.00101897 * u.Unit("GeV / cm3")

#https : // www . aanda . org/articles/aa/pdf/2024/12/aa51578 - 24. pdf
#I use second column from Table1

profile = profiles.NFWProfile(r_s=19.9526 * u.kpc)
profiles.DMProfile.DISTANCE_GC = 50. * u.kpc
profiles.DMProfile.LOCAL_DENSITY = 0.0041755 * u.Unit("GeV / cm3")

profile.scale_to_local_density()
print("LOCAL_DENSITY:", profiles.DMProfile.LOCAL_DENSITY)
print("DISTANCE_GC:", profiles.DMProfile.DISTANCE_GC)

print("*******************************")


#https://python4astronomers.github.io/astropy/coordinates.html
#https://docs.gammapy.org/dev/api/gammapy.maps.WcsGeom.html
# If you put xradius=1.0 it is a test AGAINST TABLE IN 1 in https://arxiv.org/pdf/2210.09310
# if you put xradius=3 gives the D-factor in a 3 deg radius region. Check for following calculations

#5X5 degree frame
position = SkyCoord(ra=LMC_RA, dec=LMC_DEC,  frame="icrs", unit="deg")
geom = WcsGeom.create(binsz=0.05, skydir=position, width=11.0, frame="icrs")
jfactory = MTJFactory(geom=geom, profile=profile, distance=profiles.DMProfile.DISTANCE_GC,annihilation=False)

#Compute D factor integrated over the pixel (GeV cm-2) in all the map
start_time = time.monotonic()
jfact = jfactory.MTcompute_jfactor_los(ntheta=200, ndecade=1e5)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

jfact_map = WcsNDMap(geom=geom, data=jfact)#, unit=jfact.unit)
plt.figure()
ax = jfact_map.plot(cmap="viridis", norm=LogNorm(), add_cbar=True)
ra = ax.coords[0]
ra.set_format_unit('deg')

# Define a circle with a a given radius in deg and compute the D factor
xradius=5.0
sky_reg = CircleSkyRegion(center=position, radius=xradius * u.deg)
pix_reg = sky_reg.to_pixel(wcs=geom.wcs)
pix_reg.plot(ax=ax, facecolor="none", edgecolor="red", label="1 deg circle")
total_jfact = (pix_reg.to_mask().multiply(jfact).sum())
print("D-factor in ",xradius," deg circle assuming a "f"{profile.__class__.__name__} is {total_jfact:.3g} GeV cm-2")


plt.title(f"D-Factor [{jfact_map.unit}] GeV cm-2")
plt.savefig('Dfactor map.pdf')



xradius = 5.0 # circle outside around center in deg
#CHEESEMAP
#### DATAFILE LOADING#####################
# FROM JORGE, coordinates center 80.89417, -69.75611
path="/Users/marcotaoso/Documents/2024/eROSITA/FromJorge/LMC_5deg/"
filecheesemap = path+"cheesemask_comb_LMC_rad5deg_rebin80.fits"

chmap_f = fits.open(filecheesemap)[0]
chmap = fits.open(filecheesemap)[0].data
wcs = WCS(chmap_f.header)
chmap_shape = chmap.shape
print("Cheese map shape ",chmap_shape) #
#print("NAXIS1 ",chmap_f.header['NAXIS1']," name ",chmap_f.header['CTYPE1'])
#print("NAXIS2 ",chmap_f.header['NAXIS2']," name ",chmap_f.header['CTYPE2'])
#print("RA_CEN DEC_CEN ",chmap_f.header['RA_CEN']," ",chmap_f.header['DEC_CEN'])
print("CDELT in cheese map in deg (see fits file) ",wcs.wcs.cdelt[0]," ",wcs.wcs.cdelt[1])
#print("CRVAL ",chmap_f.header['CRVAL1']," ",chmap_f.header['CRVAL2'])
#print("CRPIX ",chmap_f.header['CRPIX1']," ",chmap_f.header['CRPIX2'])


#Nmap = 200
#plt.close()
#ax = plt.subplot(projection=wcs)
#graf = ax.imshow(chmap,cmap='viridis')
#ra = ax.coords[0]
#ra.set_format_unit('deg')
#cbar = plt.colorbar(graf)
#ax.grid(color='white', ls='solid')
#ax.set_xlabel('RA')
#ax.set_ylabel('DEC')
#plt.savefig('Mask.png')

# SIMPLE EXAMPLE, BASICALLY 2=DEC 3=RA
#test = np.array([[1.2,2.3,4.5],[9.2,3.1,1.4]])
#print("test.shape ",test.shape)
#print("test", test)
# OUTPUT IS test.shape  (2, 3)
#test [[1.2 2.3 4.5] [9.2 3.1 1.4]]

#
# For all pixels of the cheesemask compute the angular distance from the center of the map
# Define also a mask: mask the pixels at distances beyond xradius
#
RA_target = LMC_RA
DEC_target= LMC_DEC
RA_target_rad = np.pi/180*RA_target
DEC_target_rad = np.pi/180*DEC_target
ny, nx = chmap.shape
#As you see ny refers to AXIS2 i.e. DEC
#print("ny ",ny," nx ",nx)
y_grid, x_grid = np.indices((ny, nx))
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()
#print("x_flat ",x_flat)
#print("y_flat ",y_flat)
#https://docs.astropy.org/en/stable/api/astropy.wcs.WCS.html#astropy.wcs.WCS.wcs_world2pix
world_coords = wcs.wcs_pix2world(x_flat, y_flat, 0)  # 0 for the 3rd axis (image plane)
ra_flat = world_coords[0]  # RA in degrees
dec_flat = world_coords[1]  # DEC in degrees
ra_rad = ra_flat*np.pi/180
dec_rad = dec_flat*np.pi/180
cos_angular_distance = np.sin(DEC_target_rad) * np.sin(dec_rad) + \
                            np.cos(DEC_target_rad) * np.cos(dec_rad) * np.cos(RA_target_rad - ra_rad)
# Clip values to avoid errors in arccos due to floating-point precision
cos_angular_distance = np.clip(cos_angular_distance, -1, 1)
angular_distances_rad = np.arccos(cos_angular_distance)
angular_distances_deg = np.degrees(angular_distances_rad)
# Define Mask: TRUE = 1 WHAT I WANT TO MASK
maskCirc_Bol = angular_distances_deg > xradius

# Define an array in log space from a minimum and max separation angles.
# For this array compute corresponding the D-factor along the los (for the specific angle)
# To be used in an interpolation
ntheta = 500
sepmin = np.abs(wcs.wcs.cdelt[0]/10.)*np.pi/180.
sepmax = xradius*1.2*np.pi/180.
#print("sepmin deg ",sepmin*180/(np.pi)," sepmax deg ",sepmax*180/(np.pi))
separationlist=np.logspace(np.log10(sepmin), np.log10(sepmax), ntheta)
jfactlos2 = jfactory.MTcompute_differential_jfactor_separation_los(separationlist, ndecade=1e5)

# Compute D factor of all the pixels in the cheesemask (regardeless they are 0 or 1)
# I can simply interpolate from the (separationlist,jfactlos2)
# DeltaOmega of 1 pixel in sr
deltaOmegapx = np.abs(wcs.wcs.cdelt[0]*wcs.wcs.cdelt[1])*np.pow(np.pi/180.0,2.) #sr
separationlist_deg = separationlist*180.0/np.pi
#print("mask shape ",maskCirc_Bol.shape, " ")
#print("Shapes ",angular_distances_deg.shape, " ",separationlist_deg.shape," ",jfactlos2.shape)
Dfactor_map =  np.interp(angular_distances_deg, separationlist_deg, jfactlos2)*deltaOmegapx

#True and False are equivalent to 1 and 0 and True means masked in ma.masked_arra
# In the cheesemap 0 is the value to be masked (see plot) which here I set to True
chmap_Bol = chmap == 0
## Total Mask
#t1 = np.array([True, False, True, False]) t2 = np.array([True, True, False, False])
# t1+t2 = [ True  True  True False]
# Therefore I can sum the mask
totalMask = maskCirc_Bol + chmap_Bol.flatten()


# reshape, you can see that the shape is the same as
maskCirc_Bol = maskCirc_Bol.reshape(ny, nx)
Dfactor_map = Dfactor_map.reshape(ny, nx)
totalMask = totalMask.reshape(ny, nx)
print("Maps shapes ",maskCirc_Bol.shape," ",Dfactor_map.shape," ",totalMask.shape)


# PLOT D factor map
#careful: here I have integrated on pixels and the number of pixels is different from the one of D-factor map produced before,
#therefore I don' have the same z-axis in the plot (values of D integrated in the pixel)
plt.close()
ax = plt.subplot(projection=wcs)
graf = ax.imshow(Dfactor_map,cmap='viridis')
ra = ax.coords[0]
ra.set_format_unit('deg')
cbar = plt.colorbar(graf)
ax.grid(color='white', ls='solid')
ax.set_xlabel('RA')
ax.set_ylabel('DEC')
plt.savefig('Dfactor_map.png')


# PLOT CIRCLE MAP
plt.close()
ax = plt.subplot(projection=wcs)
graf = ax.imshow(maskCirc_Bol,cmap='viridis')
ra = ax.coords[0]
ra.set_format_unit('deg')
cbar = plt.colorbar(graf)
ax.grid(color='white', ls='solid')
ax.set_xlabel('RA')
ax.set_ylabel('DEC')
plt.savefig('Mask_circle.png')

# PLOT AGAIN CHEESE MAP
plt.close()
ax = plt.subplot(projection=wcs)
graf = ax.imshow(chmap_Bol,cmap='viridis')
ra = ax.coords[0]
ra.set_format_unit('deg')
cbar = plt.colorbar(graf)
ax.grid(color='white', ls='solid')
ax.set_xlabel('RA')
ax.set_ylabel('DEC')
plt.savefig('Mask_cheese_again.png')

# PLOT TOTAL MASK
plt.close()
ax = plt.subplot(projection=wcs)
graf = ax.imshow(totalMask,cmap='viridis')
ra = ax.coords[0]
ra.set_format_unit('deg')
cbar = plt.colorbar(graf)
ax.grid(color='white', ls='solid')
ax.set_xlabel('RA')
ax.set_ylabel('DEC')
plt.savefig('Mask_total.png')


# Counts pixels and angular areas
Totalpx_masked =np.sum(totalMask.flatten())
Totalpx = chmap_shape[0]*chmap_shape[1]
Totalpx_valid = Totalpx - Totalpx_masked
print("Final mask: pixels masked ",Totalpx_masked)
print("Pixels valid ",Totalpx_valid)
print("Fraction valid (careful, consider square)",Totalpx_valid/Totalpx)
Omegavalid_deg2 = Totalpx_valid*deltaOmegapx*pow(180/np.pi,2.)
print("Omega area valid [deg^2]", Omegavalid_deg2)
print("Omega area total [deg^2]",Totalpx*deltaOmegapx*pow(180/np.pi,2.) )

DeltaOmegacircle_deg2 = 2*np.pi*(1.0-np.cos(xradius*np.pi/180))*pow(180/np.pi,2.)
print("Omega area in circle [deg^2]",DeltaOmegacircle_deg2,"  assuming a radius of ",xradius," deg ")
print("Fraction valid ",Omegavalid_deg2/DeltaOmegacircle_deg2)



#simple test
#print("test ",(chmap_shape[0]*chmap_shape[1]-np.sum(maskCirc_Bol))*deltaOmegapx*pow(180/np.pi,2.))

# MASK using ma.masked_array
#When an element of the mask is False, the corresponding element of the associated array
#is valid and is said to be unmasked. When an element of the mask is True,
#the corresponding element of the associated array is said to be masked (invalid).

# FIRST I compute the D-factor inside xradius to check that I have the same D-factor I find before
# at the beginning of the script
Dfactor_map_masked = ma.masked_array(data=Dfactor_map.flatten(), mask=maskCirc_Bol.flatten())
total_jfact = np.sum(Dfactor_map_masked)
#print("Dfactor_map_masked shape ",Dfactor_map_masked.shape)
print("D-factor in ",xradius," deg circle assuming a "f"{profile.__class__.__name__} is {total_jfact:.3g} GeV cm-2")

# NOW COMPUTE D-factor using the FULL MASK
Dfactor_map_masked = ma.masked_array(data=Dfactor_map.flatten(), mask=totalMask.flatten())
total_jfact = np.sum(Dfactor_map_masked)
#print("Dfactor_map_masked shape ",Dfactor_map_masked.shape)
print("D-factor with mask in ",xradius," deg circle assuming a "f"{profile.__class__.__name__} is {total_jfact:.3g} GeV cm-2")

'''


#https : // www . aanda . org/articles/aa/pdf/2024/12/aa51578 - 24. pdf
#I use second column from Table1
#D-factor in  5.0  deg circle assuming a NFWProfile is 6.48e+20 GeV cm-2
#D-factor with mask in  5.0  deg circle assuming a NFWProfile is 4.04e+20 GeV cm-2
#Omega area valid [deg^2] 48.619523671191935
#Omega area in circle [deg^2] 78.48998608178113   assuming a radius of  5.0  deg
#Fraction valid  0.6194360083149175

#D-factor in  3.0  deg circle assuming a NFWProfile is 1.59e+20 GeV cm-2
#D-factor in  3.0  deg circle assuming a NFWProfile is 1.59e+20 GeV cm-2
#D-factor in  3.0  deg circle assuming a NFWProfile is 9.74e+19 GeV cm-2



Dfactor_wmask = 4.04e20 #GeV cm-2


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
    
    
File_fluxlimit1 = np.loadtxt("/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_windows/FromJ/2to9_LineEfixed_IBFixLines2-9/CombinedProfiles/CombinedBounds.dat",skiprows=1)
vEline1 = File_fluxlimit1[:,0]
vFluxlim1 = File_fluxlimit1[:,1]


skip = 14
print("I start 2-9 keV file from ",vEline1[skip])
vEline1 = vEline1[skip:]
vFluxlim1 = vFluxlim1[skip:]


#if you want to exclude the first masses
start=2
File_fluxlimit2 = np.loadtxt("/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_windows/1-2keV_fixednHDM/output/Results_LineEfixed_IBFixLines2-9/CombinedProfiles/CombinedBounds.dat",skiprows=1)
vEline2 = File_fluxlimit2[:,0]
vFluxlim2 = File_fluxlimit2[:,1]
vEline2 = vEline2[start:]
vFluxlim2 = vFluxlim2[start:]


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

data = np.loadtxt("/Users/marcotaoso/Documents/2024/eROSITA/Bounds_refs/2102.02207_Fig3top.txt",skiprows=0,delimiter=",")
XMM = interp1d(np.log10(data[:,0]), np.log10(data[:,1]), bounds_error=False, fill_value=-1)
# Above 5 keV
#ax1.plot(data[:,0], data[:,1], color='orange', linestyle='-')

# subdominant for E<9 keV wrt XMM
data = np.loadtxt("/Users/marcotaoso/Documents/2024/eROSITA/Bounds_refs/2207.04572_Fig6_blue+red.txt",skiprows=0,delimiter=",")
# Above 6 keV
Nustar = interp1d(np.log10(data[:,0]), np.log10(data[:,1]), bounds_error=False, fill_value=-1)
#ax1.plot(data[:,0], data[:,1], color='blue', linestyle='-')

# Horiuchi et al. Chandra M31
data = np.loadtxt("/Users/marcotaoso/Documents/2024/eROSITA/Bounds_refs/1311.0282_Fig4.txt",skiprows=0,delimiter=",")
# Above \sim 1 keV. At 4 keV bound on sinsq2theta is 1e-9
M31 = interp1d(np.log10(data[:,0]), np.log10(data[:,1]), bounds_error=False, fill_value=-1)
#ax1.plot(data[:,0], data[:,1], color='magenta', linestyle='-')

#Loewenstein, Kusenko, Biermann Suzaku URSA MINOR dwarf galaxy
data = np.loadtxt("/Users/marcotaoso/Documents/2024/eROSITA/Bounds_refs/0812.2710_Fig8+9.dat",skiprows=0,delimiter=",")
# Above \sim 1 keV. At 4 keV bound less stringent than M31
Suzaku = interp1d(np.log10(data[:,0]), np.log10(data[:,1]), bounds_error=False, fill_value=-1)
#ax1.plot(data[:,0], data[:,1], color='red', linestyle='-')

def totalXrays(x):
    # Evaluate all interpolation functions at x
    # Make sure x is a NumPy array to handle vectorization
    x = np.atleast_1d(x)
    values = np.vstack([
        XMM(x),
        Nustar(x),
        M31(x),
        Suzaku(x)
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



lweROSITA=2

ax1.plot(vMdm_kev, vSinsq2theta, color='mediumseagreen',linewidth=lweROSITA)
#ax1.plot(vMdm_kev_1, vSinsq2theta_1, color='black',linestyle='dashed')
#ax1.plot(vMdm_kev_2, vSinsq2theta_2, color='gray',linestyle='dotted')

ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$M_{\nu_s}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$\sin^2(2\theta)$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')

# Add a major tick at 2 keV
ax1.set_xticks([2, 3, 5, 10, 20])
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())

ax1.set_ylim(2e-14,1e-7)#2e-12,1e-7
ax1.set_xlim(1.9,20.)
#plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Bounds_SterileNu.pdf')
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
ax1.plot(vMdm_kev_alp, vgagg, color='mediumseagreen',linewidth=lweROSITA)
#ax1.plot(vMdm_kev_alp_1, vgagg_1, color='black',linestyle='dashed')
#ax1.plot(vMdm_kev_alp_2, vgagg_2, color='gray',linestyle='dotted')
 
 
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


ax1.set_ylim(1e-18,1e-16)
ax1.set_xlim(1.9,20.)
#plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Bounds_ALP.pdf')
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
ax1.set_xlabel(r'$E_{\gamma}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$\Phi [{\rm photons\,}{\rm cm}^{-2} {\rm s}^{-1}]$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')


ax1.plot(vEline, vFluxlim, color='mediumseagreen',linewidth=lweROSITA)

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


ax1.set_ylim(1e-5,1e-1)
ax1.set_xlim(0.9,10.)
#plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Bounds_flux.pdf')
#######################################################


############################################
########### PLOT LIFETIME FOR DM -> gamma + X ################
#######################################################

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()


ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$M_{DM}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$\tau_{\chi\rightarrow \gamma + X}~~ [{\rm s}]$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')



Mass_bound_Sinsq2theta, bound_tau = FSinsq2theta_tau(Mass_bound_Sinsq2theta, bound_Sinsq2theta)
ax1.plot(Mass_bound_Sinsq2theta, bound_tau, color='lightgray', linestyle='-')
plt.fill_between(Mass_bound_Sinsq2theta,bound_tau,y2=1e-30,facecolor='lightgray',zorder=-1)

data = np.column_stack((Mass_bound_Sinsq2theta, bound_tau))
np.savetxt("Bound_tau_Literature.txt", data, fmt="%.6e", delimiter=' ')

ax1.plot(vMdm_kev_tau, vbound_tau, color='mediumseagreen',linewidth=lweROSITA)

data = np.column_stack((vMdm_kev_tau, vbound_tau))
np.savetxt("Bound_tau_OURS.txt", data, fmt="%.6e", delimiter=' ')

# Add a major tick at 2 keV
ax1.set_xticks([2, 3, 5, 10, 20])
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())


ax1.set_ylim(1e25,4e29)
ax1.set_xlim(1.9,20.)
#plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Bounds_tau.pdf')
#######################################################
