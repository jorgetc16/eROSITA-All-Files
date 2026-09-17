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


# D [GeV cm-2]
# M [GeV]
# tau [s]
#Out: cm-2 s-1
def FluxLine(Mdm, Dfactor, tau, nphoton=1):
    return Dfactor/(4*(np.pi)*Mdm*tau)*nphoton



prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}' #package mathpazo siunitx
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') #serif
plt.rcParams['axes.linewidth'] = 2



print('#######################')
print(' Welcome')
print('#######################')
#https://docs.gammapy.org/dev/api/gammapy.astro.darkmatter.JFactory.html
#https://docs.gammapy.org/dev/tutorials/api/astro_dark_matter.html#j-factors

# FROM JORGE, coordinates center of LMC map 80.89417, -69.75611
LMC_RA, LMC_DEC = 80.89417, -69.75611

# DM density profile from
#https://arxiv.org/pdf/2106.08025.pdf
profile = profiles.NFWProfile(r_s=9.8 * u.kpc)
profiles.DMProfile.DISTANCE_GC = 50. * u.kpc
profiles.DMProfile.LOCAL_DENSITY = 0.00101897 * u.Unit("GeV / cm3")

profile.scale_to_local_density()
print("LOCAL_DENSITY:", profiles.DMProfile.LOCAL_DENSITY)
print("DISTANCE_GC:", profiles.DMProfile.DISTANCE_GC)

print("*******************************")


#https://python4astronomers.github.io/astropy/coordinates.html
#https://docs.gammapy.org/dev/api/gammapy.maps.WcsGeom.html
# If you put xradius=1.0 it is a test AGAINST TABLE IN 1 in https://arxiv.org/pdf/2210.09310
# if you put xradius=3 gives the D-factor in a 3 deg radius region. Check for following calculations
#3X3 degree frame
position = SkyCoord(ra=LMC_RA, dec=LMC_DEC,  frame="icrs", unit="deg")
geom = WcsGeom.create(binsz=0.05, skydir=position, width=9.0, frame="icrs")
jfactory = MTJFactory(geom=geom, profile=profile, distance=profiles.DMProfile.DISTANCE_GC,annihilation=False)

#Compute D factor integrated over the pixel (GeV cm-2) in all the map
start_time = time.monotonic()
jfact = jfactory.MTcompute_jfactor_los(ntheta=100, ndecade=1e5)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

jfact_map = WcsNDMap(geom=geom, data=jfact)#, unit=jfact.unit)
plt.figure()
ax = jfact_map.plot(cmap="viridis", norm=LogNorm(), add_cbar=True)
ra = ax.coords[0]
ra.set_format_unit('deg')

# Define a circle with a a given radius in deg and compute the D factor
xradius=3.0
sky_reg = CircleSkyRegion(center=position, radius=xradius * u.deg)
pix_reg = sky_reg.to_pixel(wcs=geom.wcs)
pix_reg.plot(ax=ax, facecolor="none", edgecolor="red", label="1 deg circle")
total_jfact = (pix_reg.to_mask().multiply(jfact).sum())
print("D-factor in ",xradius," deg circle assuming a "f"{profile.__class__.__name__} is {total_jfact:.3g} GeV cm-2")


plt.title(f"D-Factor [{jfact_map.unit}] GeV cm-2")
plt.savefig('Dfactor map.pdf')



xradius = 3.0 # circle outside around center in deg
#CHEESEMAP
#### DATAFILE LOADING#####################
# FROM JORGE, coordinates center 80.89417, -69.75611
path="/Users/marcotaoso/Documents/2024/eROSITA/FromJorge/LMC_test_3deg/"
filecheesemap = path+"cheesemask_LMC_Circle_masked.fits"

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
print("Fraction valid ",Totalpx_valid/Totalpx)
print("Omega area valid [deg^2]",Totalpx_valid*deltaOmegapx*pow(180/np.pi,2.) )
print("Omega area total [deg^2]",Totalpx*deltaOmegapx*pow(180/np.pi,2.) )

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
print("D-factor in ",xradius," deg circle assuming a "f"{profile.__class__.__name__} is {total_jfact:.3g} GeV cm-2")


#D-factor in  3.0  deg circle assuming a NFWProfile is 1.59e+20 GeV cm-2
#D-factor in  3.0  deg circle assuming a NFWProfile is 1.59e+20 GeV cm-2
#D-factor in  3.0  deg circle assuming a NFWProfile is 9.74e+19 GeV cm-2



datacounts_M = np.loadtxt("../CombinedBounds.dat",skiprows=0)
Eline = datacounts_M[:,0] # in keV
Fluxlim = datacounts_M[:,1] # in cm-2 s-1
#print("Eline ",Eline)
#print("Fluxlim ",Fluxlim)
xtau=1.0 #s
xDfactor = total_jfact # GeV cm-2  9.73e+19 GeV cm-2
vMdm_M=2*Eline*1e-6 # GeV # Mass= 2E_gamma
vFluxDM = FluxLine(vMdm_M, xDfactor, xtau, nphoton=1)
#print("vFluxDM ",vFluxDM)
vbound_M = xtau*vFluxDM/Fluxlim # in sec
#print("vbound ",vbound_M)
#print(np.column_stack((vMdm_M, vbound_M)))

# For sterile nu, see eq.1 of https://arxiv.org/pdf/1911.09120 and eq. 4.13 of https://arxiv.org/pdf/1602.04816
vSinsq2theta_M = 1./(vbound_M*1.389e-22)*pow(1./(vMdm_M*1e6),5.)
#print("vSinsq2theta ",vSinsq2theta_M)





plt.figure()
ax1 = plt.subplot()

ax1.plot(vMdm_M*1e6, vbound_M, color='mediumseagreen', label='')
#ax1.plot(vMdm_XSp*1e6, vbound_XSp, color='orange', label='')
#ax1.plot(vMdm_M2*1e6, vbound_M2, color='mediumseagreen', label='',linestyle=':')
#ax1.plot(vMdm_XSp2*1e6, vbound_XSp2, color='orange', label='',linestyle=':')

ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
#ax1.yaxis.get_ticklocs(minor=True)
#ax1.minorticks_on()
#ax1.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
#ax1.xaxis.get_ticklocs(minor=True)
#ax1.minorticks_on()
#ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax1.set_xlabel(r'$M_{\chi}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$\tau_{\chi} [{\rm s}]$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Bound.pdf')


fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()

data = np.loadtxt("/Users/marcotaoso/Documents/2024/eROSITA/Bounds_refs/2102.02207_Fig3top.txt",skiprows=0,delimiter=",")
XMM = interp1d(np.log10(data[:,0]), np.log10(data[:,1]), bounds_error=False, fill_value=-1)
# Above 5 keV
#ax1.plot(data[:,0], data[:,1], color='orange', linestyle=':')

data = np.loadtxt("/Users/marcotaoso/Documents/2024/eROSITA/Bounds_refs/2207.04572_Fig6_blue+red.txt",skiprows=0,delimiter=",")
# Above 6 keV
Nustar = interp1d(np.log10(data[:,0]), np.log10(data[:,1]), bounds_error=False, fill_value=-1)
#ax1.plot(data[:,0], data[:,1], color='blue', linestyle='-')

data = np.loadtxt("/Users/marcotaoso/Documents/2024/eROSITA/Bounds_refs/1311.0282_Fig4.txt",skiprows=0,delimiter=",")
# Above \sim 1 keV. At 4 keV bound on sinsq2theta is 1e-9
M31 = interp1d(np.log10(data[:,0]), np.log10(data[:,1]), bounds_error=False, fill_value=-1)
#ax1.plot(data[:,0], data[:,1], color='green', linestyle='-')

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

x = np.logspace(np.log10(1.1), np.log10(20.), 300)
y = np.pow(10.,totalXrays(np.log10(x)))
#print(y)
ax1.plot(x, y, color='lightgray', linestyle='-')
plt.fill_between(x,y,y2=1e-5,facecolor='lightgray',zorder=-1)

ax1.plot(vMdm_M*1e6, vSinsq2theta_M, color='mediumseagreen')

ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$M_{\chi}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$\sin^2(2\theta)$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(2e-14,1e-5)
ax1.set_xlim(1.0,20.)
#plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('BoundSterileNu_bounds.pdf')


#print("test",np.pow(10.,Suzaku(np.log10(3.0))))
