import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion, RectangleSkyRegion
from matplotlib.colors import LogNorm
import pandas as pd
import numpy.ma as ma

import time
from datetime import timedelta

from gammapy.maps import Map

import profiles
from MTJFactory import MTJFactory

from gammapy.astro.darkmatter import (
    JFactory,
)

from gammapy.maps import WcsGeom, WcsNDMap


# ALP decay: a -> 2γ
# Decay rate: Γ = (g_aγ)² * m_a³ / (64π)
# Lifetime: τ = 1/Γ = 64π / [(g_aγ)² * m_a³]
# Flux: Φ = D / (4π * m_a * τ) * n_photons
# where m_a is in GeV, g_aγ is in GeV^-1, τ is in s, D is in GeV cm^-2

def FluxLine_ALP(ma_GeV, Dfactor, gag_GeVm1, nphoton=2):
    """
    ALP decay flux
    ma_GeV: ALP mass in GeV
    Dfactor: D-factor in GeV cm^-2
    gag_GeVm1: axion-photon coupling in GeV^-1
    nphoton: number of photons per decay (2 for a->γγ)
    Returns: flux in cm^-2 s^-1
    """
    # Decay width in GeV: Γ = (g_aγ)² * m_a³ / (64π)
    Gamma_GeV = (gag_GeVm1**2) * (ma_GeV**3) / (64.0 * np.pi)
    # Lifetime in s: τ = ℏ/Γ (ℏ = 6.582119569e-25 GeV·s)
    hbar_GeV_s = 6.582119569e-25
    tau_s = hbar_GeV_s / Gamma_GeV
    # Flux
    return Dfactor / (4.0 * np.pi * ma_GeV * tau_s) * nphoton


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 2


print('#######################')
print(' Welcome to ALP Analysis')
print('#######################')

# LMC parameters
LMC_RA, LMC_DEC = 80.26, -69.26
RS_Paper = 10**4.3  # pc
Rho_Paper = 10**-2.47  # Msun/pc³
Msun_pc3_to_GeV_cm3 = 7.14397  
Rho_Paper = Rho_Paper * Msun_pc3_to_GeV_cm3

profile = profiles.NFWProfile(r_s=RS_Paper*10**(-3) * u.kpc)
profiles.DMProfile.DISTANCE_GC = 49.9 * u.kpc
profiles.DMProfile.LOCAL_DENSITY = Rho_Paper * u.Unit("GeV / cm3")

profile.scale_to_local_density()
print("LOCAL_DENSITY:", profiles.DMProfile.LOCAL_DENSITY)
print("DISTANCE_GC:", profiles.DMProfile.DISTANCE_GC)

print("*******************************")

# Create geometry
position = SkyCoord(ra=LMC_RA, dec=LMC_DEC,  frame="icrs", unit="deg")
geom = WcsGeom.create(binsz=0.05, skydir=position, width=9.0, frame="icrs")
jfactory = MTJFactory(geom=geom, profile=profile, distance=profiles.DMProfile.DISTANCE_GC, annihilation=False)

# Compute D factor
start_time = time.monotonic()
jfact = jfactory.MTcompute_jfactor_los(ntheta=100, ndecade=1e5)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

jfact_map = WcsNDMap(geom=geom, data=jfact)
plt.figure()
ax = jfact_map.plot(cmap="viridis", norm=LogNorm(), add_cbar=True)
ra = ax.coords[0]
ra.set_format_unit('deg')

# Define region
xradius = 5.0
sky_reg = CircleSkyRegion(center=position, radius=xradius * u.deg)
pix_reg = sky_reg.to_pixel(wcs=geom.wcs)
pix_reg.plot(ax=ax, facecolor="none", edgecolor="red", label="5 deg circle")
total_jfact = (pix_reg.to_mask().multiply(jfact).sum())
print(f"D-factor in {xradius} deg circle: {total_jfact:.3g} GeV cm^-2")

plt.title(f"D-Factor [{jfact_map.unit}] GeV cm^-2")
plt.savefig('Dfactor_map_ALP.pdf')

# Load cheese map and compute masked D-factor (same as sterile neutrino code)
path = "/home/jortecal/GitHub/eRosita/LMC5DegEv/"
filecheesemap = path + "cheesemask_comb_LMC_rad5deg_rebin80.fits"

chmap_f = fits.open(filecheesemap)[0]
chmap = fits.open(filecheesemap)[0].data
wcs = WCS(chmap_f.header)
chmap_shape = chmap.shape
print("Cheese map shape", chmap_shape)

RA_target = LMC_RA
DEC_target = LMC_DEC
RA_target_rad = np.pi/180*RA_target
DEC_target_rad = np.pi/180*DEC_target
ny, nx = chmap.shape
y_grid, x_grid = np.indices((ny, nx))
x_flat = x_grid.flatten()
y_flat = y_grid.flatten()

world_coords = wcs.wcs_pix2world(x_flat, y_flat, 0)
ra_flat = world_coords[0]
dec_flat = world_coords[1]
ra_rad = ra_flat*np.pi/180
dec_rad = dec_flat*np.pi/180

cos_angular_distance = np.sin(DEC_target_rad) * np.sin(dec_rad) + \
                       np.cos(DEC_target_rad) * np.cos(dec_rad) * np.cos(RA_target_rad - ra_rad)
cos_angular_distance = np.clip(cos_angular_distance, -1, 1)
angular_distances_rad = np.arccos(cos_angular_distance)
angular_distances_deg = np.degrees(angular_distances_rad)

maskCirc_Bol = angular_distances_deg > xradius

ntheta = 500
sepmin = np.abs(wcs.wcs.cdelt[0]/10.)*np.pi/180.
sepmax = xradius*1.2*np.pi/180.
separationlist = np.logspace(np.log10(sepmin), np.log10(sepmax), ntheta)
jfactlos2 = jfactory.MTcompute_differential_jfactor_separation_los(separationlist, ndecade=1e5)

deltaOmegapx = np.abs(wcs.wcs.cdelt[0]*wcs.wcs.cdelt[1])*np.pow(np.pi/180.0, 2.)
separationlist_deg = separationlist*180.0/np.pi
Dfactor_map = np.interp(angular_distances_deg, separationlist_deg, jfactlos2)*deltaOmegapx

chmap_Bol = chmap == 0
totalMask = maskCirc_Bol + chmap_Bol.flatten()

maskCirc_Bol = maskCirc_Bol.reshape(ny, nx)
Dfactor_map = Dfactor_map.reshape(ny, nx)
totalMask = totalMask.reshape(ny, nx)

Dfactor_map_masked = ma.masked_array(data=Dfactor_map.flatten(), mask=totalMask.flatten())
total_jfact = np.sum(Dfactor_map_masked)
print(f"D-factor (masked) in {xradius} deg circle: {total_jfact:.3g} GeV cm^-2")

# Load flux limits from your data
datacounts_M = np.loadtxt("/home/jortecal/GitHub/eRosita/LMC5Deg/Results/Proves_MT_Script/CombinedProfiles/CombinedBounds.dat", skiprows=0)
Eline = datacounts_M[:, 0]  # keV
Fluxlim = datacounts_M[:, 1]  # cm^-2 s^-1

# ALP mass = 2 * photon energy (a -> γγ)
vMa_GeV = 2 * Eline * 1e-6  # GeV
xDfactor = total_jfact  # GeV cm^-2

# For each mass, compute the g_aγ limit
# Flux_observed = Flux_ALP(m_a, D, g_aγ)
# g_aγ_limit such that Flux_ALP = Fluxlim
# From Flux = D/(4π m_a τ) * 2, and τ = 64π / [(g_aγ)² m_a³] * ℏ
# Flux = D/(4π m_a) * (g_aγ)² m_a³ / (64π ℏ) * 2
# Flux = D * (g_aγ)² * m_a² / (128 π² ℏ)
# g_aγ² = Flux * 128 π² ℏ / (D * m_a²)

hbar_GeV_s = 6.582119569e-25
vgag_limit = np.sqrt(Fluxlim * 128.0 * np.pi**2 * hbar_GeV_s / (xDfactor * vMa_GeV**2))

print("ALP coupling limits computed")

# Plot g_aγ vs m_a
fig = plt.figure(figsize=(8, 7))
ax1 = plt.subplot()

# Plot your result
ax1.plot(vMa_GeV*1e6, vgag_limit, color='purple', label='Our Work (LMC)', linewidth=2)

# Add other ALP bounds if available (you'll need to add data files)
# Example: CAST, helioscopes, astrophysical bounds, etc.

ax1.tick_params(which='major', direction='in', width=1, length=10, top=True, right=True, pad=10)
ax1.tick_params(which='minor', axis='y', direction='in', width=1, length=7, top=True, right=True, pad=10)
ax1.tick_params(which='minor', axis='x', direction='in', width=1, length=7, top=True, right=True, pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$m_a\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$g_{a\gamma}\, [{\rm GeV}^{-1}]$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlim(1.0, 20.)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Bound_ALP_gagamma.pdf')

print("ALP analysis complete!")