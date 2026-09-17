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


# D [GeV cm-2]
# M [GeV]
# tau [s]
# Out: cm-2 s-1
def FluxLine(Mdm, Dfactor, tau, nphoton=1):
    return Dfactor/(4*(np.pi)*Mdm*tau)*nphoton


# ALP decay: a -> 2γ
# Decay rate: Γ = (g_aγ)² * m_a³ / (64π)
# Lifetime: τ = 1/Γ = 64π / [(g_aγ)² * m_a³]
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
print(' Welcome')
print('#######################')

# FROM https://www.aanda.org/articles/aa/pdf/2024/12/aa51578-24.pdf
LMC_RA, LMC_DEC = 80.26, -69.26
RS_Paper = 10**4.3
Rho_Paper = 10**-2.47
Msun_pc3_to_GeV_cm3 = 38.11  
Rho_Paper = Rho_Paper * Msun_pc3_to_GeV_cm3

profile = profiles.NFWProfile(r_s=RS_Paper*10**(-3) * u.kpc, rho_s=Rho_Paper * u.Unit("GeV / cm3"))
profiles.DMProfile.DISTANCE_GC = 49.9 * u.kpc
rho_earth = profile(profiles.DMProfile.DISTANCE_GC).value

profiles.DMProfile.LOCAL_DENSITY = rho_earth * u.Unit("GeV / cm3")

profile.scale_to_local_density()
print("LOCAL_DENSITY:", profiles.DMProfile.LOCAL_DENSITY)
print("DISTANCE_GC:", profiles.DMProfile.DISTANCE_GC)

print("*******************************")

position = SkyCoord(ra=LMC_RA, dec=LMC_DEC, frame="icrs", unit="deg")
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

xradius = 5.0
sky_reg = CircleSkyRegion(center=position, radius=xradius * u.deg)
pix_reg = sky_reg.to_pixel(wcs=geom.wcs)
pix_reg.plot(ax=ax, facecolor="none", edgecolor="red", label="5 deg circle")
total_jfact = (pix_reg.to_mask().multiply(jfact).sum())
print(f"D-factor in {xradius} deg circle: {total_jfact:.3g} GeV cm^-2")

plt.title(f"D-Factor [{jfact_map.unit}] GeV cm-2")
plt.savefig('Dfactor map_pm20.pdf')

# Load cheese map and compute masked D-factor
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

# Compute masked D-factor
Dfactor_map_masked = ma.masked_array(data=Dfactor_map.flatten(), mask=totalMask.flatten())
total_jfact = np.sum(Dfactor_map_masked)
print(f"D-factor (masked) in {xradius} deg circle: {total_jfact:.3g} GeV cm^-2")

# Load flux limits
datacounts_M = np.loadtxt("/home/jortecal/GitHub/eRosita/LMC5Deg/ResultsNewData/2to9/Results/CombinedProfiles/CombinedBounds.dat", skiprows=0)
Eline = datacounts_M[:, 0]  # keV
Fluxlim = datacounts_M[:, 1]  # cm^-2 s^-1

xDfactor = total_jfact  # GeV cm^-2

# =============================================================================
# STERILE NEUTRINO ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("STERILE NEUTRINO ANALYSIS")
print("="*60)

xtau = 1.0  # s
vMdm_M = 2 * Eline * 1e-6  # GeV
vFluxDM = FluxLine(vMdm_M, xDfactor, xtau, nphoton=1)
vbound_M = xtau * vFluxDM / Fluxlim  # s
vSinsq2theta_M = 1./(vbound_M*1.389e-22) * pow(1./(vMdm_M*1e6), 5.)

# Plot sterile neutrino bound
fig = plt.figure(figsize=(8, 7))
ax1 = plt.subplot()

# Load comparison data
data = np.loadtxt("/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/2102.02207_Fig3top.txt", skiprows=0, delimiter=",")
XMM = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]), bounds_error=False, fill_value=-1)

data = np.loadtxt("/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/2207.04572_Fig6_blue+red.txt", skiprows=0, delimiter=",")
Nustar = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]), bounds_error=False, fill_value=-1)

data = np.loadtxt("/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/1311.0282_Fig4.txt", skiprows=0, delimiter=",")
M31 = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]), bounds_error=False, fill_value=-1)

data = np.loadtxt("/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/0812.2710_Fig8+9.dat", skiprows=0, delimiter=",")
Suzaku = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]), bounds_error=False, fill_value=-1)

data = pd.read_csv("/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/2401.16747_Fig6_yellow.csv", delim_whitespace=True, skiprows=1).values
eFEDS = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]), bounds_error=False, fill_value=-1)

# data_new = pd.read_csv("/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/2401.16747_Fig6_yellow_new.csv", delim_whitespace=True, skiprows=1).values
# eFEDS_new = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]), bounds_error=False, fill_value=-1)

def totalXrays(x):
    x = np.atleast_1d(x)
    values = np.vstack([XMM(x), Nustar(x), M31(x), Suzaku(x)])
    min_values = np.min(values, axis=0)
    return min_values[0] if np.isscalar(x) else min_values

x = np.logspace(np.log10(1.1), np.log10(20.), 300)
y = np.pow(10., totalXrays(np.log10(x)))
ax1.plot(x, y, color='lightgray', linestyle='-')
plt.fill_between(x, y, y2=1e-5, facecolor='lightgray', zorder=-1)

ax1.plot(data[:, 0], data[:, 1], color='mediumseagreen', linestyle='-', label='eFEDS', linewidth=1)
# ax1.plot(data_new[:, 0], data_new[:, 1], color='blue', linestyle='-', label='eFEDS_new', linewidth=1)
ax1.plot(vMdm_M*1e6, vSinsq2theta_M, color='purple', label='Our Work (LMC)', linewidth=2)

ax1.tick_params(which='major', direction='in', width=1, length=10, top=True, right=True, pad=10)
ax1.tick_params(which='minor', axis='y', direction='in', width=1, length=7, top=True, right=True, pad=10)
ax1.tick_params(which='minor', axis='x', direction='in', width=1, length=7, top=True, right=True, pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$M_{\chi}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$\sin^2(2\theta)$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(1e-14, 2e-8)
ax1.set_xlim(1.0, 20.)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('BoundSterileNu_bounds_pm20.pdf')
print("✓ Saved: BoundSterileNu_bounds_pm20.pdf")

# =============================================================================
# AXION-LIKE PARTICLE (ALP) ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("AXION-LIKE PARTICLE ANALYSIS")
print("="*60)

# ALP mass = 2 * photon energy (a -> γγ)
vMa_GeV = 2 * Eline * 1e-6  # GeV

# For each mass, solve for g_aγ limit from observed flux
# Flux = D * (g_aγ)² * m_a² / (128 π² ℏ)
# g_aγ² = Flux * 128 π² ℏ / (D * m_a²)
hbar_GeV_s = 6.582119569e-25
vgag_limit = np.sqrt(Fluxlim * 128.0 * np.pi**2 * hbar_GeV_s / (xDfactor * vMa_GeV**2))

print(f"ALP coupling limits computed for masses {vMa_GeV[0]*1e6:.2f} - {vMa_GeV[-1]*1e6:.2f} keV")

# Plot g_aγ vs m_a
fig = plt.figure(figsize=(8, 7))
ax2 = plt.subplot()

# Load X-ray ALP bounds and create envelope
data_xray = np.loadtxt("/home/jortecal/GitHub/eRosita/XRayALP.txt", skiprows=0)
# Format: mass [keV], g_aγ [GeV^-1]
data_xray[:, 0] = data_xray[:, 0]/1e3
# Create interpolation function for X-ray bounds
XRayALP = interp1d(np.log10(data_xray[:, 0]), np.log10(data_xray[:, 1]), 
                   bounds_error=False, fill_value=-1)

# Create combined X-ray envelope
x_alp = np.logspace(np.log10(min(data_xray[:, 0])), np.log10(max(data_xray[:, 0])), 300)
y_alp = np.pow(10., XRayALP(np.log10(x_alp)))

# Plot gray shaded region for X-ray bounds
ax2.plot(x_alp, y_alp, color='lightgray', linestyle='-')
plt.fill_between(x_alp, y_alp, y2=1e-5, facecolor='lightgray', zorder=-1, 
                 label='X-ray bounds')

# Load eROSITA early data release limits (2401.16747)
data_erosita = np.loadtxt("/home/jortecal/GitHub/eRosita/2401Limits.txt", skiprows=3)
# Format: Column 0 = mass [eV], Column 1 = g_aγ [GeV^-1]
erosita_mass_keV = data_erosita[:, 0] / 1000.0  # Convert eV to keV
erosita_gag = data_erosita[:, 1]  # Already in GeV^-1

# Sort by mass for proper plotting
sorted_indices = np.argsort(erosita_mass_keV)
erosita_mass_keV = erosita_mass_keV[sorted_indices]
erosita_gag = erosita_gag[sorted_indices]

# Plot eROSITA limit
ax2.plot(erosita_mass_keV, erosita_gag, color='mediumseagreen', linestyle='-', 
         label='eROSITA eFEDS (2401.16747)', linewidth=1.5)

# Plot your LMC result
ax2.plot(vMa_GeV*1e6, vgag_limit, color='purple', label='Our Work (LMC)', linewidth=2)

ax2.tick_params(which='major', direction='in', width=1, length=10, top=True, right=True, pad=10)
ax2.tick_params(which='minor', axis='y', direction='in', width=1, length=7, top=True, right=True, pad=10)
ax2.tick_params(which='minor', axis='x', direction='in', width=1, length=7, top=True, right=True, pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax2.set_xlabel(r'$m_a\, [{\rm keV}]$', size=24)
ax2.set_ylabel(r'$g_{a\gamma}\, [{\rm GeV}^{-1}]$', size=24)
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlim(1.0, 20.)
ax2.set_ylim(1e-19, 2e-17)


# Add grid for better readability
ax2.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.5)
ax2.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.3)

plt.legend(fontsize=15, loc='best')
plt.tight_layout()
plt.savefig('Bound_ALP_gagamma.pdf')
print("✓ Saved: Bound_ALP_gagamma.pdf")

# Print comparison at a reference mass
ref_mass_keV = 3.5  # keV
if ref_mass_keV >= vMa_GeV[0]*1e6 and ref_mass_keV <= vMa_GeV[-1]*1e6:
    our_gag_at_ref = np.interp(ref_mass_keV, vMa_GeV*1e6, vgag_limit)
    print(f"\nAt m_a = {ref_mass_keV} keV:")
    print(f"  Our limit: g_aγ < {our_gag_at_ref:.3e} GeV^-1")
    
    if ref_mass_keV >= erosita_mass_keV[0] and ref_mass_keV <= erosita_mass_keV[-1]:
        erosita_gag_at_ref = np.interp(ref_mass_keV, erosita_mass_keV, erosita_gag)
        print(f"  eROSITA eFEDS: g_aγ < {erosita_gag_at_ref:.3e} GeV^-1")
        improvement = erosita_gag_at_ref / our_gag_at_ref
        if improvement > 1:
            print(f"  Our limit is {improvement:.1f}× stronger")
        else:
            print(f"  eROSITA limit is {1/improvement:.1f}× stronger")
    
    # Also compare with X-ray bounds
    if ref_mass_keV >= x_alp[0] and ref_mass_keV <= x_alp[-1]:
        xray_gag_at_ref = np.interp(ref_mass_keV, x_alp, y_alp)
        if xray_gag_at_ref > 0 and np.isfinite(xray_gag_at_ref):
            print(f"  X-ray bounds: g_aγ < {xray_gag_at_ref:.3e} GeV^-1")


# Diagnostic: Compare scaling at reference mass
ref_mass_keV = 3.5
print("\n" + "="*60)
print("DIAGNOSTIC: Comparing Dataset Performance")
print("="*60)

# Debug: Print array ranges
print(f"\nData ranges:")
print(f"  Sterile ν mass range: {vMdm_M[0]*1e6:.2f} - {vMdm_M[-1]*1e6:.2f} keV")
print(f"  ALP mass range: {vMa_GeV[0]*1e6:.2f} - {vMa_GeV[-1]*1e6:.2f} keV")
print(f"  eROSITA sterile mass range: {data[:, 0][0]:.2f} - {data[:, 0][-1]:.2f} keV")
print(f"  eROSITA ALP mass range: {erosita_mass_keV[0]:.2f} - {erosita_mass_keV[-1]:.2f} keV")
print(f"  Reference mass: {ref_mass_keV} keV")

sterile_ratio = None
alp_ratio = None

# Sterile neutrino comparison
try:
    our_sterile_at_ref = np.interp(ref_mass_keV, vMdm_M*1e6, vSinsq2theta_M)
    efeds_sterile_at_ref = np.interp(ref_mass_keV, data[:, 0], data[:, 1])
    sterile_ratio = efeds_sterile_at_ref / our_sterile_at_ref
    print(f"\nSterile Neutrino at {ref_mass_keV} keV:")
    print(f"  LMC (our work): sin²(2θ) < {our_sterile_at_ref:.3e}")
    print(f"  eFEDS:          sin²(2θ) < {efeds_sterile_at_ref:.3e}")
    print(f"  eFEDS/LMC ratio: {sterile_ratio:.3f}")
    if sterile_ratio > 1:
        print(f"  → LMC is {sterile_ratio:.2f}× STRONGER")
    else:
        print(f"  → eFEDS is {1/sterile_ratio:.2f}× STRONGER")
except Exception as e:
    print(f"\n✗ Could not compute sterile ratio: {e}")

# ALP comparison
try:
    our_alp_at_ref = np.interp(ref_mass_keV, vMa_GeV*1e6, vgag_limit)
    efeds_alp_at_ref = np.interp(ref_mass_keV, erosita_mass_keV, erosita_gag)
    alp_ratio = efeds_alp_at_ref / our_alp_at_ref
    print(f"\nALP at {ref_mass_keV} keV:")
    print(f"  LMC (our work): g_aγ < {our_alp_at_ref:.3e} GeV⁻¹")
    print(f"  eFEDS:          g_aγ < {efeds_alp_at_ref:.3e} GeV⁻¹")
    print(f"  eFEDS/LMC ratio: {alp_ratio:.3f}")
    if alp_ratio > 1:
        print(f"  → LMC is {alp_ratio:.2f}× STRONGER")
    else:
        print(f"  → eFEDS is {1/alp_ratio:.2f}× STRONGER")
except Exception as e:
    print(f"\n✗ Could not compute ALP ratio: {e}")

# Expected relationship
print(f"\n" + "-"*60)
print("THEORETICAL EXPECTATION:")
print("-"*60)
if sterile_ratio is not None and alp_ratio is not None:
    expected_alp = np.sqrt(sterile_ratio)
    print(f"Since both use same datasets (eFEDS vs LMC):")
    print(f"  ALP_ratio should equal √(Sterile_ratio)")
    print(f"")
    print(f"  √(Sterile_ratio) = √({sterile_ratio:.3f}) = {expected_alp:.3f}")
    print(f"  Actual ALP_ratio = {alp_ratio:.3f}")
    print(f"")
    discrepancy = 100*(alp_ratio/expected_alp - 1)
    print(f"  Discrepancy: {discrepancy:.1f}%")
    if abs(discrepancy) < 10:
        print(f"  ✓ Consistent! (within 10%)")
    else:
        print(f"  ✗ INCONSISTENT - investigate formulas!")
else:
    print("Could not compute both ratios - check mass ranges overlap")

print("="*60)