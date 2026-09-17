#!/usr/bin/env python
"""
SignalDM_Gener_Compare.py

Comparison script: plots both the pre-corrected (TBABS nH=0.22) and
corrected (TBABS nH=0.145) limits on the same figures.

The correction factor comes from
    RefereeChecks/nH/Marco_nH/CorrectionFactor.txt
which gives the energy-dependent ratio:
    ratio(E) = TBABS(nH=0.22) / TBABS(nH=0.145)

Output: 4 PDF figures with _compare suffix, each showing both curves.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys

# ---------------------------------------------------------------------------
# matplotlib / LaTeX setup
# ---------------------------------------------------------------------------
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 2

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
Dfactor_wmask = 4.04e20   # GeV cm-2
t_Universe    = 4.35e17    # s

# ---------------------------------------------------------------------------
# Conversion / bound functions  (identical to SignalDM_Gener.py)
# ---------------------------------------------------------------------------

def FluxLine(Mdm, Dfactor, tau, nphoton=1):
    return Dfactor / (4.0 * np.pi * Mdm * tau) * nphoton


def bound_tau(Eline, Fluxlim, Dfactor, nphoton=1):
    vMdm_kev = Eline * 2.0
    vMdm_gev = vMdm_kev * 1e-6
    xtau = 1.0
    vFluxDM = FluxLine(vMdm_gev, Dfactor, xtau, nphoton)
    vbound_tau = xtau * vFluxDM / Fluxlim
    return vMdm_kev, vbound_tau


def FSinsq2theta(Eline, Fluxlim, Dfactor):
    vMdm_kev, vbound_tau = bound_tau(Eline, Fluxlim, Dfactor, nphoton=1)
    vSinsq2theta = 7.2e21 * pow(vbound_tau, -1.0) * pow(vMdm_kev, -5.0)
    return vMdm_kev, vSinsq2theta


def Fgagg(Eline, Fluxlim, Dfactor):
    vMdm_kev, vbound_tau = bound_tau(Eline, Fluxlim, Dfactor, nphoton=2)
    eVtosec = 1.51927e15
    vMdm_gev = vMdm_kev * 1e-6
    vGamma_GeV = 1.0 / vbound_tau * 1.0 / eVtosec * 1e-9
    vgagg = np.sqrt(64.0 * np.pi * vGamma_GeV / (vMdm_gev ** 3))
    return vMdm_kev, vgagg


def FSinsq2theta_gagg(vMdm_kev, vSinsq2theta):
    vMdm_gev = vMdm_kev * 1e-6
    vbound_tau = 7.2e21 * pow(vSinsq2theta, -1.0) * pow(vMdm_kev, -5.0)
    vbound_tau = vbound_tau * 2.0        # ALP: 2 photons vs 1
    eVtosec = 1.51927e15
    vGamma_GeV = 1.0 / vbound_tau * 1.0 / eVtosec * 1e-9
    vgagg = np.sqrt(64.0 * np.pi * vGamma_GeV / (vMdm_gev ** 3))
    return vMdm_kev, vgagg


def FSinsq2theta_tau(vMdm_kev, vSinsq2theta):
    vbound_tau = 7.2e21 * pow(vSinsq2theta, -1.0) * pow(vMdm_kev, -5.0)
    return vMdm_kev, vbound_tau


# ===================================================================
# 1.  Load the flux-limit data  (without correction)
# ===================================================================

# --- high-energy region (2to9 Fixed Lines) ---
f1 = np.loadtxt("/home/jortecal/GitHub/eRosita/2to9FixedLines/Results/CombinedProfiles/CombinedBounds.dat",
                skiprows=1)
raw_E1 = f1[:, 0]
raw_F1 = f1[:, 1]

# --- low-energy region (Full Range dbl pwl) ---
f2 = np.loadtxt("/home/jortecal/GitHub/eRosita/Full_Range_24_dblpwl/Results/CombinedProfiles/CombinedBounds.dat",
                skiprows=1)
raw_E2 = f2[:, 0]
raw_F2 = f2[:, 1]

# --- energy masks ---
Energy_Change = 2.28
Energy_min    = 1.088

mask1 = raw_E1 >= Energy_Change
E1_unc = raw_E1[mask1]
F1_unc = raw_F1[mask1]

mask2 = raw_E2 < Energy_Change
E2_unc = raw_E2[mask2]
F2_unc = raw_F2[mask2]
mask3 = E2_unc >= Energy_min
E2_unc = E2_unc[mask3]
F2_unc = F2_unc[mask3]

# Combined uncorrected arrays (sorted by energy)
E_all_unc = np.concatenate([E2_unc, E1_unc])
F_all_unc = np.concatenate([F2_unc, F1_unc])
srt = np.argsort(E_all_unc)
E_all_unc = E_all_unc[srt]
F_all_unc = F_all_unc[srt]

# ===================================================================
# 2.  Apply the TBABS nH correction
# ===================================================================
corr_data = np.loadtxt(
    "/home/jortecal/GitHub/eRosita/RefereeChecks/nH/Marco_nH/CorrectionFactor.txt")
corr_factor = interp1d(corr_data[:, 0], corr_data[:, 1],
                        bounds_error=False, fill_value=1.0)

# Corrected fluxes: F_corr = F_unc * ratio(E)
F1_cor = F1_unc * corr_factor(E1_unc)
F2_cor = F2_unc * corr_factor(E2_unc)

E_all_cor = E_all_unc   # energies unchanged
F_all_cor = F_all_unc * corr_factor(E_all_unc)

# ===================================================================
# 3.  Compute derived bounds for BOTH cases
# ===================================================================

# --- Sterile neutrino (sin^2 2θ) ---
M_sterile_unc, Sinsq_unc = FSinsq2theta(E_all_unc, F_all_unc, Dfactor_wmask)
M_sterile_cor, Sinsq_cor = FSinsq2theta(E_all_cor, F_all_cor, Dfactor_wmask)

# --- ALP (g_aγγ) ---
M_alp_unc, gagg_unc = Fgagg(E_all_unc, F_all_unc, Dfactor_wmask)
M_alp_cor, gagg_cor = Fgagg(E_all_cor, F_all_cor, Dfactor_wmask)

# --- Lifetime ---
M_tau_unc, tau_unc = bound_tau(E_all_unc, F_all_unc, Dfactor_wmask, nphoton=1)
M_tau_cor, tau_cor = bound_tau(E_all_cor, F_all_cor, Dfactor_wmask, nphoton=1)

# ===================================================================
# 4.  Literature comparison (same as SignalDM_Gener.py)
# ===================================================================
data = np.loadtxt(
    "/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/2102.02207_Fig3top.txt",
    skiprows=0, delimiter=",")
XMM = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]),
                bounds_error=False, fill_value=-1)

data = np.loadtxt(
    "/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/2207.04572_Fig6_blue+red.txt",
    skiprows=0, delimiter=",")
Nustar = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]),
                   bounds_error=False, fill_value=-1)

data = np.loadtxt(
    "/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/1311.0282_Fig4.txt",
    skiprows=0, delimiter=",")
M31 = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]),
                bounds_error=False, fill_value=-1)

data = np.loadtxt(
    "/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/0812.2710_Fig8+9.dat",
    skiprows=0, delimiter=",")
Suzaku = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]),
                   bounds_error=False, fill_value=-1)

data = np.loadtxt(
    "/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/2405.17861_Fig1.txt",
    skiprows=0, delimiter=" ")
Nustar2025 = interp1d(np.log10(data[:, 0]), np.log10(data[:, 1]),
                       bounds_error=False, fill_value=-1)

data_eFEDS = np.loadtxt(
    "/home/jortecal/GitHub/eRosita/JuliolTries/DM_XJorge/DM_XJorge/2401.16747_Fig6_yellow_new.csv")


def totalXrays(x):
    x = np.atleast_1d(x)
    vals = np.vstack([XMM(x), Nustar(x), M31(x), Suzaku(x), Nustar2025(x)])
    min_vals = np.min(vals, axis=0)
    return min_vals[0] if np.isscalar(x) else min_vals


Mass_lit = np.logspace(np.log10(1.1), np.log10(20.0), 300)
bound_Sinsq_lit = 10.0 ** totalXrays(np.log10(Mass_lit))

# Derived literature bounds for ALP and lifetime
_, bound_gagg_lit = FSinsq2theta_gagg(Mass_lit, bound_Sinsq_lit)
_, bound_tau_lit  = FSinsq2theta_tau(Mass_lit, bound_Sinsq_lit)

# ===================================================================
# 5.  Plotting  —  4 panels, each with both corrected & uncorrected
# ===================================================================

# ------ style helpers ------
def style_ax(ax, xlab, ylab):
    ax.tick_params(which='major', direction='in', width=1, length=10,
                   top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='y', direction='in', width=1,
                   length=7, top=True, right=True, pad=10)
    ax.tick_params(which='minor', axis='x', direction='in', width=1,
                   length=7, top=True, right=True, pad=10)
    ax.set_xlabel(xlab, size=24)
    ax.set_ylabel(ylab, size=24)
    ax.set_xscale('log')
    ax.set_yscale('log')
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(22)


# ===================================================================
# 5a.  STERILE NEUTRINO  —  sin²(2θ) vs m_s
# ===================================================================
fig, ax = plt.subplots(figsize=(8, 7))

# Literature
ax.fill_between(Mass_lit, bound_Sinsq_lit, 1e-5,
                facecolor='lightgray', zorder=-1)
ax.plot(Mass_lit, bound_Sinsq_lit, color='lightgray', linestyle='-')
ax.plot(data_eFEDS[:, 0], data_eFEDS[:, 1], color='mediumseagreen',
        linestyle='-', label='eFEDS', linewidth=1)

# Our bounds
ax.plot(M_sterile_unc, Sinsq_unc, color='darkorange', linestyle='--',
        label=r'LMC DR1 ($n_{\rm H}=0.22$)', linewidth=2)
ax.plot(M_sterile_cor, Sinsq_cor, color='darkviolet', linestyle='-',
        label=r'LMC DR1 ($n_{\rm H}=0.145$)', linewidth=2)

style_ax(ax, r'$m_{s}\, [{\rm keV}]$', r'$\sin^2(2\theta)$')
ax.set_xticks([2, 3, 5, 10, 20])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_ylim(2e-12, 1e-7)
ax.set_xlim(1.9, 20.0)
ax.legend(fontsize=13, loc='lower left')
fig.tight_layout()
fig.savefig('Bounds_SterileNu_compare.pdf')
print("Saved Bounds_SterileNu_compare.pdf")

# ===================================================================
# 5b.  ALP  —  g_{aγγ} vs m_a
# ===================================================================
fig, ax = plt.subplots(figsize=(8, 7))

ax.fill_between(Mass_lit, bound_gagg_lit, 1e-5,
                facecolor='lightgray', zorder=-1)
ax.plot(Mass_lit, bound_gagg_lit, color='lightgray', linestyle='-')

ax.plot(M_alp_unc, gagg_unc, color='darkorange', linestyle='--',
        label=r'LMC DR1 ($n_{\rm H}=0.22$)', linewidth=2)
ax.plot(M_alp_cor, gagg_cor, color='darkviolet', linestyle='-',
        label=r'LMC DR1 ($n_{\rm H}=0.145$)', linewidth=2)

style_ax(ax, r'$m_{a}\, [{\rm keV}]$', r'$g_{a\gamma}\, [{\rm GeV}^{-1}]$')
ax.set_xticks([2, 3, 5, 10, 20])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_ylim(1e-18, 1e-16)
ax.set_xlim(1.9, 20.0)
ax.legend(fontsize=14, loc='lower left')
fig.tight_layout()
fig.savefig('Bounds_ALP_compare.pdf')
print("Saved Bounds_ALP_compare.pdf")

# ===================================================================
# 5c.  FLUX  —  φ vs Energy
# ===================================================================
fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(E_all_unc, F_all_unc, color='darkorange', linestyle='--',
        label=r'$n_{\rm H}=0.22$', linewidth=2)
ax.plot(E_all_cor, F_all_cor, color='darkviolet', linestyle='-',
        label=r'$n_{\rm H}=0.145$ (corrected)', linewidth=2)

style_ax(ax, r'Energy\, [keV]',
         r'$\phi\, [{\rm photons\,}{\rm cm}^{-2} {\rm s}^{-1}]$')
ax.set_xticks([1, 2, 4, 6, 9])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_ylim(1e-5, 1e-1)
ax.set_xlim(0.9, 10.0)
ax.legend(fontsize=14)
fig.tight_layout()
fig.savefig('Bounds_flux_compare.pdf')
print("Saved Bounds_flux_compare.pdf")

# ===================================================================
# 5d.  LIFETIME  —  τ vs m_DM
# ===================================================================
fig, ax = plt.subplots(figsize=(8, 7))

ax.fill_between(Mass_lit, bound_tau_lit, 1e30,
                facecolor='lightgray', zorder=-1)
ax.plot(Mass_lit, bound_tau_lit, color='lightgray', linestyle='-')

ax.plot(M_tau_unc, tau_unc, color='darkorange', linestyle='--',
        label=r'LMC DR1 ($n_{\rm H}=0.22$)', linewidth=2)
ax.plot(M_tau_cor, tau_cor, color='darkviolet', linestyle='-',
        label=r'LMC DR1 ($n_{\rm H}=0.145$)', linewidth=2)

style_ax(ax, r'$m_{\rm DM}\, [{\rm keV}]$', r'$\tau_{\rm DM}\, [{\rm s}]$')
ax.set_xticks([2, 3, 5, 10, 20])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_ylim(1e25, 4e29)
ax.set_xlim(1.9, 20.0)
ax.legend(fontsize=13, loc='lower left')
fig.tight_layout()
fig.savefig('Bounds_tau_compare.pdf')
print("Saved Bounds_tau_compare.pdf")

print("\nAll comparison plots generated successfully.")
