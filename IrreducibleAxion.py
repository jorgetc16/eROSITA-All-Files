import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root_scalar
import matplotlib as mpl
import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}' #package mathpazo siunitx
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') #serif
plt.rcParams['axes.linewidth'] = 2

eV_to_s_inv = 1.51927e15         # 1 eV = 1.51927e15 s^-1
GeV_to_s_inv = eV_to_s_inv * 1e9 # 1 GeV = 1.51927e24 s^-1
t_universe = 4.35e17    

# Path the LMC limits
LMC_file = "/home/jortecal/GitHub/eRosita/Bounds_ALP_LMCDR1_corrected.txt"
LMC_data = np.loadtxt(LMC_file)
ma_LMC = LMC_data[:, 0]          # mass [keV]
g_lim_LMC = LMC_data[:, 1]         # original limits on g under DM assumption [GeV^-1]

# Path to limits from literature
Literature_file = "/home/jortecal/GitHub/eRosita/Bounds_ALP_Literature.txt"
Literature_data = np.loadtxt(Literature_file)
ma_Literature = Literature_data[:, 0]          # mass [keV]
g_lim_Literature = Literature_data[:, 1]         # original limits on g under DM assumption [GeV^-1]

# Path to irreducible abundance data
irred_data = np.loadtxt("/home/jortecal/GitHub/eRosita/PhotoPhillic_TRH=5MeV.csv", delimiter=',')  

CRB_data = np.loadtxt("/home/jortecal/GitHub/eRosita/CRB.txt")

ma_CRB = CRB_data[:, 0]          # mass [keV]
g_lim_CRB = CRB_data[:, 1]         # limits on g from CRB [GeV^-1]
# Split into vectors
ma_grid = np.unique(irred_data[:, 0])
g_grid  = np.unique(irred_data[:, 1])

CMB_data = np.loadtxt("/home/jortecal/GitHub/eRosita/CMB.txt")

ma_CMB = CMB_data[:, 0]          # mass [keV]
g_lim_CMB = CMB_data[:, 1]         # limits on g from CMB [GeV^-1]
# Split into vectors
ma_grid = np.unique(irred_data[:, 0])
g_grid  = np.unique(irred_data[:, 1])

# Ensure grid sizes match
n_ma = len(ma_grid)
n_g  = len(g_grid)

# Reshape F values into a 2D table
# order must match how the file was generated: sorted by mass, then coupling
F_table = irred_data[:, 2].reshape((n_ma, n_g))

# 2D interpolator for Fa(m, g)
Fa_interp = RegularGridInterpolator((ma_grid, g_grid), F_table,
                                   bounds_error=False,
                                   fill_value=None)

def Fa(m, g):
    """
    Wrapper for fractional abundance interpolation.
    """
    pts = np.array([[m, g]])
    fa_val = Fa_interp(pts)[0]
    # If outside range, clip
    return max(fa_val, 0.0)

def tau_from_g(m_keV, g_gevinv):
    """
    Lifetime tau_a (seconds) for axion of mass m_keV and coupling g (GeV^-1).
    Units: m_keV in keV, g in GeV^-1.
    Formula: Gamma [GeV] = g^2 * m^3 / (64*pi)  with m in GeV
             tau [s] = 1 / (Gamma [GeV] * GeV_to_s_inv)
    """
    m_GeV = m_keV * 1e-6
    Gamma_GeV = (g_gevinv**2) * (m_GeV**3) / (64.0 * np.pi)
    # avoid division by zero
    if Gamma_GeV <= 0:
        return np.inf
    return 1.0 / (Gamma_GeV * GeV_to_s_inv)

def Fmax_local(m_keV, g_gevinv, g_DM):
    """
    RHS of Eq.(8): (tau_a / tau_DM_min) * exp(tU / tau_a).
    Here tau_DM_min is computed from the published g_DM (the limit
    on g assuming axion is all DM) by converting g_DM -> lifetime.
    """
    tau_a = tau_from_g(m_keV, g_gevinv)
    tau_DM_min = tau_from_g(m_keV, g_DM)
    # safety for extreme tau_a (if tau_a extremely large, exp(tU/tau_a)~1)
    # but keep full expression
    return (tau_a / tau_DM_min) * np.exp(t_universe / tau_a)

def eq8_rootfunc(g, m_keV, g_DM):
    """
    Function whose root corresponds to equality Fa(m,g) = Fmax_local(...).
    Returns Fa(m,g) - Fmax_local(m,g,g_DM).
    """
    # Fa wrapper expects scalar; if interpolation returns None or nan, handle gracefully
    fa_val = Fa(m_keV, g)
    rhs = Fmax_local(m_keV, g, g_DM)
    return fa_val - rhs

def find_irreducible_bounds(m_keV, g_DM, g_search_min=None, g_search_max=None, nscan=500):
    """
    Find both lower and upper bounds by finding all roots of eq8_rootfunc.
    Uses the equation: Fa(m,g) = (tau_a / tau_DM_min) * exp(tU / tau_a)
    
    Returns:
        (g_lower, g_upper): Lower and upper bounds on coupling g.
                            If only one root found, both are the same.
                            If no roots found, both are np.nan.
    """
    try:
        g_grid_min = g_grid.min()
        g_grid_max = g_grid.max()
    except NameError:
        g_grid_min, g_grid_max = 1e-20, 1e-5

    # Expand search range more aggressively
    if g_search_min is None:
        g_search_min = max(1e-28, g_grid_min * 0.01)
    if g_search_max is None:
        g_search_max = min(1e-9, g_grid_max * 100)
    if g_search_min <= 0:
        g_search_min = 1e-28

    # Ensure we search a wide enough range
    g_search_min = min(g_search_min, g_DM * 1e-6)
    g_search_max = max(g_search_max, g_DM * 1e2)

    g_scan = np.logspace(np.log10(g_search_min), np.log10(g_search_max), nscan)
    fvals = np.array([eq8_rootfunc(g, m_keV, g_DM) for g in g_scan])
    
    # Check if we have valid function values
    valid_mask = np.isfinite(fvals)
    if not np.any(valid_mask):
        print(f"Warning: No valid function values for m={m_keV:.3e} keV")
        return (np.nan, np.nan)
    
    # Find sign changes only where both neighboring values are valid
    sign_changes = []
    for i in range(len(fvals)-1):
        if valid_mask[i] and valid_mask[i+1]:
            if np.sign(fvals[i]) * np.sign(fvals[i+1]) <= 0 and fvals[i] != fvals[i+1]:
                sign_changes.append(i)
    
    if len(sign_changes) == 0:
        # Try a different strategy: look for minimum absolute value
        abs_fvals = np.abs(fvals)
        abs_fvals[~valid_mask] = np.inf
        min_idx = np.argmin(abs_fvals)
        if abs_fvals[min_idx] < 0.01:  # Close enough to zero
            return (g_scan[min_idx], g_scan[min_idx])
        return (np.nan, np.nan)
    
    roots = []
    for idx in sign_changes:
        g_lo, g_hi = g_scan[idx], g_scan[idx+1]
        try:
            sol = root_scalar(lambda gg: eq8_rootfunc(gg, m_keV, g_DM),
                              bracket=[g_lo, g_hi], method='brentq',
                              xtol=1e-25, rtol=1e-14, maxiter=500)
            if sol.converged:
                roots.append(sol.root)
        except Exception:
            # Try bisect as fallback
            try:
                sol = root_scalar(lambda gg: eq8_rootfunc(gg, m_keV, g_DM),
                                  bracket=[g_lo, g_hi], method='bisect',
                                  xtol=1e-25, rtol=1e-14, maxiter=500)
                if sol.converged:
                    roots.append(sol.root)
            except Exception:
                continue

    if len(roots) == 0:
        return (np.nan, np.nan)
    if len(roots) == 1:
        return (roots[0], roots[0])
    return (min(roots), max(roots))

# --- Compute both lower and upper bounds for LMC and Literature datasets ---

g_irreducible_LMC = np.zeros_like(ma_LMC)
g_eq8_LMC = np.zeros_like(ma_LMC)
g_irreducible_Literature = np.zeros_like(ma_Literature)
g_eq8_Literature = np.zeros_like(ma_Literature)

for i, m in enumerate(ma_LMC):
    g_DM = g_lim_LMC[i]
    if np.isnan(g_DM) or g_DM <= 0:
        g_irreducible_LMC[i] = np.nan
        g_eq8_LMC[i] = np.nan
        continue
    g_lo, g_hi = find_irreducible_bounds(m, g_DM)
    g_irreducible_LMC[i] = g_lo
    g_eq8_LMC[i] = g_hi

for i, m in enumerate(ma_Literature):
    g_DM = g_lim_Literature[i]
    if np.isnan(g_DM) or g_DM <= 0:
        g_irreducible_Literature[i] = np.nan
        g_eq8_Literature[i] = np.nan
        continue
    g_lo, g_hi = find_irreducible_bounds(m, g_DM)
    g_irreducible_Literature[i] = g_lo
    g_eq8_Literature[i] = g_hi

############################################
########### PLOT IRREDUCIBLE AXION ################
#######################################################
fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()


ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$m_{a}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$g_{a\gamma}\, [{\rm GeV}^{-1}]$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')

# Print the irreducible axion bounds LMC
print("LMC DR1 Irreducible Axion Bounds:")
for i in range(len(ma_LMC)):
    print(f"m = {ma_LMC[i]:.3e} keV, g_irreducible = {g_irreducible_LMC[i]:.3e} GeV^-1, g_eq8 = {g_eq8_LMC[i]:.3e} GeV^-1")
    
ax1.plot(ma_CRB, g_lim_CRB, color='darkgray',  linewidth=2,zorder=-1)
plt.fill_between(ma_CRB, g_lim_CRB, y2=1e-3, facecolor='darkgray')


ax1.plot(ma_CMB, g_lim_CMB, color='darkgray',  linewidth=2,zorder=-1)
plt.fill_between(ma_CMB, g_lim_CMB, y2=1e-3, facecolor='darkgray')

ax1.plot(ma_Literature, g_irreducible_Literature , color='lightgray', linestyle='-',zorder=-1)
ax1.plot(ma_Literature, g_eq8_Literature , color='lightgray', linestyle='-',zorder=-1)
plt.fill_between(ma_Literature,g_irreducible_Literature,y2=g_eq8_Literature,facecolor='lightgray')

ax1.plot(ma_LMC, g_irreducible_LMC, color='darkviolet', label='LMC DR1', linewidth=2)
ax1.plot(ma_LMC, g_eq8_LMC, color='darkviolet', linewidth=2)
plt.fill_between(ma_LMC,g_irreducible_LMC,y2=g_eq8_LMC,facecolor='darkviolet',alpha=0.5)



# Add a major tick at 2 keV
ax1.set_xticks([2, 3, 5, 10, 20])
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())


ax1.set_ylim(3e-14, 3e-11)
ax1.set_xlim(1.9,20.)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Bounds_IrreducibleAxion_corrected.pdf')
