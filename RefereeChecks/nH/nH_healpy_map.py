"""
Extract nH values from the HI4PI HEALPix map (NHI_HPX.fits)
within a 5 deg radius ROI centered on the LMC, interpolate onto the
cheesemask grid, apply the mask, and produce:
  - a gridded nH map (imshow) with mask applied
  - a histogram of nH values (masked only)
  - summary statistics (masked only)

Usage:
    python nH_healpy_map.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import healpy as hp
from scipy.interpolate import griddata


# ===========================================================================
# 1. Load HEALPix nH data
# ===========================================================================
filename_nh = "/home/jortecal/Downloads/NHI_HPX.fits"
hdul_nh = fits.open(filename_nh)
data_nh = hdul_nh[1].data
header_nh = hdul_nh[1].header
nside = header_nh["NSIDE"]

nh_vals = data_nh["NHI"]
hpx_index = data_nh["HPXINDEX"]

npix_full = hp.nside2npix(nside)
nh_map = np.zeros(npix_full)
nh_map[hpx_index] = nh_vals

print(f"NSIDE = {nside}, full-sky pixels = {npix_full}")
print(f"HEALPix resolution ≈ {hp.nside2resol(nside, arcmin=True):.1f} arcmin")
hdul_nh.close()

# ===========================================================================
# 2. Define ROI and query nH — now driven by the cheesemask grid extent
# ===========================================================================
lmc_ra, lmc_dec = 80.260, -69.260       # deg (kept for the 5° circle overlay)
radius_deg = 5.0

# We'll first determine the cheesemask grid extent, then query nH
# for all HEALPix pixels that fall anywhere within that bounding box.
# This replaces the previous 5° disc query and guarantees full coverage.

# ===========================================================================
# 3. Load cheesemask and its WCS
# ===========================================================================
filename_mask = ("/home/jortecal/GitHub/eRosita/LMC5DegEv/"
                 "cheesemask_comb_LMC_rad5deg_rebin80.fits")
hdul_mask = fits.open(filename_mask)
mask_img = hdul_mask[0].data.astype(np.float64)   # 1 = clean, 0 = masked
wcs = WCS(hdul_mask[0].header)

ny, nx = mask_img.shape
print(f"Cheesemask shape = ({ny}, {nx})")

# ---- DIAGNOSTIC: print WCS details ----
print(f"\nWCS CTYPE1 = {wcs.wcs.ctype[0]}")
print(f"WCS CTYPE2 = {wcs.wcs.ctype[1]}")
print(f"WCS CRPIX  = ({wcs.wcs.crpix[0]:.1f}, {wcs.wcs.crpix[1]:.1f})")
print(f"WCS CRVAL  = ({wcs.wcs.crval[0]:.4f}, {wcs.wcs.crval[1]:.4f})")

# Determine native coordinate frame from CTYPE
ctype1 = wcs.wcs.ctype[0]
if "RA" in ctype1 or "DEC" in ctype1:
    coord_frame = "equatorial"   # RA---TAN, DEC--TAN
elif "GLON" in ctype1 or "GLAT" in ctype1:
    coord_frame = "galactic"     # GLON-TAN, GLAT-TAN
elif "ELON" in ctype1 or "ELAT" in ctype1:
    coord_frame = "ecliptic"
else:
    coord_frame = "unknown"
print(f"Native coordinate frame: {coord_frame}")

# Compute pixel scale from WCS (deg/pixel)
if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None:
    pix_scale_x = np.sqrt(wcs.wcs.cd[0, 0]**2 + wcs.wcs.cd[0, 1]**2)
    pix_scale_y = np.sqrt(wcs.wcs.cd[1, 0]**2 + wcs.wcs.cd[1, 1]**2)
else:
    pix_scale_x = abs(wcs.wcs.cdelt[0])
    pix_scale_y = abs(wcs.wcs.cdelt[1])
print(f"Pixel scale ≈ ({pix_scale_x:.4f}, {pix_scale_y:.4f}) deg/pixel")

# ---- Get the RA/DEC extent from the four corners of the mask ----
# Corner pixels: (1,1), (1,nx), (ny,1), (ny,nx) — 1-based FITS
corner_ra, corner_dec = wcs.all_pix2world(
    [1,   1,   nx,  nx],
    [1,   ny,  1,   ny],
    0
)
ra_min, ra_max   = corner_ra.min(), corner_ra.max()
dec_min, dec_max = corner_dec.min(), corner_dec.max()
margin = 0.5  # small extra margin in degrees
ra_min  -= margin
ra_max  += margin
dec_min -= margin
dec_max += margin
print(f"\nCheesemask bounding box (with {margin}° margin):")
print(f"  RA  = [{ra_min:.2f}, {ra_max:.2f}]")
print(f"  DEC = [{dec_min:.2f}, {dec_max:.2f}]")

# ===========================================================================
# 4. Build coordinate grid from cheesemask WCS
#    We get BOTH native coords AND RA/DEC for every pixel
# ===========================================================================
y_idx, x_idx = np.mgrid[0:ny, 0:nx]
# astropy WCS all_pix2world uses 1-based FITS convention
native_lon, native_lat = wcs.all_pix2world(x_idx + 1, y_idx + 1, 0)

# If the WCS is NOT equatorial, also compute RA/DEC for each pixel
if coord_frame == "galactic":
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    native_coords = SkyCoord(l=native_lon * u.deg, b=native_lat * u.deg,
                             frame="galactic")
    ra_grid  = native_coords.icrs.ra.deg
    dec_grid = native_coords.icrs.dec.deg
    print("Converted Galactic → Equatorial for all grid pixels.")
elif coord_frame == "equatorial":
    ra_grid  = native_lon
    dec_grid = native_lat
    print("WCS is already in Equatorial RA/DEC.")
else:
    ra_grid  = native_lon
    dec_grid = native_lat
    print(f"WARNING: unknown coord frame '{coord_frame}', assuming RA/DEC.")

# Flatten for interpolation
native_lon_flat = native_lon.ravel()
native_lat_flat = native_lat.ravel()

# ---- DIAGNOSTIC: print grid extent ----
print(f"Native lon range: [{native_lon.min():.2f}, {native_lon.max():.2f}] deg")
print(f"Native lat range: [{native_lat.min():.2f}, {native_lat.max():.2f}] deg")
print(f"RA  range: [{ra_grid.min():.2f}, {ra_grid.max():.2f}] deg")
print(f"DEC range: [{dec_grid.min():.2f}, {dec_grid.max():.2f}] deg")

# ===========================================================================
# 5. Query nH for the FULL cheesemask bounding box
#    Use query_disc with a radius large enough to cover all corners.
# ===========================================================================
# Compute maximum angular distance from LMC centre to bounding box corners
corners_ra  = np.array([ra_min, ra_min, ra_max, ra_max])
corners_dec = np.array([dec_min, dec_max, dec_min, dec_max])
dc = np.radians(corners_dec - lmc_dec)
dr = np.radians(corners_ra - lmc_ra) * np.cos(np.radians(lmc_dec))
corner_dists = np.degrees(np.sqrt(dr**2 + dc**2))
disc_radius = corner_dists.max() + 0.2  # small safety margin
print(f"Max corner distance = {corner_dists.max():.2f}°, "
      f"using disc radius = {disc_radius:.2f}°")

# Query disc in Galactic coordinates
theta_center = np.radians(90.0 - lmc_dec)
phi_center   = np.radians(lmc_ra)
r_gal = hp.Rotator(coord=["C", "G"])
theta_gal_c, phi_gal_c = r_gal(theta_center, phi_center)
pix_roi = hp.query_disc(nside, hp.ang2vec(theta_gal_c, phi_gal_c),
                        np.radians(disc_radius))
nH_roi = nh_map[pix_roi]
print(f"nH pixels inside {disc_radius:.1f}° disc: {len(pix_roi)}")

# Convert HEALPix pixels → RA / DEC (Equatorial)
r_inv = hp.Rotator(coord=["G", "C"])
theta_gal_pix, phi_gal_pix = hp.pix2ang(nside, pix_roi)
theta_eq, phi_eq = r_inv(theta_gal_pix, phi_gal_pix)
ra_roi  = np.degrees(phi_eq)
dec_roi = np.degrees(np.pi / 2.0 - theta_eq)

# Convert to native coordinate frame for interpolation
if coord_frame == "galactic":
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    eq_coords = SkyCoord(ra=ra_roi * u.deg, dec=dec_roi * u.deg, frame="icrs")
    gal_coords = eq_coords.galactic
    lon_roi = gal_coords.l.deg
    lat_roi = gal_coords.b.deg
else:
    lon_roi = ra_roi
    lat_roi = dec_roi

# ---- DIAGNOSTIC: check coverage ----
print(f"nH lon range: [{lon_roi.min():.2f}, {lon_roi.max():.2f}]")
print(f"nH lat range: [{lat_roi.min():.2f}, {lat_roi.max():.2f}]")

# ===========================================================================
# 6. Interpolate nH onto cheesemask grid
# ===========================================================================
print("Interpolating nH onto cheesemask grid …")
nH_flat = griddata(
    (lon_roi, lat_roi), nH_roi,
    (native_lon_flat, native_lat_flat),
    method="linear",
    fill_value=np.nan,
)
nH_grid = nH_flat.reshape(ny, nx)

# ---- DIAGNOSTIC: how many grid cells got valid nH? ----
n_valid_interp = np.sum(~np.isnan(nH_grid))
print(f"Grid cells with valid nH (from interpolation): {n_valid_interp} "
      f"({100 * n_valid_interp / nH_grid.size:.1f}%)")

# ===========================================================================
# 7. Apply cheesemask — multiply by 0/1 mask
# ===========================================================================
nH_masked_grid = nH_grid * mask_img   # pixels where mask=0 become 0
nH_masked_grid[mask_img == 0] = np.nan  # set to NaN for stats/plotting

hdul_mask.close()

# ===========================================================================
# 8. Build DataFrame from masked pixels (non-zero mask pixels only)
# ===========================================================================
keep = (mask_img > 0) & (~np.isnan(nH_grid))  # clean mask + valid nH

nH_m_flat = nH_grid[keep]
ra_m_flat = ra_grid[keep]
dec_m_flat = dec_grid[keep]

# Angular distance from LMC centre
dra_m  = np.radians(ra_m_flat - lmc_ra) * np.cos(np.radians(lmc_dec))
ddec_m = np.radians(dec_m_flat - lmc_dec)
dist_m = np.degrees(np.sqrt(dra_m**2 + ddec_m**2))

df_masked = pd.DataFrame({"RA": ra_m_flat, "DEC": dec_m_flat,
                          "Dist": dist_m, "nH": nH_m_flat})

print(f"\nTotal cheesemask pixels: {mask_img.size}")
print(f"Clean mask pixels (mask = 1): {(mask_img > 0).sum()}")
print(f"Valid nH after masking: {len(df_masked)}")

# ===========================================================================
# 9. Summary statistics (masked only)
# ===========================================================================
print(f"\n===== MASKED nH STATISTICS =====")
print(df_masked.describe())
print(f"Mean nH (masked):  {df_masked['nH'].mean():.3e} cm⁻²")
print(f"Median nH (masked): {df_masked['nH'].median():.3e} cm⁻²")

df_5deg = df_masked[df_masked["Dist"] <= 5.0]
print(f"Mean nH within 5 deg (masked): {df_5deg['nH'].mean():.3e} cm⁻²")

# ===========================================================================
# 10. Plotting functions
# ===========================================================================

def plot_gridded_map(nH_array, ra_arr, dec_arr, title_suffix, outpath):
    """
    pcolormesh-based map using the actual RA/DEC grid (handles SIN
    projection non-linearity correctly).
    nH_array: 2D array (ny, nx) with NaN where masked/outside ROI.
    """
    # ---- Compute pixel edges from centres for pcolormesh ----
    ra_edges = np.empty((ra_arr.shape[0] + 1, ra_arr.shape[1] + 1))
    dec_edges = np.empty_like(ra_edges)

    # Interior edges: average of 4 neighbouring centres
    ra_edges[1:-1, 1:-1] = 0.25 * (
        ra_arr[:-1, :-1] + ra_arr[:-1, 1:] +
        ra_arr[1:, :-1] + ra_arr[1:, 1:]
    )
    dec_edges[1:-1, 1:-1] = 0.25 * (
        dec_arr[:-1, :-1] + dec_arr[:-1, 1:] +
        dec_arr[1:, :-1] + dec_arr[1:, 1:]
    )
    # Extrapolate boundary edges
    ra_edges[0, 1:-1]  = 2*ra_edges[1, 1:-1]  - ra_edges[2, 1:-1]
    ra_edges[-1, 1:-1] = 2*ra_edges[-2, 1:-1] - ra_edges[-3, 1:-1]
    ra_edges[1:-1, 0]  = 2*ra_edges[1:-1, 1]  - ra_edges[1:-1, 2]
    ra_edges[1:-1, -1] = 2*ra_edges[1:-1, -2] - ra_edges[1:-1, -3]
    dec_edges[0, 1:-1]  = 2*dec_edges[1, 1:-1]  - dec_edges[2, 1:-1]
    dec_edges[-1, 1:-1] = 2*dec_edges[-2, 1:-1] - dec_edges[-3, 1:-1]
    dec_edges[1:-1, 0]  = 2*dec_edges[1:-1, 1]  - dec_edges[1:-1, 2]
    dec_edges[1:-1, -1] = 2*dec_edges[1:-1, -2] - dec_edges[1:-1, -3]
    # Corners
    ra_edges[0, 0]   = 2*ra_edges[1, 1]   - ra_edges[2, 2]
    ra_edges[0, -1]  = 2*ra_edges[1, -2]  - ra_edges[2, -3]
    ra_edges[-1, 0]  = 2*ra_edges[-2, 1]  - ra_edges[-3, 2]
    ra_edges[-1, -1] = 2*ra_edges[-2, -2] - ra_edges[-3, -3]
    dec_edges[0, 0]   = 2*dec_edges[1, 1]   - dec_edges[2, 2]
    dec_edges[0, -1]  = 2*dec_edges[1, -2]  - dec_edges[2, -3]
    dec_edges[-1, 0]  = 2*dec_edges[-2, 1]  - dec_edges[-3, 2]
    dec_edges[-1, -1] = 2*dec_edges[-2, -2] - dec_edges[-3, -3]

    # Use the full cheesemask grid extent for plot limits
    plot_ra_min, plot_ra_max = ra_edges.min(), ra_edges.max()
    plot_dec_min, plot_dec_max = dec_edges.min(), dec_edges.max()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(
        ra_edges, dec_edges, nH_array,
        cmap="viridis",
        vmin=0.5e21, vmax=6.2e21,
        shading="flat",
    )
    plt.colorbar(im, ax=ax, label=r"$n_H$ [cm$^{-2}$]")

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("DEC [deg]")
    ax.invert_xaxis()
    ax.set_xlim(plot_ra_max, plot_ra_min)   # inverted
    ax.set_ylim(plot_dec_min, plot_dec_max)
    ax.set_aspect("auto")
    ax.set_title(f"HI column density around the LMC {title_suffix}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")
    plt.close(fig)


def plot_nH_distribution(df, outpath="nH_healpy_distribution.png"):
    """Histogram of masked nH values."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["nH"] / 1e21, bins=30, color="steelblue",
            edgecolor="white", alpha=0.8)
    ax.axvline(df["nH"].mean() / 1e21, color="red", linestyle="--",
               label=f"Mean: {df['nH'].mean():.2e}")
    ax.axvline(df["nH"].median() / 1e21, color="orange", linestyle=":",
               label=f"Median: {df['nH'].median():.2e}")
    ax.set_xlabel(r"$n_H$ [$10^{21}$ cm$^{-2}$]")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of HI column density around the LMC (masked)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")
    plt.close(fig)


# ===========================================================================
# 11. Generate all outputs
# ===========================================================================
# Unmasked nH map (interpolated, no mask applied)
plot_gridded_map(nH_grid, ra_grid, dec_grid,
                 "(interpolated, no mask)", "nH_healpy_grid_unmasked.png")

# Masked nH map (cheesemask × nH)
plot_gridded_map(nH_masked_grid, ra_grid, dec_grid,
                 "(cheesemask applied)", "nH_healpy_grid_masked.png")

plot_nH_distribution(df_masked)