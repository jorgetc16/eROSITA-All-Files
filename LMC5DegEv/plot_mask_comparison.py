#!/usr/bin/env python3
"""
Plot the eROSITA LMC emission map with and without the cheese mask.

This script reads the rebinned image and cheese mask (both eSASS products from
eROSITA DR1) and produces a side-by-side comparison to visually verify that
the mask successfully masks the brightest point sources.

The cheese mask convention from eSASS/ermask:
  0 = masked pixels (point sources)
  1 = unmasked pixels (diffuse emission / background)

Usage:
    python plot_mask_comparison.py

Output:
    LMC_mask_comparison.png  —  side-by-side figure
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
IMAGE_FILE = "/home/jortecal/GitHub/eRosita/LMC5DegEv/image_comb_LMC_rad5deg_rebin80.fits"
MASK_FILE  = "/home/jortecal/GitHub/eRosita/LMC5DegEv/cheesemask_comb_LMC_rad5deg_rebin80.fits"
OUTPUT_PNG = "/home/jortecal/GitHub/eRosita/LMC5DegEv/LMC_mask_comparison.png"

# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
print("Loading image ...")
with fits.open(IMAGE_FILE) as hdul:
    image_data = hdul[0].data.astype(np.float64)
    image_header = hdul[0].header
    wcs = WCS(image_header)

print("Loading cheese mask ...")
with fits.open(MASK_FILE) as hdul:
    mask_data = hdul[0].data.astype(np.float64)

print(f"  Image shape : {image_data.shape}")
print(f"  Mask shape  : {mask_data.shape}")
print(f"  Mask unique values: {np.unique(mask_data)}")

# ---------------------------------------------------------------------------
# 2. Build the masked image
# ---------------------------------------------------------------------------
# The cheese mask is 0 where sources are masked, 1 for good sky pixels.
# Set masked pixels to NaN so they appear blank in the map.
masked_image = image_data.copy()
masked_image[mask_data == 0] = np.nan

# Also build a binary mask for contour overlay (0 = masked, 1 = good)
# ---------------------------------------------------------------------------
# 3. Determine sensible colour stretches
# ---------------------------------------------------------------------------
# Panel 1 (raw image): stretch from ALL pixels, including bright sources,
# so the point sources that the mask will remove are clearly visible.
all_pixels = image_data[np.isfinite(image_data) & (image_data > 0)]
vmin_raw = np.percentile(all_pixels, 5)
vmax_raw = np.percentile(all_pixels, 99.5)
norm_raw = LogNorm(vmin=vmin_raw, vmax=vmax_raw)

# Panel 3 (masked image): stretch from UNMASKED pixels only, so the
# diffuse emission / background is well visible after sources are removed.
good_pixels = image_data[(mask_data == 1) & np.isfinite(image_data) & (image_data > 0)]
if len(good_pixels) == 0:
    good_pixels = all_pixels

vmin_masked = np.percentile(good_pixels, 5)
vmax_masked = np.percentile(good_pixels, 99.5)
norm_masked = LogNorm(vmin=vmin_masked, vmax=vmax_masked)

print(f"  Raw image colour scale:    {vmin_raw:.2f} – {vmax_raw:.2f} counts (log)")
print(f"  Masked image colour scale: {vmin_masked:.2f} – {vmax_masked:.2f} counts (log)")

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 8))

# -- Panel 1: Original image (no mask) ------------------------------------
ax1 = fig.add_subplot(131, projection=wcs)
im1 = ax1.imshow(image_data, origin="lower", cmap="viridis", norm=norm_raw)
ax1.set_title("Original image (no mask)", fontsize=13, fontweight="bold")
ax1.set_xlabel("RA")
ax1.set_ylabel("Dec")
lon1 = ax1.coords[0]
lat1 = ax1.coords[1]
lon1.set_axislabel("RA", size=11)
lat1.set_axislabel("Dec", size=11)
lon1.set_major_formatter("d.dd")
lat1.set_major_formatter("d.dd")
ax1.grid(color="white", ls=":", alpha=0.4)
cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02, shrink=0.82)
cbar1.set_label("Counts", size=11)

# -- Panel 2: Cheese mask -------------------------------------------------
ax2 = fig.add_subplot(132, projection=wcs)
# Show the mask: 0 (masked, black) vs 1 (unmasked, white)
im2 = ax2.imshow(mask_data, origin="lower", cmap="gray",
                 vmin=0, vmax=1, interpolation="nearest")
ax2.set_title("Cheese mask (0 = masked source)", fontsize=13, fontweight="bold")
ax2.set_xlabel("RA")
ax2.set_ylabel("Dec")
lon2 = ax2.coords[0]
lat2 = ax2.coords[1]
lon2.set_axislabel("RA", size=11)
lat2.set_axislabel("Dec", size=11)
lon2.set_major_formatter("d.dd")
lat2.set_major_formatter("d.dd")
ax2.grid(color="gray", ls=":", alpha=0.4)
cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02, shrink=0.82,
                     ticks=[0, 1])
cbar2.set_label("Mask value", size=11)

# -- Panel 3: Masked image ------------------------------------------------
ax3 = fig.add_subplot(133, projection=wcs)
im3 = ax3.imshow(masked_image, origin="lower", cmap="viridis", norm=norm_masked)
ax3.set_title("Masked image (sources removed)", fontsize=13, fontweight="bold")
ax3.set_xlabel("RA")
ax3.set_ylabel("Dec")
lon3 = ax3.coords[0]
lat3 = ax3.coords[1]
lon3.set_axislabel("RA", size=11)
lat3.set_axislabel("Dec", size=11)
lon3.set_major_formatter("d.dd")
lat3.set_major_formatter("d.dd")
ax3.grid(color="white", ls=":", alpha=0.4)
cbar3 = plt.colorbar(im3, ax=ax3, pad=0.02, shrink=0.82)
cbar3.set_label("Counts", size=11)

# ---------------------------------------------------------------------------
# 5. Statistics summary
# ---------------------------------------------------------------------------
n_total  = image_data.size
n_masked = np.sum(mask_data == 0)
frac_masked = 100.0 * n_masked / n_total

total_counts_raw    = np.nansum(image_data)
total_counts_masked = np.nansum(masked_image)
frac_counts_removed = 100.0 * (total_counts_raw - total_counts_masked) / total_counts_raw

summary = (
    f"Pixels masked: {n_masked} / {n_total} ({frac_masked:.1f}%)\n"
    f"Counts removed: {total_counts_raw - total_counts_masked:.1f} / {total_counts_raw:.1f}"
    f" ({frac_counts_removed:.1f}%)"
)
fig.text(0.5, 0.01, summary, ha="center", fontsize=11,
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

# ---------------------------------------------------------------------------
# 6. Save and show
# ---------------------------------------------------------------------------
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_PNG}")
plt.show()
