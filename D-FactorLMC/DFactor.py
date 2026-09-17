import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
import matplotlib.pyplot as plt

# === Constants ===
D_LMC = 50 * u.kpc          # Distance to LMC
r_s = 3 * u.kpc             # Scale radius
rho_0 = 1 * u.GeV / u.cm**3 # Central DM density

# === Load mask FITS ===
mask_file = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/cheesemask_LMC_Circle_masked.fits"  # Your uploaded file
hdul = fits.open(mask_file)
mask_data = hdul[0].data
header = hdul[0].header
wcs = WCS(header)
hdul.close()

# === Pixel grid ===
ny, nx = mask_data.shape
y, x = np.mgrid[:ny, :nx]
ra, dec = wcs.wcs_pix2world(x, y, 0)
sky_coords = SkyCoord(ra*u.deg, dec*u.deg)

# === LMC center ===
lmc_center = SkyCoord("05h23m34.6s", "-69d45m22s", frame='icrs')

# === Angular separation for all pixels ===
sep = sky_coords.separation(lmc_center)  # In deg

# === Projected radial distance in kpc ===
R = (sep.to(u.rad).value * D_LMC).to(u.kpc)  # R = D × θ

# === DM column density profile: integrate exp(-r/rs) along LOS ===
# Approximate result of LOS integral: ρ0 × rs × 2
# So column_density ≈ 2 × ρ0 × r_s × exp(-R/r_s)
column_density = (2 * rho_0 * r_s * np.exp(-R / r_s)).to(u.GeV / (u.cm**2))

# === Solid angle per pixel ===
pix_area_sr = (np.abs(header['CDELT1']) * u.deg).to(u.rad).value ** 2

# === Apply mask ===
masked_col_density = np.where(mask_data == 1, column_density.value, 0.0)

# === Compute total D-factor ===
D_factor = np.sum(masked_col_density) * pix_area_sr  # Units: GeV/cm² × sr

print(f"\n D-factor within masked region:")
print(f"   D = {D_factor:.3e} GeV/cm²·sr\n")

# === Plot: D-factor map (masked column density) ===
fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(projection=wcs)
im = ax.imshow(masked_col_density, origin="lower", cmap="magma", vmin=0)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("DM Column Density [GeV/cm²]")
ax.set_title("D-factor Map (Masked Column Density)")
ax.set_xlabel("RA")
ax.set_ylabel("Dec")
plt.tight_layout()
plt.show()