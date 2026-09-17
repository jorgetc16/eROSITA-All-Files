import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.patches as patches
from matplotlib.path import Path
from astroML.plotting import plot_tissot_ellipse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

# ----------------------------------------------------------------------
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)

# ------------------------------------------------------------
# Load the FITS file
fits_file = '/home/jortecal/GitHub/eRosita/SKYMAPS_052022_MPE.fits'
hdul = fits.open(fits_file)

print("FITS file structure:")
hdul.info()

# The actual data is in extension 1 (SMAPS table)
smaps = hdul[1].data
print(f"\n✓ Found binary table 'SMAPS' with {len(smaps)} rows and {len(smaps.columns)} columns")

# Use Equatorial coordinates (matches the starmap image)
coord_system = 'Equatorial (J2000)'

print(f"\n{'='*60}")
print(f"Using {coord_system} coordinates")
print(f"Fields: {len(smaps)}")
print(f"{'='*60}")

# Load background image (equirectangular projection, celestial coordinates)
background_image = plt.imread('/home/jortecal/GitHub/eRosita/starmap_2020_4k_print.jpg')

# Create figure with 2:1 aspect ratio (standard for equirectangular)
fig = plt.figure(figsize=(20, 10))

# Create axes WITHOUT projection (equirectangular is just a rectangular plot)
ax = fig.add_subplot(111)

# Display background image
# Equirectangular: RA from 0° to 360° (x-axis), Dec from -90° to 90° (y-axis)
ax.imshow(background_image, extent=[0, 360, -90, 90], aspect='auto', origin='upper')

# Draw sky tiles as rectangles
print("\nDrawing sky tiles...")
tiles_drawn = 0
tiles_skipped = 0

for i, row in enumerate(smaps):
    # Get RA/DEC boundaries directly
    ra_min = row['RA_MIN']
    ra_max = row['RA_MAX']
    dec_min = row['DE_MIN']
    dec_max = row['DE_MAX']
    
    # Skip invalid data
    if not (np.isfinite(ra_min) and np.isfinite(ra_max) and 
            np.isfinite(dec_min) and np.isfinite(dec_max)):
        tiles_skipped += 1
        continue
    
    # Don't skip polar regions anymore - just handle RA wrapping
    
    # Handle RA wrapping at 0°/360° boundary
    if ra_max - ra_min > 180:
        # Tile crosses the boundary - draw in two parts
        # Part 1: from ra_min to 360°
        rect1 = patches.Rectangle(
            (ra_min, dec_min), 360 - ra_min, dec_max - dec_min,
            linewidth=0.3, edgecolor='cyan', facecolor='none',
            alpha=0.6
        )
        ax.add_patch(rect1)
        
        # Part 2: from 0° to ra_max
        rect2 = patches.Rectangle(
            (0, dec_min), ra_max, dec_max - dec_min,
            linewidth=0.3, edgecolor='cyan', facecolor='none',
            alpha=0.6
        )
        ax.add_patch(rect2)
        tiles_drawn += 1
    else:
        # Normal tile (doesn't cross boundary)
        width = ra_max - ra_min
        height = dec_max - dec_min
        
        rect = patches.Rectangle(
            (ra_min, dec_min), width, height,
            linewidth=0.3, edgecolor='cyan', facecolor='none',
            alpha=0.6
        )
        ax.add_patch(rect)
        tiles_drawn += 1

print(f"✓ Drew {tiles_drawn} sky tiles")
print(f"✗ Skipped {tiles_skipped} tiles (invalid data)")

# Configure axes
ax.set_xlim(0, 360)
ax.set_ylim(-90, 90)

# Set up tick marks
ax.set_xticks(np.arange(0, 361, 30))
ax.set_xticks(np.arange(0, 361, 15), minor=True)
ax.set_yticks(np.arange(-90, 91, 30))
ax.set_yticks(np.arange(-90, 91, 15), minor=True)

# Format labels
ax.set_xlabel('Right Ascension (degrees)', fontsize=16, color='white')
ax.set_ylabel('Declination (degrees)', fontsize=16, color='white')

# Style the plot
ax.tick_params(axis='both', which='major', labelsize=14, colors='white', length=6, width=1.5)
ax.tick_params(axis='both', which='minor', labelsize=10, colors='white', length=3, width=1)
ax.grid(True, which='minor', alpha=0.1, color='white', linewidth=0.3, linestyle=':')
ax.grid(True, which='major', alpha=0.3, color='white', linewidth=0.8, linestyle='-')

# Set background color
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Add title
title = f'eROSITA Sky Tiles ({coord_system})'
ax.set_title(title, fontsize=22, pad=20, color='white', weight='bold')

# Add text annotation
ax.text(0.02, 0.98, f'{tiles_drawn} sky tiles\n({tiles_skipped} skipped)', 
        transform=ax.transAxes, fontsize=12, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='cyan'),
        color='white')

# Add coordinate system label (consistent terminology)
ax.text(0.98, 0.02, coord_system, 
        transform=ax.transAxes, fontsize=11, va='bottom', ha='right',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='cyan'),
        color='white')

plt.tight_layout()
plt.savefig('/home/jortecal/GitHub/eRosita/eRosita_skymap_equatorial.png', dpi=200, bbox_inches='tight', facecolor='black')
print(f"\n✓ Saved: /home/jortecal/GitHub/eRosita/eRosita_skymap_equatorial.png")
plt.show()

hdul.close()
