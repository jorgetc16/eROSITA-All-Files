import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# Open the FITS file
filename = '/home/jortecal/GitHub/eRosita/LMC5DegEv/cheesemask_comb_LMC_rad5deg_rebin80.fits'
with fits.open(filename, mode='update') as hdul:
    data = hdul[0].data
    header = hdul[0].header
    
    # Set up WCS
    w = WCS(header)
    print(w)
    # Get the image shape
    ny, nx = data.shape

    # Create a grid of pixel coordinates
    y, x = np.mgrid[:ny, :nx]

    # Convert pixel coordinates to world coordinates (RA/Dec)
    ra, dec = w.wcs_pix2world(x, y, 0)

    # Get the center of the image in pixel coordinates and convert to world coordinates
    center_x = nx // 2
    center_y = ny // 2
    ra_center, dec_center = w.wcs_pix2world(center_x, center_y, 0)

    print(ra_center, dec_center)
    # Compute angular distance to the center
    coords_center = SkyCoord(ra_center*u.deg, dec_center*u.deg)
    coords = SkyCoord(ra*u.deg, dec*u.deg)
    separation = coords.separation(coords_center)

    # Mask: keep pixels within 5 degrees
    mask = separation < 5.01 * u.deg

    # Apply mask (set to 0 outside 5 degrees)
    data[~mask] = 0

    # Save the result (overwrite or create a new file)
    hdul.flush()  
