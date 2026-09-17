import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# Open the FITS file
filename = '/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/cheesemask_LMC_Circle_masked_Bckg_masked.fits'
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
    # center_x = nx // 2
    # center_y = ny // 2
    # Use the coordinates of the center of the background region in galactic coordinates
   
    ra_center = 82.174
    dec_center = -69.021

    print(ra_center, dec_center)
    # Compute angular distance to the center
    coords_center = SkyCoord(ra_center*u.deg, dec_center*u.deg)
    coords = SkyCoord(ra*u.deg, dec*u.deg)
    separation = coords.separation(coords_center)

    # Mask: keep pixels outside .5 degrees from the center
    
    mask = separation > .5 * u.deg

    # Apply mask (set to NaN outside 3 degrees)
    data[~mask] = 0

    # Save the result (overwrite or create a new file)
    hdul.flush()  
