from astropy.io import fits
import healpy as hp
from matplotlib import rc
import matplotlib.patheffects as path_effects

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('font', weight='bold')

import matplotlib.pyplot as plt

# Of the 4700 sky tiles, eROSITA-DE have proprietary rights on 2248 of them, and 199 sky tiles have shared rights between the German and Russian consortiums. eROSITA-DE DR1 comprises 2447 sky tiles in total, of which 199 have partial eROSITA-DE data. The FITS table here, shows the boundaries and center coordinates of these 2447 sky tiles.
# The FITS table contains the following columns:
# ColDefs(
#     name = 'SRVMAP'; format = 'J'
#     name = 'OWNER'; format = 'I'
#     name = 'RA_MIN'; format = 'D'; unit = 'deg'
#     name = 'RA_MAX'; format = 'D'; unit = 'deg'
#     name = 'DE_MIN'; format = 'D'; unit = 'deg'
#     name = 'DE_MAX'; format = 'D'; unit = 'deg'
#     name = 'RA_CEN'; format = 'D'; unit = 'deg'
#     name = 'DE_CEN'; format = 'D'; unit = 'deg'
#     name = 'ELON_CEN'; format = 'D'; unit = 'deg'
#     name = 'ELAT_CEN'; format = 'D'; unit = 'deg'
#     name = 'GLON_CEN'; format = 'D'; unit = 'deg'
#     name = 'GLAT_CEN'; format = 'D'; unit = 'deg'
#     name = 'X_MIN'; format = 'D'; unit = 'arcmin'
#     name = 'Y_MIN'; format = 'D'; unit = 'arcmin'
#     name = 'N_NBRS'; format = 'I'
#     name = 'FIELD1'; format = 'J'
#     name = 'FIELD2'; format = 'J'
#     name = 'FIELD3'; format = 'J'
#     name = 'FIELD4'; format = 'J'
#     name = 'FIELD5'; format = 'J'
#     name = 'FIELD6'; format = 'J'
#     name = 'FIELD7'; format = 'J'
#     name = 'FIELD8'; format = 'J'
#     name = 'FIELD9'; format = 'J'
# )
# # Plot the sky tiles in the eROSITA-DE DR1 catalog.
# The sky tiles are represented by the HEALPix pixels, and the boundaries of the sky tiles are shown as the red lines.

# Load the FITS table
hdul = fits.open('/home/jortecal/GitHub/eRosita/SKYMAPS_052022_MPE.fits')
data = hdul[1].data

# Plot the sky tiles and their boundaries
hp.mollview(title='eROSITA-DE DR1 Sky Tiles', coord='C', cbar=False)
for i in range(len(data)):
    ra_min = data['RA_MIN'][i]
    ra_max = data['RA_MAX'][i]
    de_min = data['DE_MIN'][i]
    de_max = data['DE_MAX'][i]
    hp.projplot([ra_min, ra_max, ra_max, ra_min, ra_min], [de_min, de_min, de_max, de_max, de_min], lonlat=True, color='dodgerblue', linewidth=0.5)
# add the grid and coordinates
hp.graticule()
# Add the location of various dwarf galaxies
dwarf_galaxies = {'Draco': (260.051, 57.915),
                  'Ursa Minor': (227.242, 67.222),
                  'Sculptor': (15.039, -33.708),
                  'Fornax': (39.997, -34.450),
                  'Carina': (100.401, -50.966),
                  'Leo I': (152.117, 12.306),
                  'Leo II': (168.083, 22.168),
                  'Sextans': (243.358, -1.569),
                  'Large Magellanic Cloud': (80.893, -69.756)}
for name, (ra, dec) in dwarf_galaxies.items():
    #add their name and a point
    hp.projtext(ra, dec, name, lonlat=True, fontsize=25, color='k')
    hp.projplot(ra, dec, '*', lonlat=True, markersize=5, color='k')


plt.show()