import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
#import sys
#import matplotlib as mpl

print('#######################')
print(' Welcome')
print('#######################')

filenamefits='/home/jortecal/GitHub/eRosita/Test/events_comb_Sculptor.fits'


#filename = get_pkg_data_filename(filenamefits)
hdu=fits.open(filenamefits)[0]
print('sum',np.sum(hdu.data))
print('max',np.max(hdu.data)," min ",np.min(hdu.data))
wcs=WCS(hdu.header)

ax = plt.subplot(projection=wcs)
ax.imshow(hdu.data, vmin=0, vmax=np.max(hdu.data), origin='lower')
ax.grid(color='white', ls='solid')
#ax.colorbar()
ax.set_xlabel('Galactic Longitude')
ax.set_ylabel('Galactic Latitude')
# plt.savefig('test.pdf')   
plt.show()

#plt.subplot(projection=wcs)
#plt.imshow(hdu.data, vmin=0, vmax=np.max(hdu.data), origin='lower')
#plt.contour(hdu.data, levels=np.logspace(0,3,3),colors='white',alpha=0.5)
#plt.grid(color='white', ls='solid')
#plt.xlabel('Galactic Longitude')
#plt.ylabel('Galactic Latitude')
#plt.colorbar()
#plt.savefig('test.pdf')
