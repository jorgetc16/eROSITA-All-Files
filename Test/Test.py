from astropy.io import fits
import healpy as hp
from matplotlib import rc
import matplotlib.patheffects as path_effects

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('font', weight='bold')

import matplotlib.pyplot as plt

#  Load the FITS table
hdul = fits.open('/home/jortecal/GitHub/eRosita/Test/events_comb_Sculptor_rad2deg.fits')
print(hdul.info())

data = hdul[1].data
print(data.columns)
print(data['X'].size)

# plot a map of the image with y axis DEC in degress and x axis RA in degrees
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='mollweide')
ax.grid(True)
ax.set_xlabel('RA [deg]', fontsize=20)
ax.set_ylabel('DEC [deg]', fontsize=20)
ax.set_title('LMC', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

# plot the image
cmap = plt.get_cmap('inferno')
cmap.set_bad(color='black')
img = hp.orthview(data['X'], coord='C', cmap=cmap, min=0, max=0.1, half_sky=True, hold=True, sub=111, title='LMC',
                  cbar=False)

# add a colorbar
cb = plt.colorbar(img, orientation='horizontal')
cb.set_label('Counts/s', fontsize=20)
cb.ax.tick_params(labelsize=20)

# add a text
text = ax.text(0.5, 0.5, 'LMC', ha='center', va='center', fontsize=20, color='white')
text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
plt.show()

