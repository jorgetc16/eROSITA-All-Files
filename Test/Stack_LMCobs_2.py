import os
import glob
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import matplotlib as mpl

IMG_FILE_PATH = '/Users/mnegro/MyDocuments/XRAY_PROBE/eROSITA/'

OBS1_LIST = glob.glob(IMG_FILE_PATH+'LMC_N132D/*.fits')
OBS2_LIST = glob.glob(IMG_FILE_PATH+'LMC_SN1987A/*_c001.fits')
OBS3_LIST = glob.glob(IMG_FILE_PATH+'PSR_J0537m6910/*_c001.fits')
OBS4_LIST = glob.glob(IMG_FILE_PATH+'PSR_J0540m6919/*_c001.fits')
OBS5_LIST = glob.glob(IMG_FILE_PATH+'PSR_J0540_PSR_J0537/*_c001.fits')

TOT_LIST = OBS1_LIST + OBS2_LIST + OBS3_LIST + OBS4_LIST + OBS5_LIST

RA = []
DEC = []
PI = []
EXPOSURE = []
DEADC = 0.99

for i, f in enumerate(TOT_LIST):
    print('Considering file %i'%(i+1))
    hdu = fits.open(f)['EVENTS']
    data = hdu.data
    header = hdu.header
    tstart, tstop = header['TSTART'], header['TSTOP']
    livetime = float(tstop) - float(tstart)
    RA = RA + list(data.field('RA'))
    DEC = DEC + list(data.field('DEC'))
    PI = PI + list(data.field('PI'))
    EXPOSURE = EXPOSURE + list(np.full(len(data.field('RA')), livetime))

    
RA = np.array(RA)
DEC = np.array(DEC)
PI = np.array(PI)
EXPOSURE = np.array(EXPOSURE) * DEADC


r_mask = (PI > 200) & (PI<900)
g_mask = (PI > 900) & (PI<5000)
b_mask = (PI > 5000) & (PI<10000)

print(np.sum(r_mask),np.sum(g_mask),np.sum(b_mask) )

print(RA.shape, DEC.shape, EXPOSURE.shape)

norm1 = mpl.colors.LogNorm()
norm2 = mpl.colors.LogNorm()

fig, ax = plt.subplots(figsize=(8,7))
ax.set_facecolor('black')
h = ax.hist2d(RA, DEC, bins=800, cmap='bone', weights=1/EXPOSURE, range=[[77, 87],[-71.2, -68.4]], norm=norm1)
plt.xlabel('RA [deg]', size=20)
plt.ylabel('DEC [deg]', size=20)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8,7))
ax.set_facecolor('black')
h1 = ax.hist2d(RA, DEC, bins=800, cmap='bone', range=[[77, 87],[-71.2, -68.4]], norm=norm2)
plt.xlabel('RA [deg]', size=20)
plt.ylabel('DEC [deg]', size=20)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()


fig, ax = plt.subplots(figsize=(8,7))
ax.set_facecolor('black')

h1 = ax.hist2d(RA[r_mask], DEC[r_mask], bins=800, cmap='Reds_r', weights=1/EXPOSURE[r_mask], range=[[77, 87],[-71.2, -68.4]], norm=norm1, alpha=0.5)
plt.xlabel('RA [deg]', size=20)
plt.ylabel('DEC [deg]', size=20)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8,7))
ax.set_facecolor('black')
h2 = ax.hist2d(RA[g_mask], DEC[g_mask], bins=800, cmap='Greens_r', weights=1/EXPOSURE[g_mask], range=[[77, 87],[-71.2, -68.4]], norm=norm1, alpha=0.5)
plt.xlabel('RA [deg]', size=20)
plt.ylabel('DEC [deg]', size=20)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8,7))
ax.set_facecolor('black')
h3 = ax.hist2d(RA[b_mask], DEC[b_mask], bins=800, cmap='Blues_r', weights=1/EXPOSURE[b_mask], range=[[77, 87],[-71.2, -68.4]], norm=norm1, alpha=0.5)
plt.xlabel('RA [deg]', size=20)
plt.ylabel('DEC [deg]', size=20)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()

plt.xlabel('RA [deg]', size=20)
plt.ylabel('DEC [deg]', size=20)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()


plt.show()
