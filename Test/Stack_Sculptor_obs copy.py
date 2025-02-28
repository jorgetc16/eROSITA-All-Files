import os
import glob
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import RectBivariateSpline, interp1d

N = 500
SRC_MASK_RAD = 0.03 #degrees

SELREG_IMAGE_FILE = "/home/jortecal/GitHub/eRosita/Test/events_comb_Sculptor.fits"
# SELREG_IMAGE_FILE = "image_comb_Sculptor_rebin40_1-10keV.fits"

# AEFF_F = 'eROSITA_aeff.csv'
# aeff = np.loadtxt(AEFF_F).T
# aeff_en = aeff[0]
# aeff_cm2 = aeff[1]
# aeff_spline = interp1d(aeff_en, aeff_cm2)

# EXPOSURE_FILE = "/home/jortecal/GitHub/eRosita/Test/expmap_comb_Sculptor.fits"
EXPOSURE_FILE = "/home/jortecal/GitHub/eRosita/Test/image_comb_Sculptor.fits"
expmap_f = fits.open(EXPOSURE_FILE)[0]
expmap = fits.open(EXPOSURE_FILE)[0].data
wcs = WCS(expmap_f.header)
expmap_shape = expmap.shape
print('exp_shape',expmap_shape)

# Generate arrays of pixel positions
x_coords, y_coords = np.meshgrid(np.arange(expmap_shape[1]), np.arange(expmap_shape[0]))

# Convert pixel coordinates to sky coordinates (RA, DEC)
exp_ra, exp_dec = wcs.all_pix2world(x_coords, y_coords, 0)
# print(exp_ra[0], exp_dec.T[0])
expmap_spline2d = RectBivariateSpline(exp_ra[0][::-1], exp_dec.T[0], expmap[::-1])

# import sys
# sys.exit()
# print(expmap)
# plt.imshow(expmap)
# plt.show()

ARF_FILES = ["/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm1/bcf/tm1_arf_filter_000101v02.fits",
             "/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm2/bcf/tm2_arf_filter_000101v02.fits",
             "/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm3/bcf/tm3_arf_filter_000101v02.fits",
             "/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm4/bcf/tm4_arf_filter_000101v02.fits",
             "/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm5/bcf/tm5_arf_filter_000101v02.fits",
             "/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm6/bcf/tm6_arf_filter_000101v02.fits",
             "/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm7/bcf/tm7_arf_filter_000101v02.fits",
            ]


# import sys
# sys.exit()

RMF_FILES = ["/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm1/bcf/tm1_rmf_141103v02.fits",
             "/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm2/bcf/tm2_rmf_141103v02.fits",
             "/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm3/bcf/tm3_rmf_141103v02.fits",
             "/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm4/bcf/tm4_rmf_141103v02.fits",
             "/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm5/bcf/tm5_rmf_141103v02.fits",
             "/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm6/bcf/tm6_rmf_141103v02.fits",
             "/home/jortecal/GitHub/eRosita/Test/RMF_ARF/tm7/bcf/tm7_rmf_141103v02.fits",
            ]
             

CAT_FILE = "eRASS1_Main.v1.1.fits"
src_list = fits.open(CAT_FILE)['CATALOG'].data
src_RA = src_list.field('RA')
src_DEC = src_list.field('DEC')
src_ext = src_list.field('EXT')/60/60 #arcsec/60/60=deg

Sculptor_RA, Sculptor_DEC = 15.0118799, -33.61837634

def pi2keV(pi, pimin=200, pimax=10000, emin=0.5, emax=10.0):
    m = (emin - emax) / (pimin - pimax)
    q = emin - m*pimin
    en = m*pi + q
    return en

def mask_src_wrong(evt_ras, evt_decs, src_ras, src_decs, radius=0.01):
    to_be_masked_events = np.array([1957685])
    for i, (r,d) in enumerate(zip(src_ras, src_decs)):
        print('%i ...'%i, r,d)
        # iii_ = np.where(((evt_ras > r-radius) & (evt_ras < r+radius) & (evt_decs > d-radius) & (evt_decs < d+radius)))[0]
        # Calculate distance between events and source
        distances = np.sqrt((evt_ras - r)**2 + (evt_decs - d)**2)
        # Find indices within circular region
        iii_ = np.where(distances < radius)[0]
        to_be_masked_events = np.concatenate((to_be_masked_events, iii_))
    return np.unique(to_be_masked_events)

def mask_src(evt_ras, evt_decs, src_ras, src_decs, radius=0.01):
    
    to_be_masked_events = np.array([1957685])
    c1 = SkyCoord(ra=evt_ras, dec=evt_decs,  frame="icrs", unit="deg")

    for i, (r,d,e) in enumerate(zip(src_ras, src_decs, src_ext)):
        print('%i ...'%i, r,d,e)
        # c2 = SkyCoord(ra=r, dec=d,  frame="icrs", unit="deg")
        # Calculate distance between events and source
        distances = np.arccos(np.sin(evt_decs*np.pi/180) * np.sin(d*np.pi/180) + 
                 np.cos(evt_decs*np.pi/180) * np.cos(d*np.pi/180) * np.cos((evt_ras-r)*np.pi/180)) * 180/np.pi
        
        #slower
        # distances = (c1.separation(c2)).deg
        if e > 0.0:
            print('Mask radius = %.3f'%(e*3))
            iii_ = np.where(distances < e*3)[0]
        else:
            iii_ = np.where(distances < radius)[0]
        to_be_masked_events = np.concatenate((to_be_masked_events, iii_))
    return np.unique(to_be_masked_events)

def remove_elements_by_indices(arr, indices_to_remove):
    print('Number of elements to remove:', len(indices_to_remove))
    return arr[~np.isin(np.arange(len(arr)), indices_to_remove)]

RA = []
DEC = []
PI = []

# DEADC = 0.99
hdu = fits.open(SELREG_IMAGE_FILE)['EVENTS']
data = hdu.data
header = hdu.header
tstart, tstop = header['TSTART'], header['TSTOP']
livetime = float(tstop) - float(tstart)
RA = data.field('RA')
iii_ra = np.argsort(RA)
RA = RA[iii_ra]
DEC = data.field('DEC')[iii_ra]
PI = data.field('PI')[iii_ra]
TM = data.field('TM_NR')[iii_ra]
# EXPOSURE = EXPOSURE + list(np.full(len(data.field('RA')), livetime))
# EXPOSURE = np.array(EXPOSURE) * DEADC
EXPOSURE = expmap_spline2d(RA, DEC, grid=False)


mask = (RA != 0.0) & (DEC != 0.0)
RA = RA[mask]
DEC = DEC[mask]
PI = PI[mask]
ENERGY = pi2keV(PI)
# EXPOSURE = EXPOSURE[mask]

# selecting sources in the ROI form the eRASS1 catalog:
sel_src = (src_RA > 10) & (src_RA < 20) & (src_DEC >-36 ) & (src_DEC < -30)
print('Selecting only %i sources in the Sculptor region'%(sum(sel_src)))

# remove events near sources
if os.path.exists('mask_pointsrc_events_indeces.npy'):
    mask_pointsrc = np.load('mask_pointsrc_events_indeces.npy')
else:
    mask_pointsrc = mask_src(RA, DEC, src_RA[sel_src], src_DEC[sel_src], radius=SRC_MASK_RAD)
    np.save('mask_pointsrc_events_indeces.npy', mask_pointsrc)

RA_new = remove_elements_by_indices(RA, mask_pointsrc)
DEC_new = remove_elements_by_indices(DEC, mask_pointsrc)
PI_new = remove_elements_by_indices(PI, mask_pointsrc)
TM_new = remove_elements_by_indices(TM, mask_pointsrc)

ENERGY_new = pi2keV(PI_new)
EXPOSURE_new = remove_elements_by_indices(EXPOSURE, mask_pointsrc)


# fig = plt.figure(figsize=(8,7))
# norm1 = mpl.colors.LogNorm()
# plt.title('Counts map (unmasked)')
# h2 = plt.hist2d(RA, DEC, bins=N, cmap='viridis', norm=norm1) #, weights=1/EXPOSURE_new
# plt.xlabel('RA [deg]', size=20)
# plt.ylabel('DEC [deg]', size=20)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.gca().invert_xaxis()
# plt.colorbar()
# plt.tight_layout()

# fig = plt.figure(figsize=(8,7))
# norm1 = mpl.colors.LogNorm()
# plt.title('Counts map')
# h2 = plt.hist2d(RA_new, DEC_new, bins=N, cmap='viridis', norm=norm1) #, weights=1/EXPOSURE_new
# plt.xlabel('RA [deg]', size=20)
# plt.ylabel('DEC [deg]', size=20)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.colorbar()
# plt.tight_layout()

# fig = plt.figure(figsize=(8,7))
# norm1 = mpl.colors.LogNorm()
# plt.title('Exposure map')
# h2 = plt.imshow(expmap, aspect="auto", origin='lower', extent=(exp_ra[0][0], exp_ra[0][-1], exp_dec.T[0][0], exp_dec.T[0][-1]))
# plt.xlabel('RA [deg]', size=20)
# plt.ylabel('DEC [deg]', size=20)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.colorbar()
# plt.tight_layout()

# fig = plt.figure(figsize=(8,7))
# norm1 = mpl.colors.LogNorm()
# plt.title('Rate map')
# h2 = plt.hist2d(RA_new, DEC_new, bins=N, weights=1/EXPOSURE_new, cmap='viridis', norm=norm1) #
# plt.xlabel('RA [deg]', size=20)
# plt.ylabel('DEC [deg]', size=20)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.colorbar()
# plt.tight_layout()


mask_tm1 = (TM_new == 1)
mask_tm2 = (TM_new == 2)
mask_tm3 = (TM_new == 3)
mask_tm4 = (TM_new == 4)
mask_tm5 = (TM_new == 5)
mask_tm6 = (TM_new == 6)
mask_tm7 = (TM_new == 7)

# cnt1, cnt1_xedges = np.histogram(ENERGY, bins=245) #980 (9800 channels)
# cnt2, cnt2_xedges = np.histogram(ENERGY_new, bins=245) #980 (9800 channels)
# h1, h1_xedges = np.histogram(ENERGY, weights=1/EXPOSURE, bins=245) #980 (9800 channels)
# h2, h2_xedges = np.histogram(ENERGY_new, weights=1/EXPOSURE_new, bins=245) #980 (9800 channels)
# h1_x = (h1_xedges[1:]+h1_xedges[:-1])/2
# h2_x = (h2_xedges[1:]+h2_xedges[:-1])/2
# plt.figure()
# plt.errorbar(h1_x, h1, yerr = np.sqrt(cnt1)/np.mean(EXPOSURE), label='w/ sources')
# plt.errorbar(h2_x, h2, yerr = np.sqrt(cnt2)/np.mean(EXPOSURE_new), label='w/o sources')
# plt.xlabel('Energy (keV)', size=18)
# plt.ylabel('cnt/s', size=18)
# plt.yscale('log')
# plt.legend(fontsize=15)
# plt.tight_layout()

# cnt1, cnt1_xedges = np.histogram(ENERGY_new, bins=490) #980 (9800 channels)
# h1, h1_xedges = np.histogram(ENERGY_new, weights=1/EXPOSURE_new, bins=490) #980 (9800 channels)
# h1_x = (h1_xedges[1:]+h1_xedges[:-1])/2

# plt.figure()
# plt.errorbar(h1_x, h1/aeff_spline(h1_x), yerr = np.sqrt(cnt1)/np.mean(EXPOSURE_new)/aeff_spline(h1_x))
# plt.errorbar(h1_x, h1, yerr = np.sqrt(h1), fmt='.', label='ph/s', color='0.3')
# plt.xlabel('Energy (keV)', size=20)
# plt.ylabel(r'ph cm$^{-2}$ s$^{-1}$', size=20)
# plt.yscale('log')
# plt.legend(fontsize=15)
# plt.tight_layout()


cnt1, cnt1_xedges = np.histogram(ENERGY_new[mask_tm1], bins=490) #980 (9800 channels)
h1, h1_xedges = np.histogram(ENERGY_new[mask_tm1], weights=1/EXPOSURE_new[mask_tm1]/7, bins=490) #980 (9800 channels)
h1_x = (h1_xedges[1:]+h1_xedges[:-1])/2
cnt2, cnt2_xedges = np.histogram(ENERGY_new[mask_tm2], bins=490) #980 (9800 channels)
h2, h2_xedges = np.histogram(ENERGY_new[mask_tm2], weights=1/EXPOSURE_new[mask_tm2]/7, bins=490) #980 (9800 channels)
h2_x = (h2_xedges[1:]+h2_xedges[:-1])/2
cnt3, cnt3_xedges = np.histogram(ENERGY_new[mask_tm3], bins=490) #980 (9800 channels)
h3, h3_xedges = np.histogram(ENERGY_new[mask_tm3], weights=1/EXPOSURE_new[mask_tm3]/7, bins=490) #980 (9800 channels)
h3_x = (h3_xedges[1:]+h3_xedges[:-1])/2
cnt4, cnt4_xedges = np.histogram(ENERGY_new[mask_tm4], bins=490) #980 (9800 channels)
h4, h4_xedges = np.histogram(ENERGY_new[mask_tm4], weights=1/EXPOSURE_new[mask_tm4]/7, bins=490) #980 (9800 channels)
h4_x = (h4_xedges[1:]+h4_xedges[:-1])/2
cnt5, cnt5_xedges = np.histogram(ENERGY_new[mask_tm5], bins=490) #980 (9800 channels)
h5, h5_xedges = np.histogram(ENERGY_new[mask_tm5], weights=1/EXPOSURE_new[mask_tm5]/7, bins=490) #980 (9800 channels)
h5_x = (h5_xedges[1:]+h5_xedges[:-1])/2
cnt6, cnt6_xedges = np.histogram(ENERGY_new[mask_tm6], bins=490) #980 (9800 channels)
h6, h6_xedges = np.histogram(ENERGY_new[mask_tm6], weights=1/EXPOSURE_new[mask_tm6]/7, bins=490) #980 (9800 channels)
h6_x = (h6_xedges[1:]+h6_xedges[:-1])/2
cnt7, cnt7_xedges = np.histogram(ENERGY_new[mask_tm7], bins=490) #980 (9800 channels)
h7, h7_xedges = np.histogram(ENERGY_new[mask_tm7], weights=1/EXPOSURE_new[mask_tm7]/7, bins=490) #980 (9800 channels)
h7_x = (h5_xedges[1:]+h5_xedges[:-1])/2


plt.figure()
total_arf = fits.open(ARF_FILES[0])['SPECRESP'].data
arf_emean = (total_arf.field('ENERG_LO')+total_arf.field('ENERG_HI'))/2
aeffs = []
total_arf = total_arf.field('SPECRESP')
for i, a in enumerate(ARF_FILES):
    arf_table = fits.open(a)['SPECRESP'].data
    if i == 0: 
        plt.plot(arf_emean, total_arf, label='TM1')
        arf_spline = interp1d(arf_emean, total_arf)
    else:
        arf = arf_table.field('SPECRESP')
        arf_emin = arf_table.field('ENERG_LO')
        arf_emax = arf_table.field('ENERG_HI')
        arf_emean = (arf_emin+arf_emax)/2
        arf_spline = interp1d(arf_emean, arf)
        plt.plot(arf_emean, arf_spline(arf_emean), label='TM%i'%(i+1))
        total_arf = total_arf + arf
    aeffs.append(arf_spline)
plt.plot(arf_emean, total_arf,'--', color='k', label='Total')
plt.xlabel('Energy (keV)', size=20)
plt.ylabel(r'A$_{eff} [$cm$^{2}$]', size=20)
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()

plt.figure()
plt.errorbar(h1_x, h1/aeffs[0](h1_x), yerr = np.sqrt(cnt1)/np.mean(EXPOSURE_new[mask_tm1])/aeffs[0](h1_x), label='TM 1')
plt.errorbar(h2_x, h2/aeffs[1](h2_x), yerr = np.sqrt(cnt2)/np.mean(EXPOSURE_new[mask_tm2])/aeffs[1](h2_x), label='TM 2')
plt.errorbar(h3_x, h3/aeffs[2](h3_x), yerr = np.sqrt(cnt3)/np.mean(EXPOSURE_new[mask_tm3])/aeffs[2](h3_x), label='TM 3')
plt.errorbar(h4_x, h4/aeffs[3](h4_x), yerr = np.sqrt(cnt4)/np.mean(EXPOSURE_new[mask_tm4])/aeffs[3](h4_x), label='TM 4')
plt.errorbar(h5_x, h5/aeffs[4](h5_x), yerr = np.sqrt(cnt5)/np.mean(EXPOSURE_new[mask_tm5])/aeffs[4](h5_x), label='TM 5')
plt.errorbar(h6_x, h6/aeffs[5](h6_x), yerr = np.sqrt(cnt6)/np.mean(EXPOSURE_new[mask_tm6])/aeffs[5](h6_x), label='TM 6')
plt.errorbar(h7_x, h7/aeffs[6](h7_x), yerr = np.sqrt(cnt7)/np.mean(EXPOSURE_new[mask_tm7])/aeffs[6](h7_x), label='TM 7')
plt.xlabel('Energy (keV)', size=20)
plt.ylabel(r'ph cm$^{-2}$ s$^{-1}$', size=20)
plt.yscale('log')
plt.legend(fontsize=15)
plt.tight_layout()

plt.figure()
plt.errorbar(h1_x, h1, yerr = np.sqrt(cnt1)/np.mean(EXPOSURE_new[mask_tm1]), label='TM 1')
plt.errorbar(h2_x, h2, yerr = np.sqrt(cnt2)/np.mean(EXPOSURE_new[mask_tm2]), label='TM 2')
plt.errorbar(h3_x, h3, yerr = np.sqrt(cnt3)/np.mean(EXPOSURE_new[mask_tm3]), label='TM 3')
plt.errorbar(h4_x, h4, yerr = np.sqrt(cnt4)/np.mean(EXPOSURE_new[mask_tm4]), label='TM 4')
plt.errorbar(h5_x, h5, yerr = np.sqrt(cnt5)/np.mean(EXPOSURE_new[mask_tm5]), label='TM 5')
plt.errorbar(h6_x, h6, yerr = np.sqrt(cnt6)/np.mean(EXPOSURE_new[mask_tm6]), label='TM 6')
plt.errorbar(h7_x, h7, yerr = np.sqrt(cnt7)/np.mean(EXPOSURE_new[mask_tm7]), label='TM 7')
plt.xlabel('Energy (keV)', size=20)
plt.ylabel(r'ph s$^{-1}$', size=20)
plt.yscale('log')
plt.legend(fontsize=15)
plt.tight_layout()

plt.show()
