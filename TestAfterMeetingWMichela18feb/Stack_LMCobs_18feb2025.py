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
import shutil
import sys

N = 200
SRC_MASK_RAD = 0.03 #degrees

SELREG_IMAGE_FILE = "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMCFiles/selreg_image_comb_LMC_rad3deg.fits"
map = fits.open(SELREG_IMAGE_FILE)[1].data
# plt.imshow(map)
# plt.colorbar()
# plt.show()

SELREG_IMAGE_FILE = "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMCFiles/image_fromselreg_comb_LMC_rad3deg_rebin40.fits"
map = fits.open(SELREG_IMAGE_FILE)[1].data
# plt.imshow(map)
# plt.colorbar()
# plt.show()

ARF_FILES = ["/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_120_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_220_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_320_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_420_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_520_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_620_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_720_ARF_00001.fits",
            ]

# plt.figure()
# total_arf = fits.open(ARF_FILES[0])['SPECRESP'].data
# arf_emean = (total_arf.field('ENERG_LO')+total_arf.field('ENERG_HI'))/2
# aeffs = []
# total_arf = total_arf.field('SPECRESP')
# for i, a in enumerate(ARF_FILES):
#     arf_table = fits.open(a)['SPECRESP'].data
#     if i == 0: 
#         plt.plot(arf_emean, total_arf, label='TM1')
#         arf_spline = interp1d(arf_emean, total_arf)
#     else:
#         arf = arf_table.field('SPECRESP')
#         arf_emin = arf_table.field('ENERG_LO')
#         arf_emax = arf_table.field('ENERG_HI')
#         arf_emean = (arf_emin+arf_emax)/2
#         arf_spline = interp1d(arf_emean, arf)
#         plt.plot(arf_emean, arf_spline(arf_emean), label='TM%i'%(i+1))
#         total_arf = total_arf + arf
#     aeffs.append(arf_spline)
# plt.plot(arf_emean, total_arf,'--', color='k', label='Total')
# plt.xlabel('Energy (keV)', size=20)
# plt.ylabel(r'A$_{eff} [$cm$^{2}$]', size=20)
# plt.yscale('log')
# plt.xscale('log')
# plt.legend()
# plt.show()

RMF_FILES = ["/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_120_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_220_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_320_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_420_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_520_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_620_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_720_RMF_00001.fits",
            ]

rmf_ebounds = fits.open(RMF_FILES[0])['EBOUNDS'].data 
rmf_channel = rmf_ebounds.field('CHANNEL')  
rmf_emin = rmf_ebounds.field('E_MIN')  
rmf_emax = rmf_ebounds.field('E_MAX') 
rmf_ebinedges = np.append(rmf_emin, rmf_emax[-1])


# plt.figure()
# plt.fill_between(rmf_channel, rmf_emin, rmf_emax)
# plt.show()

# import sys
# sys.exit()


###### No idea what's this for, we do not have the aeff.csv file ###########
# AEFF_F = 'eROSITA_aeff.csv'
# aeff = np.loadtxt(AEFF_F).T
# aeff_en = aeff[0]
# aeff_cm2 = aeff[1]
# aeff_spline = interp1d(aeff_en, aeff_cm2)
##############################################################################

EXPOSURE_FILE = "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMCFiles/expmap_srcreg_comb_LMC_rad3deg_rebin40.fits"
# EXPOSURE_FILE = "expmap_comb_LMC_rebin40_1-10keV.fits"

expmap_f = fits.open(EXPOSURE_FILE)
elapse_TMs = []
for i in range(1,7):
    gti = expmap_f['GTI%i'%i].data
    elapse = np.sum(gti['STOP'] - gti['START'])
    elapse_TMs.append(elapse)
print('ELAPSE TIMES', elapse_TMs)

sys.exit()
expmap = fits.open(EXPOSURE_FILE)[0].data
wcs = WCS(expmap_f[0].header)
expmap_shape = expmap.shape
print('exp_shape',expmap_shape)

# Generate arrays of pixel positions
x_coords, y_coords = np.meshgrid(np.arange(expmap_shape[1]), np.arange(expmap_shape[0]))

# Convert pixel coordinates to sky coordinates (RA, DEC)
exp_ra, exp_dec = wcs.all_pix2world(x_coords, y_coords, 0)
# print(exp_ra[0], exp_dec.T[0])
expmap_spline2d = RectBivariateSpline(exp_ra[0][::-1], exp_dec.T[0], expmap[::-1])


# plt.imshow(expmap)
# plt.colorbar()
# plt.show()

 


CAT_FILE = "/home/jortecal/GitHub/eRosita/Test/eRASS1_Main.v1.1.fits"
src_list = fits.open(CAT_FILE)['CATALOG'].data
src_RA = src_list.field('RA')
src_DEC = src_list.field('DEC')
src_ext = src_list.field('EXT')/60/60 #arcsec/60/60=deg

LMC_RA, LMC_DEC = 80.89417, -69.75611

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
    # c1 = SkyCoord(ra=evt_ras, dec=evt_decs,  frame="icrs", unit="deg")

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
sel_src = (src_RA > 72) & (src_RA < 90) & (src_DEC >-73 ) & (src_DEC < -66)
print('Selecting only %i sources in the LMC region'%(sum(sel_src)))

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
MEAN_EXPOSURE_TM1 = np.mean(EXPOSURE_new[mask_tm1])/7
MEAN_EXPOSURE_TM2 = np.mean(EXPOSURE_new[mask_tm2])/7
MEAN_EXPOSURE_TM3 = np.mean(EXPOSURE_new[mask_tm3])/7
MEAN_EXPOSURE_TM4 = np.mean(EXPOSURE_new[mask_tm4])/7
MEAN_EXPOSURE_TM5 = np.mean(EXPOSURE_new[mask_tm5])/7
MEAN_EXPOSURE_TM6 = np.mean(EXPOSURE_new[mask_tm6])/7
MEAN_EXPOSURE_TM7 = np.mean(EXPOSURE_new[mask_tm7])/7
print('MEAN EXPOSURE', MEAN_EXPOSURE_TM1)

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



# BINS = int(np.round(np.max(PI_new) - np.min(PI_new), decimals=0))
print('NUM OF BINS:', len(rmf_ebinedges), rmf_channel)
cnt1, cnt1_xedges = np.histogram(ENERGY_new[mask_tm1], bins=rmf_ebinedges) #980 (9800 channels)
h1, h1_xedges = np.histogram(ENERGY_new[mask_tm1], weights=1/EXPOSURE_new[mask_tm1]/7, bins=rmf_ebinedges) #980 (9800 channels)
h1_x = ((h1_xedges[1:]+h1_xedges[:-1])/2).astype('int')
cnt2, cnt2_xedges = np.histogram(ENERGY_new[mask_tm2], bins=rmf_ebinedges) #980 (9800 channels)
h2, h2_xedges = np.histogram(ENERGY_new[mask_tm2], weights=1/EXPOSURE_new[mask_tm2]/7, bins=rmf_ebinedges) #980 (9800 channels)
h2_x = ((h2_xedges[1:]+h2_xedges[:-1])/2).astype('int')
cnt3, cnt3_xedges = np.histogram(ENERGY_new[mask_tm3], bins=rmf_ebinedges) #980 (9800 channels)
h3, h3_xedges = np.histogram(ENERGY_new[mask_tm3], weights=1/EXPOSURE_new[mask_tm3]/7, bins=rmf_ebinedges) #980 (9800 channels)
h3_x = ((h3_xedges[1:]+h3_xedges[:-1])/2).astype('int')
cnt4, cnt4_xedges = np.histogram(ENERGY_new[mask_tm4], bins=rmf_ebinedges) #980 (9800 channels)
h4, h4_xedges = np.histogram(ENERGY_new[mask_tm4], weights=1/EXPOSURE_new[mask_tm4]/7, bins=rmf_ebinedges) #980 (9800 channels)
h4_x = ((h4_xedges[1:]+h4_xedges[:-1])/2).astype('int')
cnt5, cnt5_xedges = np.histogram(ENERGY_new[mask_tm5], bins=rmf_ebinedges) #980 (9800 channels)
h5, h5_xedges = np.histogram(ENERGY_new[mask_tm5], weights=1/EXPOSURE_new[mask_tm5]/7, bins=rmf_ebinedges) #980 (9800 channels)
h5_x = ((h5_xedges[1:]+h5_xedges[:-1])/2).astype('int')
cnt6, cnt6_xedges = np.histogram(ENERGY_new[mask_tm6], bins=rmf_ebinedges) #980 (9800 channels)
h6, h6_xedges = np.histogram(ENERGY_new[mask_tm6], weights=1/EXPOSURE_new[mask_tm6]/7, bins=rmf_ebinedges) #980 (9800 channels)
h6_x = ((h6_xedges[1:]+h6_xedges[:-1])/2).astype('int')
cnt7, cnt7_xedges = np.histogram(ENERGY_new[mask_tm7], bins=rmf_ebinedges) #980 (9800 channels)
h7, h7_xedges = np.histogram(ENERGY_new[mask_tm7], weights=1/EXPOSURE_new[mask_tm7]/7, bins=rmf_ebinedges) #980 (9800 channels)
h7_x = ((h5_xedges[1:]+h5_xedges[:-1])/2).astype('int')


channel_TMs = [np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025)]

rate_TMs    = [cnt1/MEAN_EXPOSURE_TM1,
               cnt2/MEAN_EXPOSURE_TM2,
               cnt3/MEAN_EXPOSURE_TM3,
               cnt4/MEAN_EXPOSURE_TM4,
               cnt5/MEAN_EXPOSURE_TM5,
               cnt6/MEAN_EXPOSURE_TM6,
               cnt7/MEAN_EXPOSURE_TM7]

kk = 1
error_TMs   = [np.sqrt(cnt1)/MEAN_EXPOSURE_TM1*kk,
               np.sqrt(cnt2)/MEAN_EXPOSURE_TM2*kk,
               np.sqrt(cnt3)/MEAN_EXPOSURE_TM3*kk,
               np.sqrt(cnt4)/MEAN_EXPOSURE_TM4*kk,
               np.sqrt(cnt5)/MEAN_EXPOSURE_TM5*kk,
               np.sqrt(cnt6)/MEAN_EXPOSURE_TM6*kk,
               np.sqrt(cnt7)/MEAN_EXPOSURE_TM7*kk,
               ]
               

# plt.figure()
# total_arf = fits.open(ARF_FILES[0])['SPECRESP'].data
# arf_emean = (total_arf.field('ENERG_LO')+total_arf.field('ENERG_HI'))/2
# aeffs = []
# total_arf = total_arf.field('SPECRESP')
# for i, a in enumerate(ARF_FILES):
#     arf_table = fits.open(a)['SPECRESP'].data
#     if i == 0: 
#         plt.plot(arf_emean, total_arf, label='TM1')
#         arf_spline = interp1d(arf_emean, total_arf)
#     else:
#         arf = arf_table.field('SPECRESP')
#         arf_emin = arf_table.field('ENERG_LO')
#         arf_emax = arf_table.field('ENERG_HI')
#         arf_emean = (arf_emin+arf_emax)/2
#         arf_spline = interp1d(arf_emean, arf)
#         plt.plot(arf_emean, arf_spline(arf_emean), label='TM%i'%(i+1))
#         total_arf = total_arf + arf
#     aeffs.append(arf_spline)
# plt.plot(arf_emean, total_arf,'--', color='k', label='Total')
# plt.xlabel('Energy (keV)', size=20)
# plt.ylabel(r'A$_{eff} [$cm$^{2}$]', size=20)
# plt.yscale('log')
# plt.xscale('log')
# plt.legend()
# plt.show()

# plt.figure()
# plt.errorbar(h1_x, h1/aeffs[0](h1_x), yerr = np.sqrt(cnt1)/np.mean(EXPOSURE_new[mask_tm1])/aeffs[0](h1_x), label='TM 1')
# plt.errorbar(h2_x, h2/aeffs[1](h2_x), yerr = np.sqrt(cnt2)/np.mean(EXPOSURE_new[mask_tm2])/aeffs[1](h2_x), label='TM 2')
# plt.errorbar(h3_x, h3/aeffs[2](h3_x), yerr = np.sqrt(cnt3)/np.mean(EXPOSURE_new[mask_tm3])/aeffs[2](h3_x), label='TM 3')
# plt.errorbar(h4_x, h4/aeffs[3](h4_x), yerr = np.sqrt(cnt4)/np.mean(EXPOSURE_new[mask_tm4])/aeffs[3](h4_x), label='TM 4')
# plt.errorbar(h5_x, h5/aeffs[4](h5_x), yerr = np.sqrt(cnt5)/np.mean(EXPOSURE_new[mask_tm5])/aeffs[4](h5_x), label='TM 5')
# plt.errorbar(h6_x, h6/aeffs[5](h6_x), yerr = np.sqrt(cnt6)/np.mean(EXPOSURE_new[mask_tm6])/aeffs[5](h6_x), label='TM 6')
# plt.errorbar(h7_x, h7/aeffs[6](h7_x), yerr = np.sqrt(cnt7)/np.mean(EXPOSURE_new[mask_tm7])/aeffs[6](h7_x), label='TM 7')
# plt.xlabel('Energy (keV)', size=20)
# plt.ylabel(r'ph cm$^{-2}$ s$^{-1}$', size=20)
# plt.yscale('log')
# plt.legend(fontsize=15)
# plt.tight_layout()



plt.figure()
plt.errorbar(channel_TMs[0], rate_TMs[0], fmt='.', yerr = error_TMs[0], label='TM 1')
plt.errorbar(channel_TMs[1], rate_TMs[1], fmt='.', yerr = error_TMs[1], label='TM 2')
plt.errorbar(channel_TMs[2], rate_TMs[2], fmt='.', yerr = error_TMs[2], label='TM 3')
plt.errorbar(channel_TMs[3], rate_TMs[3], fmt='.', yerr = error_TMs[3], label='TM 4')
plt.errorbar(channel_TMs[4], rate_TMs[4], fmt='.', yerr = error_TMs[4], label='TM 5')
plt.errorbar(channel_TMs[5], rate_TMs[5], fmt='.', yerr = error_TMs[5], label='TM 6')
plt.errorbar(channel_TMs[6], rate_TMs[6], fmt='.', yerr = error_TMs[6], label='TM 7')
# plt.xlabel('Energy  (keV)', size=20)
plt.xlabel('Channel (PI)', size=20)
plt.ylabel(r'ph s$^{-1}$', size=20)
plt.yscale('log')
plt.legend(fontsize=15)
plt.tight_layout()


fits_file = 'original_pha/ixpe02004701_det1_evt2_v02_src_pha1.fits'

for i in range(0,7):
    new_fits_file = '/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMCFiles/TMSeparatedFiles/LMC_TM%i_pha.fits'%(i+1)
    print(i, new_fits_file)

    with fits.open(fits_file) as hdul:
        # Access the relevant HDU (assuming it's the first HDU; change if needed)
        hdu = hdul[1]

        num_rows = len(hdu.data)
        new_data = np.zeros(1024, dtype=hdu.data.dtype)
        new_ch = channel_TMs[i]
        new_rt = rate_TMs[i]
        new_re = error_TMs[i]
        #  Update the new rows with the provided data
        new_data['CHANNEL'] = channel_TMs[i]
        new_data['RATE'] = rate_TMs[i]
        new_data['STAT_ERR'] = error_TMs[i]
        
        # Replace the HDU data with the new array
        hdu.data = new_data
        
        # Save the changes to a new file
        hdul.writeto(new_fits_file, overwrite=True)

    print(f"Updated columns and saved new file as {new_fits_file} successfully.")

    fits.setval(new_fits_file, 'DETCHANS', value=1024, ext=1)
    fits.setval(new_fits_file, 'TLMAX1', value=1024, ext=1)
    fits.setval(new_fits_file, 'TLMIN1', value=1, ext=1)
    fits.setval(new_fits_file, 'TELESCOP', value='eROSITA', ext=1)
    fits.setval(new_fits_file, 'INSTRUME', value='CCD', ext=1)
    fits.setval(new_fits_file, 'DETNAM', value='TM%i'%(i+1), ext=1)
    fits.setval(new_fits_file, 'EXPOSURE', value=np.mean(EXPOSURE_new[mask_tm7])/7, ext=1)
    fits.setval(new_fits_file, 'ONTIME', value=np.mean(EXPOSURE_new[mask_tm7])/7, ext=1)
    fits.setval(new_fits_file, 'DEADC', value=1.1, ext=1)
    fits.setval(new_fits_file, 'TSTART', value=6.294329108981885E8, ext=1)
    fits.setval(new_fits_file, 'TSTOP', value=6.451919387333338E8, ext=1)
    fits.setval(new_fits_file, 'MJDREF', value=51543.875, ext=1)
    fits.setval(new_fits_file, 'TELAPSE', value=elapse_TMs[i], ext=1)
    fits.setval(new_fits_file, 'DEADAPP', value=False, ext=1)


    plt.show()