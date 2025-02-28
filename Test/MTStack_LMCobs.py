import os
import glob
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata

import time
from datetime import timedelta

def pi2keV(pi, pimin=200, pimax=10000, emin=0.5, emax=10.0):
    m = (emin - emax) / (pimin - pimax)
    q = emin - m*pimin
    en = m*pi + q
    return en

    
def mask_src(evt_ras, evt_decs, src_ras, src_decs, src_radius, radius=0.01):
    to_be_masked_events = np.array([1957685])
    for i, (r,d,radiuscut) in enumerate(zip(src_ras, src_decs, src_radius)):
        print("%i ...",i,r, d, radiuscut)
        # Calculate distance between events and source
        distances=np.arccos(np.sin(evt_decs*np.pi/180)*np.sin(d*np.pi/180)+np.cos(evt_decs*np.pi/180)*np.cos(d*np.pi/180)*np.cos((evt_ras-r)*np.pi/180))*180/np.pi
        #c1=SkyCoord(ra=evt_ras, dec=evt_decs,  frame="icrs", unit="deg")
        #c2=SkyCoord(ra=r, dec=d,  frame="icrs", unit="deg")
        #distances=(c1.separation(c2)).deg
        # Find indices within circular region
        iii_ = np.where(distances < radiuscut)[0]
        to_be_masked_events = np.concatenate((to_be_masked_events, iii_))
    return np.unique(to_be_masked_events)

def remove_elements_by_indices(arr, indices_to_remove):
    print('indices to remove', len(indices_to_remove))
    return arr[~np.isin(np.arange(len(arr)), indices_to_remove)]


#SCulpture center
#<SkyCoord (FK5: equinox=J2000.000): (ra, dec) in deg    (15.01188228, -33.61837533)>
#Usé el galactic para descargar: l=287.5, b=-83.2


#LMC
# Events: RA.shape  (2669805,)
#Total EXPOSURE   3269211801.0374756
#Total EXPOSURE NEW   1610983065.474121


EXPOSURE_FILE = "/home/jortecal/GitHub/eRosita/Test/expmap_srcreg_comb_LMC_rad3deg_rebin40.fits"
#EXPOSURE_FILE = "/Users/marcotaoso/Documents/2024/eROSITA/testSculpt/Data/expmap_comb_Sculpt_rebin40_1-10keV.fits"

SELREG_IMAGE_FILE = "/home/jortecal/GitHub/eRosita/Test/image_fromselreg_comb_LMC_rad3deg_rebin40_1.fits"
#SELREG_IMAGE_FILE = "/Users/marcotaoso/Documents/2024/eROSITA/testSculpt/Data/image_comb_Sculpt_rebin40_1-10keV.fits"


DeltaOmega=2*np.pi*(1-np.cos(3*np.pi/180.)) #LMC
#DeltaOmega=2*np.pi*(1-np.cos(2*np.pi/180.)) #sculpt



expmap_f = fits.open(EXPOSURE_FILE)[0]
expmap = fits.open(EXPOSURE_FILE)[0].data
wcs = WCS(expmap_f.header)
expmap_shape = expmap.shape
print("Exposure shape ",expmap_shape) #Exposure shape  (16534, 16982), why???? (10902, 10902)
print("RA_CEN DEC_CEN ",expmap_f.header['RA_CEN']," ",expmap_f.header['DEC_CEN'])
print("CDELT ",wcs.wcs.cdelt[0]," ",wcs.wcs.cdelt[1])
print("CRVAL ",expmap_f.header['CRVAL1']," ",expmap_f.header['CRVAL2'])
print("CRPIX ",expmap_f.header['CRPIX1']," ",expmap_f.header['CRPIX2'])
# Generate arrays of pixel positions
x_coords, y_coords = np.meshgrid(np.arange(expmap_shape[1]), np.arange(expmap_shape[0]))
print(x_coords)
print(y_coords)
# Convert pixel coordinates to sky coordinates (RA, DEC)
exp_ra, exp_dec = wcs.all_pix2world(x_coords, y_coords, 0)
print("world coordinate ",exp_ra, " ",exp_dec)
print("*********")
print("world coordinate ",exp_ra[0], " ",exp_dec.T[0],expmap)
print("exp_ra ",exp_ra[0].shape," exp_dec",exp_dec.T[0].shape)


'''
#[::-1] revert order, print and see that in this way exp_ra[0][::-1] is sorted in increasing order
# Original Interpolation
#expmap_spline2d = RectBivariateSpline(exp_ra[0][::-1], exp_dec.T[0], expmap[::-1])
# MT I think it is not correct, the correct one is the following
# Check the first and last few RA/DEC values to determine the coordinate order
exp_ra_int=exp_ra[0]
exp_dec_int=exp_dec.T[0]
expmap_int = expmap
ra_order = 'ascending' if exp_ra_int[0] < exp_ra_int[-1] else 'descending'
dec_order = 'ascending' if exp_dec_int[0] < exp_dec_int[-1] else 'descending'
if ra_order == 'descending':
    # Flip the RA axis if it is in descending order
    expmap_int = np.flip(expmap_int, axis=1)  # Flip along the x-axis (RA axis)
    exp_ra_int = np.flip(exp_ra_int, axis=0)  # Flip the RA coordinates
    
if dec_order == 'descending':
    # Flip the DEC axis if it is in descending order
    expmap_int = np.flip(expmap_int, axis=0)  # Flip along the y-axis (DEC axis)
    exp_dec_int = np.flip(exp_dec_int, axis=0)  # Flip the DEC coordinates

print("ra_order ",ra_order," dec_order ",dec_order)
print("world coordinate ordered ",exp_ra_int, exp_dec_int, expmap_int)
print("world coordinate shapes ",exp_ra_int.shape, exp_dec_int.shape, expmap_int.T.shape)
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html
#https://stackoverflow.com/questions/35317240/confused-about-x-y-order-with-rectbivariatespline
#https://scipython.com/book/chapter-8-scipy/examples/two-dimensional-interpolation-with-scipyinterpolaterectbivariatespline/
#Careful with order of RectBivariateSpline
# To have ra as first coordinate I should transpose z
expmap_spline2d = RectBivariateSpline(exp_ra_int , exp_dec_int, expmap_int.T)
#print("exposure spline ",expmap_spline2d( expmap_f.header['DEC_CEN'], expmap_f.header['RA_CEN'], grid=False))
print("exposure spline ",expmap_spline2d(expmap_f.header['RA_CEN'], expmap_f.header['DEC_CEN'], grid=False))
#SkyCoord('5h51m13.6070s -67d07m28.025s') A point that I have taken from the LMC exposure map which value is 3927.89
#<SkyCoord (ICRS): (ra, dec) in deg    (87.80669583, -67.12445139)>
#print("exposure spline ",expmap_spline2d(87.80669583, -67.12445139, grid=False))
# SkyCoord('1h10m47.5812s -31d021m41.289s') in Sculpture map Value 224.174
#<SkyCoord (ICRS): (ra, dec) in deg    (17.698255, -31.36146917)>
print("exposure spline ",expmap_spline2d(17.698255, -31.36146917, grid=False))
#SkyCoord('0h43m40.0363s -37d30m01.177s') in Sculpture map Value 97.174
#<SkyCoord (ICRS): (ra, dec) in deg (10.91681792, -37.50032694)>
print("exposure spline ",expmap_spline2d(10.91681792, -37.50032694, grid=False))
#SkyCoord('0h43m36.3263s -31d04m55.190s') in Sculpture map Value 110.209
#<SkyCoord (ICRS): (ra, dec) in deg (10.90135958, -31.08199722)>
print("exposure spline ",expmap_spline2d(10.90135958, -31.08199722, grid=False))
#SkyCoord('1h15m04.6868s -37d12m56.474s') in Sculpture map Value 122.908
#<SkyCoord (ICRS): (ra, dec) in deg (18.76952833, -37.21568722)>
print("exposure spline ",expmap_spline2d(18.76952833, -37.21568722, grid=False))
# Use Griddata. With Griddata I reproduce the 4 tests (around corners of exposure map) + RA,DEC center
#  RectBivariateSpline gives some differences in the examples above (up to 5%), not sure if it is because of splining or a mistake.
values_test = np.array([[expmap_f.header['RA_CEN'], expmap_f.header['DEC_CEN']], [17.698255, -31.36146917], [10.91681792, -37.50032694], [10.90135958, -31.08199722], [18.76952833, -37.21568722]] )
pixel_value = griddata((exp_ra.flatten(),exp_dec.flatten()), expmap.flatten(), (values_test[:,0], values_test[:,1]), method='nearest')
print("exposure spline ",pixel_value)
'''





RA = []
DEC = []
PI = []
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
#EXPOSURE = expmap_spline2d(RA, DEC, grid=False)
EXPOSURE = griddata((exp_ra.flatten(),exp_dec.flatten()), expmap.flatten(), (RA, DEC), method='nearest')

print("RA.shape ",RA.shape)
print(EXPOSURE)
print("Total EXPOSURE ",np.sum(EXPOSURE))


mask = (RA != 0.0) & (DEC != 0.0)
RA = RA[mask]
DEC = DEC[mask]
PI = PI[mask]
ENERGY = pi2keV(PI)
RAmin = np.min(RA)
RAmax= np.max(RA)
DECmin = np.min(DEC)
DECmax= np.max(DEC)
print("Min and Max RA ",RAmin," ",RAmax)
print("Min and Max DEC ",DECmin," ",DECmax)
print("Total events ",len(RA))



#https://www.aanda.org/articles/aa/pdf/2024/02/aa47165-23.pdf
CAT_FILE = "Test/eRASS1_Main.v1.1.fits"
src_list = fits.open(CAT_FILE)['CATALOG'].data
src_RA = src_list.field('RA')
src_DEC = src_list.field('DEC')
src_EXT = src_list.field('EXT')
src_EXT_ERR = src_list.field('EXT_ERR')
print("EXT min ",np.min(src_EXT)," max ",np.max(src_EXT))
print("EXT_ERR min ",np.min(src_EXT_ERR)," max ",np.max(src_EXT_ERR))
print(len(src_RA))


# Decide radius of each source where to mask, to be improved
src_ML_FLUX_1 = src_list.field('ML_FLUX_1')
#src_ML_FLUX_P9 = src_list.field('ML_FLUX_P9')
src_DET_LIKE_0 = src_list.field('DET_LIKE_0')
print(np.min(src_DET_LIKE_0)," ",np.max(src_DET_LIKE_0))
#half-energy width‖ (HEW). This is the angular diameter of the image of a point source, which
#contains half the flux (at a given energy) focused by the telescope.
# I would say HEW=1.35*sigma while FWHM =2*Sqrt[2 Log[2.]]*sigma=2.355*sigma
#https://www.aanda.org/articles/aa/pdf/2024/02/aa47165-23.pdf
#"The measured HEWs from the source stacking method applied
#to survey data are 30.0′′ in the 0.2–2.3 keV band and 34.4′′ in
#the 2.3–5.0 keV band, very close to the pre-flights estimates
#of 28.3′′ and 36.2′′, respectively, for the shapelet representa-
#tion, and 32.0′′ and 38.0′′ for the PANTER ground-based values."
#If I take HEW=34.4 -> sigma
sigma=30./3600. #30/3600
#https://arxiv.org/pdf/1808.09225
#Approximate noise considering faintest source and 5 sigma detection
Fnoise=np.min(src_ML_FLUX_1)/5.
# Fnoise = Flux_source *exp(-r^2/2\sigma^2). Find r where this happens for each source
src_maskrad = np.sqrt(-np.log(Fnoise/src_ML_FLUX_1)*2*sigma**2)
# Not sure the above is correct!!!
#src_maskrad = np.sqrt(-np.log(Fnoise/src_ML_FLUX_1)*2*sigma**2 -2*sigma**2*np.log(2*np.pi*sigma**2) )


print(src_maskrad*3600)
print("Min and max radius of the mask in arcsec",np.min(src_maskrad)*3600," ",np.max(src_maskrad)*3600)




# selecting sources in the ROI form the eRASS1 catalog:
#sel_src = (src_RA > 72) & (src_RA < 90) & (src_DEC >-73 ) & (src_DEC < -66)
sel_src = (src_RA > RAmin) & (src_RA < RAmax) & (src_DEC > DECmin ) & (src_DEC < DECmax)
print('Selecting only %i sources in the LMC region'%(sum(sel_src)))
print("Out of n sources in catalogue ",len(src_RA))


# remove events near sources
start_time = time.monotonic()
mask_pointsrc = mask_src(RA, DEC, src_RA[sel_src], src_DEC[sel_src], src_maskrad[sel_src], radius=0.01)
end_time = time.monotonic()
print("Time for mask ",timedelta(seconds=end_time - start_time))

RA_new = remove_elements_by_indices(RA, mask_pointsrc)
DEC_new = remove_elements_by_indices(DEC, mask_pointsrc)
PI_new = remove_elements_by_indices(PI, mask_pointsrc)
ENERGY_new = pi2keV(PI_new)
EXPOSURE_new = remove_elements_by_indices(EXPOSURE, mask_pointsrc)
print("Events before masking ",len(RA))
print("Events after masking ",RA_new.shape," ",ENERGY_new.shape," ",EXPOSURE_new.shape)
print("Total EXPOSURE NEW ",np.sum(EXPOSURE_new))
print("EXPOSURE NEW Min Max",np.min(EXPOSURE_new)," ",np.max(EXPOSURE_new))


#https://erosita.mpe.mpg.de/edr/eROSITATechnical/calibration.html
#FWHM from 50 eV (@lowest E) to 160 eV (@10keV)
Nmap = 2000 # 6deg/30arcsec=1200
NEbins = 300

DeltaE = (np.max(ENERGY)-np.min(ENERGY))/NEbins
DeltaE_new = (np.max(ENERGY_new)-np.min(ENERGY_new))/NEbins
print("Min Max ENERGY  ",np.min(ENERGY)," ",np.max(ENERGY)," DeltaE ",DeltaE)
print("Min Max ENERGY NEW ",np.min(ENERGY_new)," ",np.max(ENERGY_new)," DeltaE ",DeltaE_new)


fig = plt.figure(figsize=(8,7))
norm1 = mpl.colors.LogNorm()
plt.title('Counts map')
h2 = plt.hist2d(RA_new, DEC_new, bins=Nmap, cmap='viridis', norm=norm1) #, weights=1/EXPOSURE_new
plt.xlabel('RA [deg]', size=20)
plt.ylabel('DEC [deg]', size=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.colorbar()
plt.tight_layout()
plt.savefig('Test/MT_Files/Countmap.png')


fig = plt.figure(figsize=(8,7))
norm1 = mpl.colors.LogNorm()
plt.title('Counts map')
h2 = plt.hist2d(RA, DEC, bins=Nmap, cmap='viridis', norm=norm1) #, weights=1/EXPOSURE_new
plt.xlabel('RA [deg]', size=20)
plt.ylabel('DEC [deg]', size=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.colorbar()
plt.tight_layout()
plt.savefig('Test/MT_Files/Countmap_unmasked.png')

cnt1, cnt1_xedges = np.histogram(ENERGY, bins=NEbins)
cnt2, cnt2_xedges = np.histogram(ENERGY_new, bins=NEbins)
fig = plt.figure(figsize=(8,7))
h1_x = (cnt1_xedges[1:]+cnt1_xedges[:-1])/2
h2_x = (cnt2_xedges[1:]+cnt2_xedges[:-1])/2
plt.figure()
plt.errorbar(h1_x, cnt1, yerr = np.sqrt(cnt1), label='w/ sources')
plt.errorbar(h2_x, cnt2, yerr = np.sqrt(cnt2), label='w/o sources')
plt.xlabel('Energy (keV)', size=18)
plt.ylabel('cnt', size=18)
plt.yscale('log')
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Test/MT_Files/Counts.pdf')



fig = plt.figure(figsize=(8,7))
norm1 = mpl.colors.LogNorm()
plt.title('Rate map')
h2 = plt.hist2d(RA_new, DEC_new, bins=Nmap, weights=1/EXPOSURE_new, cmap='viridis', norm=norm1) #
plt.xlabel('RA [deg]', size=20)
plt.ylabel('DEC [deg]', size=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.colorbar()
plt.tight_layout()
plt.savefig('Test/MT_Files/Ratemap.png')

fig = plt.figure(figsize=(8,7))
norm1 = mpl.colors.LogNorm()
plt.title('Rate map unmasked')
h2 = plt.hist2d(RA, DEC, bins=Nmap, weights=1/EXPOSURE, cmap='viridis', norm=norm1) #
plt.xlabel('RA [deg]', size=20)
plt.ylabel('DEC [deg]', size=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.colorbar()
plt.tight_layout()
plt.savefig('Test/MT_Files/Ratemap_unmasked.png')




h1, h1_xedges = np.histogram(ENERGY, weights=1/EXPOSURE, bins=NEbins)
h2, h2_xedges = np.histogram(ENERGY_new, weights=1/EXPOSURE_new, bins=NEbins)
h1_x = (h1_xedges[1:]+h1_xedges[:-1])/2
h2_x = (h2_xedges[1:]+h2_xedges[:-1])/2
plt.figure()
plt.errorbar(h1_x, h1, yerr = np.sqrt(cnt1)/np.mean(EXPOSURE), label='w/ sources')
plt.errorbar(h2_x, h2, yerr = np.sqrt(cnt2)/np.mean(EXPOSURE_new), label='w/o sources')
plt.xlabel('Energy (keV)', size=18)
plt.ylabel('cnt/s', size=18)
plt.yscale('log')
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Test/MT_Files/Rate.pdf')

#plt.show()


np.savetxt('Test/MT_Files/rate_masked.dat', list(zip(h2_xedges[:-1],h2_xedges[1:],h2,np.sqrt(cnt2)/np.mean(EXPOSURE_new))),fmt='%1.3f')
np.savetxt('Test/MT_Files/counts_masked.dat', list(zip(cnt2_xedges[:-1],cnt2_xedges[1:],cnt2 )),fmt='%1.3f')



print("DeltaOmega [sr] ",DeltaOmega)
#dataAeff = np.genfromtxt('/Users/marcotaoso/Documents/2024/eROSITA/testLMC/Data/eROSITA_aeff.csv', delimiter=',',filling_values=0)
## Compute Aeff for each event
#AeffENERGY=10**(np.interp(np.log10(ENERGY), np.log10(dataAeff[:,0]), np.log10(dataAeff[:,1]) ) )
#AeffENERGY_new=10**(np.interp(np.log10(ENERGY_new), np.log10(dataAeff[:,0]), np.log10(dataAeff[:,1]) ) )
flux = 1./(EXPOSURE*DeltaOmega*DeltaE)
flux_new = 1./(EXPOSURE_new*DeltaOmega*DeltaE_new)
norm1 = mpl.colors.LogNorm()
hflux, hflux_xedges = np.histogram(ENERGY_new, weights=flux_new, bins=NEbins)
hfluxerr, hfluxerr_xedges = np.histogram(ENERGY_new, weights=flux_new**2, bins=NEbins)
hflux_x = (hflux_xedges[1:]+hflux_xedges[:-1])/2
h2flux, h2flux_xedges = np.histogram(ENERGY, weights=flux, bins=NEbins)
h2fluxerr, h2fluxerr_xedges = np.histogram(ENERGY, weights=flux**2, bins=NEbins)
h2flux_x = (h2flux_xedges[1:]+h2flux_xedges[:-1])/2
plt.figure()
plt.errorbar(h2flux_x, h2flux, yerr = np.sqrt(h2fluxerr), label='w/ sources')
plt.errorbar(hflux_x, hflux, yerr = np.sqrt(hfluxerr), label='w/o sources')
plt.xlabel('Energy (keV)', size=18)
plt.ylabel('photons/s/keV/sr', size=18)
plt.yscale('log')
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Test/MT_Files/Flux.pdf')

np.savetxt('Test/MT_Files/flux_masked.dat', list(zip(hflux_xedges[:-1],hflux_xedges[1:],hflux,np.sqrt(hfluxerr))),fmt='%1.3f')

