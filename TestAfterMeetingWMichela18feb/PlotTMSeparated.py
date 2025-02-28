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
import sys
import time
from datetime import timedelta


RMF_FILES = ["/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_120_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_220_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_320_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_420_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_520_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_620_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_720_RMF_00001.fits",
            ]

rmf_ebounds_1 = fits.open(RMF_FILES[0])['EBOUNDS'].data 
rmf_channel_1 = rmf_ebounds_1.field('CHANNEL')  
rmf_emin_1 = rmf_ebounds_1.field('E_MIN')  
rmf_emax_1 = rmf_ebounds_1.field('E_MAX') 
rmf_ebinedges_1 = np.append(rmf_emin_1, rmf_emax_1[-1])

print(rmf_emin_1, rmf_emax_1)
exit()
Energy_1 = ((rmf_ebinedges_1[1:]+rmf_ebinedges_1[:-1])/2)

binsizes = rmf_emax_1 - rmf_emin_1
# print(binsizes)
# exit()
#compute the size of the energy bins

DeltaOmega=2*np.pi*(1-np.cos(3*np.pi/180.)) #LMC

# rmf_ebounds_2 = fits.open(RMF_FILES[1])['EBOUNDS'].data
# rmf_channel_2 = rmf_ebounds_2.field('CHANNEL')
# rmf_emin_2 = rmf_ebounds_2.field('E_MIN')
# rmf_emax_2 = rmf_ebounds_2.field('E_MAX')
# rmf_ebinedges_2 = np.append(rmf_emin_2, rmf_emax_2[-1])

# Energy_2 = ((rmf_ebinedges_2[1:]+rmf_ebinedges_2[:-1])/2)


# rmf_ebounds_3 = fits.open(RMF_FILES[2])['EBOUNDS'].data
# rmf_channel_3 = rmf_ebounds_3.field('CHANNEL')
# rmf_emin_3 = rmf_ebounds_3.field('E_MIN')
# rmf_emax_3 = rmf_ebounds_3.field('E_MAX')
# rmf_ebinedges_3 = np.append(rmf_emin_3, rmf_emax_3[-1])

# Energy_3 = ((rmf_ebinedges_3[1:]+rmf_ebinedges_3[:-1])/2)

# print(Energy_1, Energy_2, Energy_3)
# exit()
RATE_TMS_FILES_srctool = ["/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_120_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_220_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_320_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_420_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_520_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_620_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_720_SourceSpec_00001.fits",
                ]

RATE_TMS_FILES = ["/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMCFiles/TMSeparatedFiles/LMC_TM1_pha_backregNONE.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMCFiles/TMSeparatedFiles/LMC_TM2_pha_backregNONE.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMCFiles/TMSeparatedFiles/LMC_TM3_pha_backregNONE.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMCFiles/TMSeparatedFiles/LMC_TM4_pha_backregNONE.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMCFiles/TMSeparatedFiles/LMC_TM5_pha_backregNONE.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMCFiles/TMSeparatedFiles/LMC_TM6_pha_backregNONE.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMCFiles/TMSeparatedFiles/LMC_TM7_pha_backregNONE.fits",
                ]

channel_TMS = [np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025)]

channel_TMS_srctool = [np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025),
               np.arange(1, 1025)]

exposuretime_srctool = [fits.open(RATE_TMS_FILES_srctool[0])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[1])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[2])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[3])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[4])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[5])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[6])['SPECTRUM'].header['EXPOSURE']]


rate_TMS_srctool = [fits.open(RATE_TMS_FILES_srctool[0])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[0],
            fits.open(RATE_TMS_FILES_srctool[1])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[1],
            fits.open(RATE_TMS_FILES_srctool[2])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[2],
            fits.open(RATE_TMS_FILES_srctool[3])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[3],
            fits.open(RATE_TMS_FILES_srctool[4])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[4],
            fits.open(RATE_TMS_FILES_srctool[5])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[5],
            fits.open(RATE_TMS_FILES_srctool[6])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[6]]



rate_TMS = [fits.open(RATE_TMS_FILES[0])['SPECTRUM'].data['RATE'],
            fits.open(RATE_TMS_FILES[1])['SPECTRUM'].data['RATE'],
            fits.open(RATE_TMS_FILES[2])['SPECTRUM'].data['RATE'],
            fits.open(RATE_TMS_FILES[3])['SPECTRUM'].data['RATE'],
            fits.open(RATE_TMS_FILES[4])['SPECTRUM'].data['RATE'],
            fits.open(RATE_TMS_FILES[5])['SPECTRUM'].data['RATE'],
            fits.open(RATE_TMS_FILES[6])['SPECTRUM'].data['RATE']]

stat_err_TMS = [fits.open(RATE_TMS_FILES[0])['SPECTRUM'].data['STAT_ERR'],
                fits.open(RATE_TMS_FILES[0])['SPECTRUM'].data['STAT_ERR'],
                fits.open(RATE_TMS_FILES[0])['SPECTRUM'].data['STAT_ERR'],
                fits.open(RATE_TMS_FILES[0])['SPECTRUM'].data['STAT_ERR'],
                fits.open(RATE_TMS_FILES[0])['SPECTRUM'].data['STAT_ERR'],
                fits.open(RATE_TMS_FILES[0])['SPECTRUM'].data['STAT_ERR'],
                fits.open(RATE_TMS_FILES[0])['SPECTRUM'].data['STAT_ERR']]

stat_err_TMS_srctool =  [np.sqrt(fits.open(RATE_TMS_FILES_srctool[0])['SPECTRUM'].data['COUNTS'])/exposuretime_srctool[0],
            np.sqrt(fits.open(RATE_TMS_FILES_srctool[1])['SPECTRUM'].data['COUNTS'])/exposuretime_srctool[1],
            np.sqrt(fits.open(RATE_TMS_FILES_srctool[2])['SPECTRUM'].data['COUNTS'])/exposuretime_srctool[2],
            np.sqrt(fits.open(RATE_TMS_FILES_srctool[3])['SPECTRUM'].data['COUNTS'])/exposuretime_srctool[3],
            np.sqrt(fits.open(RATE_TMS_FILES_srctool[4])['SPECTRUM'].data['COUNTS'])/exposuretime_srctool[4],
            np.sqrt(fits.open(RATE_TMS_FILES_srctool[5])['SPECTRUM'].data['COUNTS'])/exposuretime_srctool[5],
            np.sqrt(fits.open(RATE_TMS_FILES_srctool[6])['SPECTRUM'].data['COUNTS'])/exposuretime_srctool[6]]



plt.figure()
plt.errorbar(Energy_1, rate_TMS[0]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS[0], label='TM 1')
plt.errorbar(Energy_1, rate_TMS[1]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS[1], label='TM 2')
plt.errorbar(Energy_1, rate_TMS[2]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS[2], label='TM 3')
plt.errorbar(Energy_1, rate_TMS[3]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS[3], label='TM 4')
plt.errorbar(Energy_1, rate_TMS[4]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS[4], label='TM 5')
plt.errorbar(Energy_1, rate_TMS[5]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS[5], label='TM 6')
plt.errorbar(Energy_1, rate_TMS[6]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS[6], label='TM 7')
plt.errorbar(Energy_1, rate_TMS[6]/binsizes/DeltaOmega+rate_TMS[5]/binsizes/DeltaOmega+rate_TMS[4]/binsizes/DeltaOmega+rate_TMS[3]/binsizes/DeltaOmega + rate_TMS[2]/binsizes/DeltaOmega + rate_TMS[1]/binsizes/DeltaOmega + rate_TMS[0]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS[0] +  stat_err_TMS[1] + stat_err_TMS[2] + stat_err_TMS[3] +  stat_err_TMS[4] + stat_err_TMS[5] + stat_err_TMS[6], label='Sum Total')
# plt.xlabel('Channel (PI)', size=20)
plt.xlabel('Energy (keV)', size=20)
plt.ylabel(r'ph s$^{-1}$ keV$^{-1}$ sr$^{-1}$', size=20)
plt.xlim(0.75, 11)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=15)
plt.tight_layout()

plt.show()



# plt.figure()
# plt.errorbar(channel_TMS_srctool[0], rate_TMS_srctool[0], fmt='.', yerr = stat_err_TMS_srctool[0], label='TM 1')
# plt.errorbar(channel_TMS_srctool[1], rate_TMS_srctool[1], fmt='.', yerr = stat_err_TMS_srctool[1], label='TM 2')
# plt.errorbar(channel_TMS_srctool[2], rate_TMS_srctool[2], fmt='.', yerr = stat_err_TMS_srctool[2], label='TM 3')
# plt.errorbar(channel_TMS_srctool[3], rate_TMS_srctool[3], fmt='.', yerr = stat_err_TMS_srctool[3], label='TM 4')
# plt.errorbar(channel_TMS_srctool[4], rate_TMS_srctool[4], fmt='.', yerr = stat_err_TMS_srctool[4], label='TM 5')
# plt.errorbar(channel_TMS_srctool[5], rate_TMS_srctool[5], fmt='.', yerr = stat_err_TMS_srctool[5], label='TM 6')
# plt.errorbar(channel_TMS_srctool[6], rate_TMS_srctool[6], fmt='.', yerr = stat_err_TMS_srctool[6], label='TM 7')
# plt.errorbar(channel_TMS_srctool[6], rate_TMS_srctool[6]+rate_TMS_srctool[5]+rate_TMS_srctool[4]+rate_TMS_srctool[3] + rate_TMS_srctool[2] + rate_TMS_srctool[1] + rate_TMS_srctool[0], fmt='.', yerr = stat_err_TMS_srctool[0] +  stat_err_TMS_srctool[1] + stat_err_TMS_srctool[2] + stat_err_TMS_srctool[3] +  stat_err_TMS_srctool[4] + stat_err_TMS_srctool[5] + stat_err_TMS_srctool[6], label='Sum Total')
# plt.xlabel('Channel (PI)', size=20)
# plt.ylabel(r'ph s$^{-1}$', size=20)
# plt.yscale('log')
# plt.legend(fontsize=15)
# plt.tight_layout()

# plt.show()