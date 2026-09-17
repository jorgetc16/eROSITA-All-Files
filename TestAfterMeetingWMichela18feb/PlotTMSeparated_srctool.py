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


RMF_FILES = ["/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_120_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_120_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_220_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_320_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_420_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_520_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_620_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_720_RMF_00001.fits",
            ]

rmf_ebounds_1 = fits.open(RMF_FILES[0])['EBOUNDS'].data 
rmf_channel_1 = rmf_ebounds_1.field('CHANNEL')  
rmf_emin_1 = rmf_ebounds_1.field('E_MIN')  
rmf_emax_1 = rmf_ebounds_1.field('E_MAX') 
rmf_ebinedges_1 = np.append(rmf_emin_1, rmf_emax_1[-1])

Energy_1 = ((rmf_ebinedges_1[1:]+rmf_ebinedges_1[:-1])/2)

binsizes = rmf_emax_1 - rmf_emin_1

#Remove the energies below 0.5 keV
# Energy_final = Energy_1[Energy_1 > 0.5]
# binsizes = binsizes[Energy_1 > 0.5]

# Calculate the bin widths, which we will use to convert the y-axis

# print(binsizes)
# exit()
#compute the size of the energy bins

# DeltaOmega=1 #LMC
arcsectodeg = 1/3600.
DeltaOmega=2*np.pi*(1-np.cos(1800*arcsectodeg*np.pi/180.)) #LMC

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
RATE_TMS_FILES_srctool = ["/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_020_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_120_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_220_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_320_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_420_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_520_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_620_SourceSpec_00001.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_720_SourceSpec_00001.fits",
                ]


# RATE_TMS_FILES_srctool_bckg = ["/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_020_BackgrSpec_00001.fits",
#                 "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_120_BackgrSpec_00001.fits",
#                 "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_220_BackgrSpec_00001.fits",
#                 "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_320_BackgrSpec_00001.fits",
#                 "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_420_BackgrSpec_00001.fits",
#                 "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_520_BackgrSpec_00001.fits",
#                 "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_620_BackgrSpec_00001.fits",
#                 "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/srctoolout_000_SourceProducts_00001_1800arcsecBackregNone/srctoolout_720_BackgrSpec_00001.fits",
#                 ]
RATE_TMS_FILES = ["/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/Results/TMSeparatedFiles/LMC_TM1_pha_backregNONE.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/Results/TMSeparatedFiles/LMC_TM2_pha_backregNONE.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/Results/TMSeparatedFiles/LMC_TM3_pha_backregNONE.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/Results/TMSeparatedFiles/LMC_TM4_pha_backregNONE.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/Results/TMSeparatedFiles/LMC_TM5_pha_backregNONE.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/Results/TMSeparatedFiles/LMC_TM6_pha_backregNONE.fits",
                "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/Results/TMSeparatedFiles/LMC_TM7_pha_backregNONE.fits",
                ]

channel_TMS = [np.arange(1, len(Energy_1)+1),
               np.arange(1, len(Energy_1)+1),
               np.arange(1, len(Energy_1)+1),
               np.arange(1, len(Energy_1)+1),
               np.arange(1, len(Energy_1)+1),
               np.arange(1, len(Energy_1)+1),
               np.arange(1, len(Energy_1)+1),]

channel_TMS_srctool = [np.arange(1, len(Energy_1)+1),
               np.arange(1, len(Energy_1)+1),
               np.arange(1, len(Energy_1)+1),
               np.arange(1, len(Energy_1)+1),
               np.arange(1, len(Energy_1)+1),
               np.arange(1, len(Energy_1)+1),
               np.arange(1, len(Energy_1)+1),]

exposuretime_srctool = [fits.open(RATE_TMS_FILES_srctool[0])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[1])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[2])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[3])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[4])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[5])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[6])['SPECTRUM'].header['EXPOSURE'],
                        fits.open(RATE_TMS_FILES_srctool[7])['SPECTRUM'].header['EXPOSURE']]

# exposuretime_srctool_bckg = [fits.open(RATE_TMS_FILES_srctool_bckg[0])['SPECTRUM'].header['EXPOSURE'],
#                         fits.open(RATE_TMS_FILES_srctool_bckg[1])['SPECTRUM'].header['EXPOSURE'],
#                         fits.open(RATE_TMS_FILES_srctool_bckg[2])['SPECTRUM'].header['EXPOSURE'],
#                         fits.open(RATE_TMS_FILES_srctool_bckg[3])['SPECTRUM'].header['EXPOSURE'],
#                         fits.open(RATE_TMS_FILES_srctool_bckg[4])['SPECTRUM'].header['EXPOSURE'],
#                         fits.open(RATE_TMS_FILES_srctool_bckg[5])['SPECTRUM'].header['EXPOSURE'],
#                         fits.open(RATE_TMS_FILES_srctool_bckg[6])['SPECTRUM'].header['EXPOSURE'],
#                         fits.open(RATE_TMS_FILES_srctool_bckg[7])['SPECTRUM'].header['EXPOSURE']]






rate_TMS_srctool = [fits.open(RATE_TMS_FILES_srctool[0])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[0],
            fits.open(RATE_TMS_FILES_srctool[1])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[1],
            fits.open(RATE_TMS_FILES_srctool[2])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[2],
            fits.open(RATE_TMS_FILES_srctool[3])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[3],
            fits.open(RATE_TMS_FILES_srctool[4])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[4],
            fits.open(RATE_TMS_FILES_srctool[5])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[5],
            fits.open(RATE_TMS_FILES_srctool[6])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[6],
            fits.open(RATE_TMS_FILES_srctool[7])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[7]]

# rate_TMS_srctool = [rate_TMS_srctool[Energy_1 > 0.5] for rate_TMS_srctool in rate_TMS_srctool]


# rate_TMS_srctool_bckg = [fits.open(RATE_TMS_FILES_srctool_bckg[0])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[0],
#             fits.open(RATE_TMS_FILES_srctool_bckg[1])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[1],
#             fits.open(RATE_TMS_FILES_srctool_bckg[2])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[2],
#             fits.open(RATE_TMS_FILES_srctool_bckg[3])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[3],
#             fits.open(RATE_TMS_FILES_srctool_bckg[4])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[4],
#             fits.open(RATE_TMS_FILES_srctool_bckg[5])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[5],
#             fits.open(RATE_TMS_FILES_srctool_bckg[6])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[6],
#             fits.open(RATE_TMS_FILES_srctool_bckg[7])['SPECTRUM'].data['COUNTS']/exposuretime_srctool[7]]

# rate_TMS_srctool_bckg = [rate_TMS_srctool_bckg[Energy_1 > 0.5] for rate_TMS_srctool_bckg in rate_TMS_srctool_bckg]


# rate_source_bckg = [rate_TMS_srctool[0]+rate_TMS_srctool_bckg[0],
#                     rate_TMS_srctool[1]+rate_TMS_srctool_bckg[1],
#                     rate_TMS_srctool[2]+rate_TMS_srctool_bckg[2],
#                     rate_TMS_srctool[3]+rate_TMS_srctool_bckg[3],
#                     rate_TMS_srctool[4]+rate_TMS_srctool_bckg[4],
#                     rate_TMS_srctool[5]+rate_TMS_srctool_bckg[5],
#                     rate_TMS_srctool[6]+rate_TMS_srctool_bckg[6],
#                     rate_TMS_srctool[7]+rate_TMS_srctool_bckg[7]]
                    
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
            np.sqrt(fits.open(RATE_TMS_FILES_srctool[6])['SPECTRUM'].data['COUNTS'])/exposuretime_srctool[6],
            np.sqrt(fits.open(RATE_TMS_FILES_srctool[7])['SPECTRUM'].data['COUNTS'])/exposuretime_srctool[7]]

# stat_err_TMS_srctool = [stat_err_TMS_srctool[Energy_1 > 0.5] for stat_err_TMS_srctool in stat_err_TMS_srctool]

# plt.figure()
# plt.errorbar(Energy_1, rate_TMS_srctool[0]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS_srctool[0]/binsizes/DeltaOmega, label='All TM0')
# plt.errorbar(Energy_1, rate_TMS_srctool[1]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS_srctool[1]/binsizes/DeltaOmega, label='TM 1')
# plt.errorbar(Energy_1, rate_TMS_srctool[2]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS_srctool[2]/binsizes/DeltaOmega, label='TM 2')
# plt.errorbar(Energy_1, rate_TMS_srctool[3]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS_srctool[3]/binsizes/DeltaOmega, label='TM 3')
# plt.errorbar(Energy_1, rate_TMS_srctool[4]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS_srctool[4]/binsizes/DeltaOmega, label='TM 4')
# plt.errorbar(Energy_1, rate_TMS_srctool[5]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS_srctool[5]/binsizes/DeltaOmega, label='TM 5')
# plt.errorbar(Energy_1, rate_TMS_srctool[6]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS_srctool[6]/binsizes/DeltaOmega, label='TM 6')
# plt.errorbar(Energy_1, rate_TMS_srctool[7]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS_srctool[7]/binsizes/DeltaOmega, label='TM 7')
# plt.errorbar(Energy_1, rate_TMS_srctool[6]/binsizes/DeltaOmega+rate_TMS_srctool[5]/binsizes/DeltaOmega+rate_TMS_srctool[4]/binsizes/DeltaOmega+rate_TMS_srctool[3]/binsizes/DeltaOmega + rate_TMS_srctool[2]/binsizes/DeltaOmega + rate_TMS_srctool[1]/binsizes/DeltaOmega + rate_TMS_srctool[7]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS_srctool[7]/binsizes/DeltaOmega +  stat_err_TMS_srctool[1]/binsizes/DeltaOmega + stat_err_TMS_srctool[2]/binsizes/DeltaOmega + stat_err_TMS_srctool[3]/binsizes/DeltaOmega +  stat_err_TMS_srctool[4]/binsizes/DeltaOmega + stat_err_TMS_srctool[5]/binsizes/DeltaOmega + stat_err_TMS_srctool[6]/binsizes/DeltaOmega, label='Sum All')
# # plt.xlabel('Channel (PI)', size=20)
# plt.xlabel('Energy (keV)', size=20)
# plt.ylabel(r'ph s$^{-1}$ keV$^{-1}$ sr$^{-1}$', size=20)
# plt.xlim(0.75, 11)
# plt.ylim(1e1, 1e6)
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(fontsize=15)
# plt.tight_layout()

# plt.show()


# plt.figure()
# plt.plot(Energy_1, rate_source_bckg[0]/binsizes/DeltaOmega, label='All TM0')
# # plt.plot(Energy_1, rate_TMS_srctool[1]/binsizes/DeltaOmega, label='TM 1')
# # plt.plot(Energy_1, rate_TMS_srctool[2]/binsizes/DeltaOmega, label='TM 2')
# # plt.plot(Energy_1, rate_TMS_srctool[3]/binsizes/DeltaOmega, label='TM 3')
# # plt.plot(Energy_1, rate_TMS_srctool[4]/binsizes/DeltaOmega, label='TM 4')
# # plt.plot(Energy_1, rate_TMS_srctool[5]/binsizes/DeltaOmega, label='TM 5')
# # plt.plot(Energy_1, rate_TMS_srctool[6]/binsizes/DeltaOmega, label='TM 6')
# # plt.plot(Energy_1, rate_TMS_srctool[7]/binsizes/DeltaOmega, label='TM 7')
# plt.plot(Energy_1, rate_TMS_srctool[6]/binsizes/DeltaOmega+rate_TMS_srctool[5]/binsizes/DeltaOmega+rate_TMS_srctool[4]/binsizes/DeltaOmega+rate_TMS_srctool[3]/binsizes/DeltaOmega + rate_TMS_srctool[2]/binsizes/DeltaOmega + rate_TMS_srctool[1]/binsizes/DeltaOmega + rate_TMS_srctool[7]/binsizes/DeltaOmega, label='Sum Total')
# # plt.xlabel('Channel (PI)', size=20)
# plt.xlabel('Energy (keV)', size=20)
# plt.ylabel(r'ph s$^{-1}$ keV$^{-1}$ sr$^{-1}$', size=20)
# plt.xlim(0.5, 11)
# plt.ylim(1e0, 1e8)
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(fontsize=15)
# plt.tight_layout()

# plt.show()

# plt.figure()
# plt.errorbar(channel_TMS_srctool[0], rate_TMS_srctool[0], fmt='.', yerr = stat_err_TMS_srctool[0], label='TM 1')
# plt.errorbar(channel_TMS_srctool[1], rate_TMS_srctool[1], fmt='.', yerr = stat_err_TMS_srctool[1], label='TM 2')
# plt.errorbar(channel_TMS_srctool[2], rate_TMS_srctool[2], fmt='.', yerr = stat_err_TMS_srctool[2], label='TM 3')
# plt.errorbar(channel_TMS_srctool[3], rate_TMS_srctool[3], fmt='.', yerr = stat_err_TMS_srctool[3], label='TM 4')
# plt.errorbar(channel_TMS_srctool[4], rate_TMS_srctool[4], fmt='.', yerr = stat_err_TMS_srctool[4], label='TM 5')
# plt.errorbar(channel_TMS_srctool[5], rate_TMS_srctool[5], fmt='.', yerr = stat_err_TMS_srctool[5], label='TM 6')
# plt.errorbar(channel_TMS_srctool[6], rate_TMS_srctool[6], fmt='.', yerr = stat_err_TMS_srctool[6], label='TM 7')
# plt.errorbar(channel_TMS_srctool[6], rate_TMS_srctool[6]+rate_TMS_srctool[5]+rate_TMS_srctool[4]+rate_TMS_srctool[3] + rate_TMS_srctool[2] + rate_TMS_srctool[1] + rate_TMS_srctool[0], fmt='.', yerr = stat_err_TMS_srctool[0] +  stat_err_TMS_srctool[1] + stat_err_TMS_srctool[2] + stat_err_TMS_srctool[3] +  stat_err_TMS_srctool[4] + stat_err_TMS_srctool[5] + stat_err_TMS_srctool[6], label='Sum Total')
# plt.plot(channel_TMS_srctool[0], rate_TMS_srctool[0], label='TM 1')
# plt.plot(channel_TMS_srctool[1], rate_TMS_srctool[1], label='TM 2')
# plt.plot(channel_TMS_srctool[2], rate_TMS_srctool[2], label='TM 3')
# plt.plot(channel_TMS_srctool[3], rate_TMS_srctool[3], label='TM 4')
# plt.plot(channel_TMS_srctool[4], rate_TMS_srctool[4], label='TM 5')
# plt.plot(channel_TMS_srctool[5], rate_TMS_srctool[5], label='TM 6')
# plt.plot(channel_TMS_srctool[6], rate_TMS_srctool[6], label='TM 7')
# plt.plot(channel_TMS_srctool[6], rate_TMS_srctool[6]+rate_TMS_srctool[5]+rate_TMS_srctool[4]+rate_TMS_srctool[3] + rate_TMS_srctool[2] + rate_TMS_srctool[1] + rate_TMS_srctool[0], label='Sum Total')

# plt.xlabel('Channel (PI)', size=20)
# plt.ylabel(r'ph s$^{-1}$', size=20)
# # plt.yscale('log')
# plt.legend(fontsize=15)
# plt.tight_layout()

# plt.show()

# read from .dat file with 4 columns: min_energy, max_energy, flux, flux_err

flux_masked_evtool_halfdeg = np.loadtxt('/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/Results/flux_masked.dat', unpack=True)
min_energy_masked_evtool = flux_masked_evtool_halfdeg[0]
max_energy_masked_evtool = flux_masked_evtool_halfdeg[1]
flux_masked_evtool = flux_masked_evtool_halfdeg[2]
flux_masked_evtool_err = flux_masked_evtool_halfdeg[3]

flux_evtool_halfdeg = np.loadtxt('/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/Results/flux_NOTmasked.dat', unpack=True)
min_energy_evtool = flux_evtool_halfdeg[0]
max_energy_evtool = flux_evtool_halfdeg[1]
flux_evtool = flux_evtool_halfdeg[2]
flux_evtool_err = flux_evtool_halfdeg[3]


flux_evtool_qcorr_halfdeg = np.loadtxt('/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_halfdeg_evtool/Results/flux_NOTmasked_q_corr.dat', unpack=True)
min_energy_evtool_qcorr = flux_evtool_qcorr_halfdeg[0]
max_energy_evtool_qcorr = flux_evtool_qcorr_halfdeg[1]
flux_evtool_qcorr = flux_evtool_qcorr_halfdeg[2]/binsizes
flux_evtool_qcorr_err = flux_evtool_qcorr_halfdeg[3]/binsizes

plt.figure()
# plt.errorbar((min_energy_evtool+max_energy_evtool)/2, flux_evtool/7, yerr = flux_evtool_err/7, fmt='.', label='evtool')
# plt.errorbar((min_energy_masked_evtool+max_energy_masked_evtool)/2, flux_masked_evtool, yerr = flux_masked_evtool_err, fmt='.', label='masked_evtool')
# plt.errorbar(Energy_1, (rate_TMS[1]+rate_TMS[2]+rate_TMS[3]+rate_TMS[4]+rate_TMS[5]+rate_TMS[6]+rate_TMS[0])/binsizes/DeltaOmega/7, fmt='.', yerr = (stat_err_TMS[1]+stat_err_TMS[2]+stat_err_TMS[3]+stat_err_TMS[4]+stat_err_TMS[5]+stat_err_TMS[6]+stat_err_TMS[0])/binsizes/DeltaOmega/7, label='evtool_rmf') 
plt.errorbar(Energy_1, rate_TMS_srctool[0]/binsizes/DeltaOmega, fmt='.', yerr = stat_err_TMS_srctool[0]/binsizes/DeltaOmega, label='srctool')
# plt.errorbar((min_energy_evtool_qcorr+max_energy_evtool_qcorr)/2, flux_evtool_qcorr/7, yerr = flux_evtool_qcorr_err/7, fmt='.', label='evtool_corr')
# plt.plot(Energy_1, (rate_TMS_srctool[0]/binsizes/DeltaOmega)/(flux_evtool_qcorr/7), label='srctool/evtool_corr')
# plt.plot(Energy_1, Energy_1/((min_energy_evtool_qcorr+max_energy_evtool_qcorr)/2), label='srctool/evtool_corr')

# print(len(flux_evtool_qcorr))
# print(len(Energy_final))
plt.xlabel('Energy (keV)', size=20)
# plt.ylabel(r'srctool/evtool_corr', size=20)
plt.ylabel(r'ph s$^{-1}$ keV$^{-1}$ sr$^{-1}$', size=20)
# plt.xlim(0.5, 11)
# plt.ylim(0.5, 1.5)
plt.xscale('log')
plt.yscale('log')
# plt.legend(fontsize=15)
plt.xticks(size=15)
plt.yticks(size=15)
plt.tight_layout()
plt.title('LMC 0.5 deg', size=20)
plt.show()