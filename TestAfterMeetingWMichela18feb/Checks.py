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
from astropy.table import Table


def pi2keV(pi, pimin=200, pimax=10000, emin=0.5, emax=10.0):
    m = (emin - emax) / (pimin - pimax)
    q = emin - m*pimin
    en = m*pi + q
    return en

RMF_FILES_src = ["/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_020_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_120_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_220_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_320_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_420_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_520_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_620_RMF_00001.fits",
             "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_720_RMF_00001.fits",
            ]


RMF_FILES_Michela = ["/home/jortecal/GitHub/eRosita/StuffFromMichela/tm1_rmf_141103v02.fits",
            "/home/jortecal/GitHub/eRosita/StuffFromMichela/tm2_rmf_141103v02.fits",
            "/home/jortecal/GitHub/eRosita/StuffFromMichela/tm3_rmf_141103v02.fits",
            "/home/jortecal/GitHub/eRosita/StuffFromMichela/tm4_rmf_141103v02.fits",
            "/home/jortecal/GitHub/eRosita/StuffFromMichela/tm5_rmf_141103v02.fits",
            "/home/jortecal/GitHub/eRosita/StuffFromMichela/tm6_rmf_141103v02.fits",
            "/home/jortecal/GitHub/eRosita/StuffFromMichela/tm7_rmf_141103v02.fits",

            ]


ARF_FILES_srctool = ["/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_120_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_220_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_320_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_420_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_520_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_620_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_1800arcsec/srctoolout_720_ARF_00001.fits",
            ]



ARF_FILES_Michela = ["/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_36arcsec/srctoolout_120_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_36arcsec/srctoolout_220_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_36arcsec/srctoolout_320_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_36arcsec/srctoolout_420_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_36arcsec/srctoolout_520_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_36arcsec/srctoolout_620_ARF_00001.fits",
            "/home/jortecal/GitHub/eRosita/TestAfterMeetingWMichela18feb/LMC_srctool_backregNONE/srctoolout_000_SourceProducts_00001_36arcsec/srctoolout_720_ARF_00001.fits",
            ]


############################################    RMF ENERGY    ############################################
rmf_ebounds_1_src = fits.open(RMF_FILES_src[0])['EBOUNDS'].data 
rmf_channel_1_src = rmf_ebounds_1_src.field('CHANNEL')  
rmf_emin_1_src = rmf_ebounds_1_src.field('E_MIN')  
rmf_emax_1_src = rmf_ebounds_1_src.field('E_MAX') 
rmf_ebinedges_1_src = np.append(rmf_emin_1_src, rmf_emax_1_src[-1])

Energy_1_src = ((rmf_ebinedges_1_src[1:]+rmf_ebinedges_1_src[:-1])/2)

binsizes_src = rmf_emax_1_src - rmf_emin_1_src
print('Number of energy bins src tool: ', len(Energy_1_src))
print('First bin size is: ', binsizes_src[0]
      , 'keV, last bin size is: ', binsizes_src[-1], 'keV')
# print an histogram with the binsizes
plt.figure()
plt.plot(Energy_1_src, binsizes_src, label='RMF src tool')
plt.xlabel('Energy (keV)', size=20)
plt.ylabel('Bin size (keV)', size=20)
plt.title('RMF from src tool', size=20)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()


            

# rmf_ebounds_1_Michela = fits.open(RMF_FILES_Michela[0])['EBOUNDS'].data 
# rmf_channel_1_Michela = rmf_ebounds_1_Michela.field('CHANNEL')  
# rmf_emin_1_Michela = rmf_ebounds_1_Michela.field('E_MIN')  
# rmf_emax_1_Michela = rmf_ebounds_1_Michela.field('E_MAX') 
# rmf_ebinedges_1_Michela = np.append(rmf_emin_1_Michela, rmf_emax_1_Michela[-1])

# Energy_1_Michela = ((rmf_ebinedges_1_Michela[1:]+rmf_ebinedges_1_Michela[:-1])/2)

# binsizes_Michela = rmf_emax_1_Michela - rmf_emin_1_Michela

# # plot ratios
# plt.figure()
# plt.plot(Energy_1_src, binsizes_src/binsizes_Michela, label='RMF/Michela')
# plt.legend()
# plt.xscale('log')
# plt.show()  


############################################    RMF MATRIX    ############################################

# matrix_src_1 = Table(fits.open(RMF_FILES_src[1])['MATRIX'].data)

# new_matrix_src_1 = []

# for (energy_lo, energy_hi, ngrp, fchan, nchan, mat) in matrix_src_1:
    
#     mat_new = np.concatenate((mat, np.zeros(1024-nchan))).tolist()
#     new_matrix_src_1.append(mat_new)
    
# new_matrix_src_1 = np.array(new_matrix_src_1)

# plt.matshow(new_matrix_src_1)


# matrix_Michela_1 = Table(fits.open(RMF_FILES_Michela[1])['MATRIX'].data)

# new_matrix_Michela_1 = []

# for (energy_lo, energy_hi, ngrp, fchan, nchan, mat) in matrix_Michela_1:
    
#     mat_new = np.concatenate((mat, np.zeros(1024-nchan))).tolist()
#     new_matrix_Michela_1.append(mat_new)
    
# new_matrix_Michela_1 = np.array(new_matrix_Michela_1)

# plt.matshow(new_matrix_Michela_1)

# plt.show()


############################################    ARF    ############################################

arf_total = fits.open(ARF_FILES_srctool[0])['SPECRESP'].data.field('SPECRESP') + fits.open(ARF_FILES_srctool[1])['SPECRESP'].data.field('SPECRESP') + fits.open(ARF_FILES_srctool[2])['SPECRESP'].data.field('SPECRESP') + fits.open(ARF_FILES_srctool[3])['SPECRESP'].data.field('SPECRESP') + fits.open(ARF_FILES_srctool[4])['SPECRESP'].data.field('SPECRESP') + fits.open(ARF_FILES_srctool[5])['SPECRESP'].data.field('SPECRESP') + fits.open(ARF_FILES_srctool[6])['SPECRESP'].data.field('SPECRESP')
plt.figure()


plt.plot((fits.open(ARF_FILES_srctool[0])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_srctool[0])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_srctool[0])['SPECRESP'].data.field('SPECRESP'),'--', color='k', label='TM1')
plt.plot((fits.open(ARF_FILES_srctool[1])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_srctool[1])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_srctool[1])['SPECRESP'].data.field('SPECRESP'),':', color='r', label='TM2')
plt.plot((fits.open(ARF_FILES_srctool[2])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_srctool[2])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_srctool[2])['SPECRESP'].data.field('SPECRESP'),'-.', color='g', label='TM3')
plt.plot((fits.open(ARF_FILES_srctool[3])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_srctool[3])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_srctool[3])['SPECRESP'].data.field('SPECRESP'), color='b', label='TM4')
plt.plot((fits.open(ARF_FILES_srctool[4])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_srctool[4])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_srctool[4])['SPECRESP'].data.field('SPECRESP'),'--', color='orange', label='TM5')
plt.plot((fits.open(ARF_FILES_srctool[5])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_srctool[5])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_srctool[5])['SPECRESP'].data.field('SPECRESP'), color='y', label='TM6')
plt.plot((fits.open(ARF_FILES_srctool[6])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_srctool[6])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_srctool[6])['SPECRESP'].data.field('SPECRESP'),'-.', color='c', label='TM7')
plt.plot((fits.open(ARF_FILES_srctool[6])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_srctool[6])['SPECRESP'].data.field('ENERG_HI'))/2, arf_total, color='k', label='Total')
plt.title('ARF from src tool', size=20)
plt.xlabel('Energy (keV)', size=20)
plt.ylabel(r'A$_{eff} [$cm$^{2}$]', size=20)
plt.yscale('log')
plt.xscale('log')
plt.legend()

#Now the same but with Michela's ARF
arf_total_Mich = fits.open(ARF_FILES_Michela[0])['SPECRESP'].data.field('SPECRESP') + fits.open(ARF_FILES_Michela[1])['SPECRESP'].data.field('SPECRESP') + fits.open(ARF_FILES_Michela[2])['SPECRESP'].data.field('SPECRESP') + fits.open(ARF_FILES_Michela[3])['SPECRESP'].data.field('SPECRESP') + fits.open(ARF_FILES_Michela[4])['SPECRESP'].data.field('SPECRESP') + fits.open(ARF_FILES_Michela[5])['SPECRESP'].data.field('SPECRESP') + fits.open(ARF_FILES_Michela[6])['SPECRESP'].data.field('SPECRESP')
plt.figure()


plt.plot((fits.open(ARF_FILES_Michela[0])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_Michela[0])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_Michela[0])['SPECRESP'].data.field('SPECRESP'),'--', color='k', label='TM1')
plt.plot((fits.open(ARF_FILES_Michela[1])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_Michela[1])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_Michela[1])['SPECRESP'].data.field('SPECRESP'),':', color='r', label='TM2')
plt.plot((fits.open(ARF_FILES_Michela[2])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_Michela[2])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_Michela[2])['SPECRESP'].data.field('SPECRESP'),'-.', color='g', label='TM3')
plt.plot((fits.open(ARF_FILES_Michela[3])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_Michela[3])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_Michela[3])['SPECRESP'].data.field('SPECRESP'), color='b', label='TM4')
plt.plot((fits.open(ARF_FILES_Michela[4])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_Michela[4])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_Michela[4])['SPECRESP'].data.field('SPECRESP'),'--', color='orange', label='TM5')
plt.plot((fits.open(ARF_FILES_Michela[5])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_Michela[5])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_Michela[5])['SPECRESP'].data.field('SPECRESP'), color='y', label='TM6')
plt.plot((fits.open(ARF_FILES_Michela[6])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_Michela[6])['SPECRESP'].data.field('ENERG_HI'))/2, fits.open(ARF_FILES_Michela[6])['SPECRESP'].data.field('SPECRESP'),'-.', color='c', label='TM7')
plt.plot((fits.open(ARF_FILES_Michela[6])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_Michela[6])['SPECRESP'].data.field('ENERG_HI'))/2, arf_total_Mich, color='k', label='Total')
plt.title('ARF from Michela', size=20)
plt.xlabel('Energy (keV)', size=20)
plt.ylabel(r'A$_{eff} [$cm$^{2}$]', size=20)
plt.yscale('log')
plt.xscale('log')
plt.legend()
# plt.show()

#Now let's plot the ratio between the two total ARFs
plt.figure()


plt.plot((fits.open(ARF_FILES_srctool[0])['SPECRESP'].data.field('ENERG_LO')+fits.open(ARF_FILES_srctool[0])['SPECRESP'].data.field('ENERG_HI'))/2, arf_total/arf_total_Mich, color='k')
plt.xlabel('Energy (keV)', size=20)
plt.ylabel(r'A$_{eff, src}$/A$_{eff, Mic}$', size=20)
plt.yscale('log')
plt.xscale('log')
plt.show()
#Compare the binsizes and Energy
# plt.plot(Energy_1, binsizes, label='RMF')
# plt.plot(MarcoEnergy, MarcoBinsize, label='Marco')
# plt.plot(Energy_1, binsizes/MarcoBinsizes, label='RMF/Marco')
# plt.legend()
# plt.xscale('log')
# # plt.yscale('log')
# plt.show()