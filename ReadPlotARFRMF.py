from astropy.io import fits
# import healpy as hp
from matplotlib import rc
import matplotlib.patheffects as path_effects
from matplotlib import pyplot as plt
import os 
import pandas as pd
from xspec import *

# Xset.allowPrompting = False # keeps pyxspec from hanging, waiting for a response to a prompt


#arf1 = fits.open("/Users/marcotaoso/Documents/2024/eROSITA/FromMichela/Spectra_irfs_v3/tm1_arf_filter_000101v02.fits")
#arf1.info()
arf1test = fits.open("/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_120_ARF_00001.fits")['SPECRESP']
#print(arf1test.header)
arf1ELOW = arf1test.data.field('ENERG_LO')#keV
arf1EHI = arf1test.data.field('ENERG_HI')#keV
arf1Aeff = arf1test.data.field('SPECRESP')#cm2
arf1E = (arf1ELOW + arf1EHI)/2.

arf2test = fits.open("/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_220_ARF_00001.fits")['SPECRESP']
arf2ELOW = arf2test.data.field('ENERG_LO')#keV
arf2EHI = arf2test.data.field('ENERG_HI')#keV
arf2Aeff = arf2test.data.field('SPECRESP')#cm2
arf2E = (arf2ELOW + arf2EHI)/2.

arf3test = fits.open("/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_320_ARF_00001.fits")['SPECRESP']
arf3ELOW = arf3test.data.field('ENERG_LO')#keV
arf3EHI = arf3test.data.field('ENERG_HI')#keV
arf3Aeff = arf3test.data.field('SPECRESP')#cm2
arf3E = (arf3ELOW + arf3EHI)/2.

arf4test = fits.open("/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_420_ARF_00001.fits")['SPECRESP']
arf4ELOW = arf4test.data.field('ENERG_LO')#keV
arf4EHI = arf4test.data.field('ENERG_HI')#keV
arf4Aeff = arf4test.data.field('SPECRESP')#cm2
arf4E = (arf4ELOW + arf4EHI)/2.

arf5test = fits.open("/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_520_ARF_00001.fits")['SPECRESP']
arf5ELOW = arf5test.data.field('ENERG_LO')#keV
arf5EHI = arf5test.data.field('ENERG_HI')#keV
arf5Aeff = arf5test.data.field('SPECRESP')#cm2
arf5E = (arf5ELOW + arf5EHI)/2.

arf6test = fits.open("/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_620_ARF_00001.fits")['SPECRESP']
arf6ELOW = arf6test.data.field('ENERG_LO')#keV
arf6EHI = arf6test.data.field('ENERG_HI')#keV
arf6Aeff = arf6test.data.field('SPECRESP')#cm2
arf6E = (arf6ELOW + arf6EHI)/2.

arf7test = fits.open("/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_720_ARF_00001.fits")['SPECRESP']
arf7ELOW = arf7test.data.field('ENERG_LO')#keV
arf7EHI = arf7test.data.field('ENERG_HI')#keV
arf7Aeff = arf7test.data.field('SPECRESP')#cm2
arf7E = (arf7ELOW + arf7EHI)/2.

arf0test = fits.open("/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_020_ARF_00001.fits")['SPECRESP']
arf0ELOW = arf0test.data.field('ENERG_LO')#keV
arf0EHI = arf0test.data.field('ENERG_HI')#keV
arf0Aeff = arf0test.data.field('SPECRESP')#cm2
arf0E = (arf0ELOW + arf0EHI)/2.


# fig, ax = plt.subplots()
# fig.set_size_inches(21.5, 13.5)
# # # Plot using Matplotlib:
# plt.plot(arf1E, arf1Aeff,label='TM1')
# plt.plot(arf2E, arf2Aeff,label='TM2')
# plt.plot(arf3E, arf3Aeff,label='TM3')
# plt.plot(arf4E, arf4Aeff,label='TM4')
# plt.plot(arf5E, arf5Aeff,label='TM5')
# plt.plot(arf6E, arf6Aeff,label='TM6')
# plt.plot(arf7E, arf7Aeff,label='TM7')
# plt.plot(arf7E, arf1Aeff+arf2Aeff+arf3Aeff+arf4Aeff+arf5Aeff+arf6Aeff+arf7Aeff,label='Sum',color='black',linestyle='solid')
# plt.plot(arf0E, arf0Aeff,label='All',color='gray',linestyle='dashed')

# #ax.axhline(y=10)
# ax.set_xlabel('Energy (keV)')
# ax.set_ylabel(r'A_{eff} [cm^2]')
# ax.set_xscale("log")
# ax.set_yscale("log")
# #ax.set_ylim((0.001,0.1))
# ax.grid()
# ax.legend()

# # plt.show()

s1 = Spectrum("/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_020_BackgrSpec_00001.fits")
try:
    b1 = s1.background
except Exception:
    b1 = ''
b1='' # spectrum has no background
try: 
    r1 = s1.response
except Exception:
    r1='' # no response defined
try:
    arfFileName = r1.arf
except Exception:
    arfFileName = '' # no arf defined either


if not r1:
    s1.response = "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_020_RMF_00001.fits"
    print("New Response = {0}".format(s1.response.rmf))

s1.response.arf = "/home/jortecal/GitHub/eRosita/Test/Files_srctool/LMC/srctoolout_020_ARF_00001.fits"
print("New ARF = {0}".format(s1.response.arf))

# check the data
AllData.show()

Plot.device = "/xs"

Plot.xAxis = "keV"
Plot("data")

# reset the plot
Plot.commands = ()

# https://gist.github.com/ivvv/716799056b4aadc87e41097472edbd20 (look here if you have also model with sum of components)
#https://github.com/elastufka/solar_all_purpose/blob/main/pyXspec%20test.ipynb
#Save the data
save_data = 'specfit_ch.qdp'
if (os.path.isfile(save_data)):
    os.remove(save_data)
# following Andy Beardmore idea
Plot.device = '/null'
Plot.add = True
Plot.addCommand(f'wd {save_data}')
Plot.xAxis = "channel"
Plot("ldata") 
# # Read the data
names = ['e','de','rate','rate_err']
df = pd.read_table('specfit_ch.qdp',skiprows=3,names=names, delimiter=' ')
print(df)

# fig, ax = plt.subplots(figsize=(10,6))
# # # Plot using Matplotlib:
# ax.errorbar(df.e, df.rate, xerr=df.de,yerr=df.rate_err,fmt='.',markersize='5',label='data')
# ax.axhline(y=10)
# ax.set_xlabel('Channel')
# ax.set_ylabel(r'counts/s')
# #ax.set_xscale("log")
# #ax.set_yscale("log")
# #ax.set_ylim((0.001,0.1))
# ax.grid()
# ax.legend()
# plt.show()