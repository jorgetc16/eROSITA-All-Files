#!/usr/bin/env python
import os
import sys
import numpy
import matplotlib.pyplot as plt
import matplotlib
#import pyatomdb
from threeML import *
# from ixpe.ixpeOGIPSpectrumLike import ixpeOGIPSpectrumLike
from astromodels import *
from astromodels.functions.functions_1D.apec import APEC
from astropy.io import fits
from scipy.stats import gaussian_kde
from threeML.utils.OGIP.pha import PHAII
from threeML.plugins.OGIPLike import OGIPLike
from threeML.data_list import DataList

import warnings
warnings.filterwarnings("ignore")

# matplotlib.rcParams["savefig.directory"] = "/Users/mnegro/Desktop/GRB221009A/IXPE/02250101/figures_bkg/"

datadir = '/Users/mnegro/MyDocuments/XRAY_PROBE/eROSITA/LMC_DR1/LMCspectra/'

def rename_fits_extension(input_file):
    # Open the FITS file in update mode
    with fits.open(input_file, mode='update') as hdulist:
        # Check if the first extension is named "SPECRESP_1PX"
        if hdulist[1].header['EXTNAME'] == 'SPECRESP_1PX':
            # Rename the first extension to "MATRIX"
            hdulist[1].header['EXTNAME'] = 'MATRIX'
            print("Extension name changed to 'MATRIX'")
        else:
            print(f"First extension name is not 'SPECRESP_1PX', it is {hdulist[1].header['EXTNAME']}")
        # Save changes to the original file
        hdulist.flush()
        print(f"File {input_file} has been updated")

SRC = None
BAYES = False
ADD_NUSTAR_XMM = False

#--- plotting params ---

# backscal = 28.184783156960265
ra_src, dec_src = 80.89417, -69.75611
fit_emin, fit_emax  = 0.9, 9. #0.3, 80.
erange = '%f-%f' % (fit_emin, fit_emax)

obs_list = [datadir + 'LMC_TM1_pha.fits',
            datadir + 'LMC_TM2_pha.fits',
            datadir + 'LMC_TM3_pha.fits',
            datadir + 'LMC_TM4_pha.fits',
            datadir + 'LMC_TM5_pha.fits',
            datadir + 'LMC_TM6_pha.fits',
            datadir + 'LMC_TM7_pha.fits',
            ]
arf_list = [datadir + 'tm1_arf_filter_000101v02.fits',
            datadir + 'tm2_arf_filter_000101v02.fits',
            datadir + 'tm3_arf_filter_000101v02.fits',
            datadir + 'tm4_arf_filter_000101v02.fits',
            datadir + 'tm5_arf_filter_000101v02.fits',
            datadir + 'tm6_arf_filter_000101v02.fits',
            datadir + 'tm7_arf_filter_000101v02.fits',
            ]
rmf_list = [datadir + 'tm1_rmf_141103v02.fits',
            datadir + 'tm2_rmf_141103v02.fits',
            datadir + 'tm3_rmf_141103v02.fits',
            datadir + 'tm4_rmf_141103v02.fits',
            datadir + 'tm5_rmf_141103v02.fits',
            datadir + 'tm6_rmf_141103v02.fits',
            datadir + 'tm7_rmf_141103v02.fits',
            ]

# spectral_model1 = TbAbs() * (Powerlaw() + Blackbody()) + Constant()
# spectral_model1 = TbAbs() * APEC() + TbAbs() * Powerlaw() + Constant()#+ APEC() 
spectral_model1 =  TbAbs() * ( APEC() +  APEC() + Powerlaw()) + Constant()


if SRC == None:
    label = '2APECsPL'
    norm1 = 1.20e-2
    index1 = -4
    BBkT = 0.1 #keV
    BBnorm = 1.
    nh = 1 #e22 cm-2

for rmf in rmf_list:
    rename_fits_extension(rmf)

erosita_data = []
data = DataList()
for i, (obs_, arf_, rmf_) in enumerate(zip(obs_list, arf_list, rmf_list)):
    plugin = OGIPLike('erosita_tm%s' % str(i+1), observation=obs_, response=rmf_, arf_file=arf_, spectrum_number=i+1)
    plugin.set_active_measurements("0.5-10")
    erosita_data.append(plugin)
    data.insert(plugin)
# data = Datalist(*erosita_data)

print('------------------------------------------------------')
source1 = PointSource('APEC', ra_src, dec_src, spectral_shape=spectral_model1)

    
model = Model(source1)
model.display(complete=True)

model.APEC.spectrum.main.composite.NH_1.prior = Log_uniform_prior(lower_bound=0.001, upper_bound=10)
model.APEC.spectrum.main.composite.NH_1 = nh 
model.APEC.spectrum.main.composite.NH_1.free = True

model.APEC.spectrum.main.composite.K_2.prior = Log_uniform_prior(lower_bound=0.00001, upper_bound=100)
model.APEC.spectrum.main.composite.K_2 = 1
model.APEC.spectrum.main.composite.K_2.free = True
model.APEC.spectrum.main.composite.kT_2.prior = Uniform_prior(lower_bound=0.08, upper_bound=64.)
model.APEC.spectrum.main.composite.kT_2 = 0.6
model.APEC.spectrum.main.composite.kT_2.free = True
model.APEC.spectrum.main.composite.abund_2 = 1.
model.APEC.spectrum.main.composite.abund_2.free = False
model.APEC.spectrum.main.composite.redshift_2 = 0.0
model.APEC.spectrum.main.composite.redshift_2.free = False

model.APEC.spectrum.main.composite.K_3.prior = Log_uniform_prior(lower_bound=0.00001, upper_bound=100)
model.APEC.spectrum.main.composite.K_3 = 1
model.APEC.spectrum.main.composite.K_3.free = True
model.APEC.spectrum.main.composite.kT_3.prior = Uniform_prior(lower_bound=0.08, upper_bound=64.)
model.APEC.spectrum.main.composite.kT_3 = 0.6
model.APEC.spectrum.main.composite.kT_3.free = True
model.APEC.spectrum.main.composite.abund_3 = 1.
model.APEC.spectrum.main.composite.abund_3.free = False
model.APEC.spectrum.main.composite.redshift_3 = 0.0
model.APEC.spectrum.main.composite.redshift_2.free = False

# model.APEC.spectrum.main.composite.NH_3.prior = Log_uniform_prior(lower_bound=0.001, upper_bound=10)
# model.APEC.spectrum.main.composite.NH_3 = nh 
# model.APEC.spectrum.main.composite.NH_3.free = True
model.APEC.spectrum.main.composite.K_4.prior = Log_uniform_prior(lower_bound=0.00001, upper_bound=1)
model.APEC.spectrum.main.composite.K_4 = norm1
model.APEC.spectrum.main.composite.K_4.free = True
model.APEC.spectrum.main.composite.index_4.prior = Uniform_prior(lower_bound=-5.0, upper_bound=0.0)
model.APEC.spectrum.main.composite.index_4 = 1.45
model.APEC.spectrum.main.composite.index_4.free = True
model.APEC.spectrum.main.composite.k_5.prior = Uniform_prior(lower_bound=0.001, upper_bound=1)
model.APEC.spectrum.main.composite.k_5 = 0.01
model.APEC.spectrum.main.composite.k_5.free = True
    
print('------------------------------------------------------')

# source1 = PointSource('PL1', ra_src, dec_src, spectral_shape=spectral_model1)

    
# model = Model(source1)
# model.display(complete=True)

# model.PL1.spectrum.main.composite.K_2.prior = Log_uniform_prior(lower_bound=0.00001, upper_bound=1)
# model.PL1.spectrum.main.composite.K_2 = norm1
# model.PL1.spectrum.main.composite.K_2.free = True
# model.PL1.spectrum.main.composite.index_2.prior = Uniform_prior(lower_bound=-5.0, upper_bound=0.0)
# model.PL1.spectrum.main.composite.index_2 = index1
# model.PL1.spectrum.main.composite.index_2.free = True
# model.PL1.spectrum.main.composite.NH_1.prior = Log_uniform_prior(lower_bound=0.001, upper_bound=10)
# model.PL1.spectrum.main.composite.NH_1 = nh 
# model.PL1.spectrum.main.composite.NH_1.free = True
# model.PL1.spectrum.main.composite.kT_3 = BBkT
# model.PL1.spectrum.main.composite.kT_3.free = True
# model.PL1.spectrum.main.composite.kT_3.prior = Uniform_prior(lower_bound=0.001, upper_bound=10)
# model.PL1.spectrum.main.composite.K_3 = BBnorm
# model.PL1.spectrum.main.composite.K_3.free = True
# model.PL1.spectrum.main.composite.K_3.prior = Log_uniform_prior(lower_bound=0.00001, upper_bound=100)
# model.PL1.spectrum.main.composite.k_4.prior = Uniform_prior(lower_bound=0.001, upper_bound=1)
# model.PL1.spectrum.main.composite.k_4 = 0.01
# model.PL1.spectrum.main.composite.k_4.free = True

print('------------------------------------------------------')

# model.display(complete=True)

# f = open('%s/plots/%s_output.txt'%(obsid, label),'w')
# sys.stdout = f

if BAYES:
    like = BayesianAnalysis(model, data)
    # AreaCorrections={'DU1':1,'DU2':0.95,'DU3':0.90}
    # ixpe.applyAreaCorrection(AreaCorrections)
    # ixpe.fitAreaCorrection(model)
    like.set_sampler("emcee")
    # like.sampler.setup(min_num_live_points=400)
    like.sampler.setup(n_iterations=100)
    like.sample()
    like.restore_median_fit()
    like.results.corner_plot()
    plt.tight_layout()
    plt.show()
    # plt.savefig('../plots/Corner_plot_%s.png'%ring, dpi=300)

else:
    like = JointLikelihood(model, data, verbose=False)    
    param_df, like_df = like.fit()
    
figI = display_spectrum_model_counts(like, data=['erosita_tm1',
                                                 'erosita_tm2',
                                                 'erosita_tm3',
                                                 'erosita_tm4',
                                                 'erosita_tm5',
                                                 'erosita_tm6',
                                                 'erosita_tm7'],
                                                 show_background=False)




# fig, ax = plt.subplots(figsize=(7., 5))#figsize=(6,4.5)
# ax.spines['top'].set_visible(True)
# ax.spines['right'].set_visible(True)

# abs = TbAbs() 
# bb  = Blackbody() 
# pl  = Powerlaw() 
# c = Constant()

# x = numpy.linspace(0.5, 10, 1000)

# abs.NH = model.PL1.spectrum.main.composite.NH_1.value 
# pl.indes = model.PL1.spectrum.main.composite.index_2.value 
# pl.K = model.PL1.spectrum.main.composite.K_2.value 
# bb.kT = model.PL1.spectrum.main.composite.kT_3.value 
# bb.K = model.PL1.spectrum.main.composite.K_3.value 
# c.K = model.PL1.spectrum.main.composite.k_4.value
# cc = numpy.full(len(x), c.K)

# figI.axes[0].plot(x, abs(x)*bb(x), color='k', alpha=1, linewidth=1, linestyle='--')
# figI.axes[0].plot(x, abs(x)*pl(x), color='k', alpha=1, linewidth=1, linestyle='-.')
# figI.axes[0].plot(x, cc, color='k', alpha=1, linewidth=1, linestyle=':')
# figI.axes[0].plot(x, abs(x) * (bb(x) + pl(x)) + cc, color='k', alpha=1, linewidth=2)

figI.axes[0].set_xlim(0.5, 10)
figI.axes[0].set_ylim(10, 1500)
plt.savefig('%s_Ispec.png'%(label), dpi=300)

plt.show()
