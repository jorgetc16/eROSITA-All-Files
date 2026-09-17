import warnings

warnings.simplefilter("ignore")
import numpy as np
import matplotlib.pyplot as plt
np.seterr(all="ignore")
from threeML import *
from astromodels import *
from astromodels.functions.function import Function1D, FunctionMeta
from astropy import units as u
from threeML.utils.OGIP.response import OGIPResponse
from threeML.io.package_data import get_path_of_data_file

LMC3deg_pha = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001/srctoolout_120_SourceSpec_00001.fits"
LMC3deg_rmf = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001/srctoolout_120_RMF_00001.fits"
LMC3deg_arf = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001/srctoolout_120_ARF_00001.fits"
eROSITA_ParticleBackground_TM1 = "/home/jortecal/GitHub/eRosita/CookBook/TM1_FWC_v1.0.dat"

spectral_model1 =  TbAbs() * (APEC() + Powerlaw()) + APEC() 
###############################################################################################################################
# Inclusion of the particle background model FWC for TM1
########################################################################################################################################
import re

file_path = "/home/jortecal/Downloads/TM1_FWC_v1.0.dat"

with open(file_path, "r") as f:
    lines = f.readlines()

# Extract the model definition (first line)

# Extract parameter values from the remaining lines
parameters = []
for line in lines[1:]:
    if line.strip():  # Skip empty lines
        parameters.append([float(x) for x in re.split(r"\s+", line.strip())])


# Define the components of the model
constant = Constant()
gaussians = [Gaussian() for _ in range(14)]  # 14 Gaussian components
powerlaw1 = Powerlaw()
powerlaw2 = Powerlaw()

# Define the expfac component
class ExpFac(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        Exponential modification of a spectrum.

        M(E) = 1 + A * exp(-f * E) for E > E_c
               1                   for E <= E_c

    parameters :
        A :
            desc : Amplitude of the exponential modification
            initial value : 1.0
            min : -100.0
            max : 100.0
            delta : 0.1
        f :
            desc : Exponential factor
            initial value : 1.0
            min : -100.0
            max : 100.0
            delta : 0.1
        E_c :
            desc : Start energy of the modification
            initial value : 1.0
            min : -100.0
            max : 100.0
            delta : 0.1
    """

    def _set_units(self, x_unit, y_unit):
        self.A.unit = y_unit
        self.f.unit = 1 / x_unit
        self.E_c.unit = x_unit

    def evaluate(self, x, A, f, E_c):
        return 1 + A * np.exp(-f * x) * (x > E_c)

# Define the bkn2pow component
class Bkn2Pow(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        A three-segment broken power law.

        A(E) = K * E^(-Gamma_1) for E <= E_break1
               K * E_break1^(Gamma_2 - Gamma_1) * E^(-Gamma_2) for E_break1 < E <= E_break2
               K * E_break1^(Gamma_2 - Gamma_1) * E_break2^(Gamma_3 - Gamma_2) * E^(-Gamma_3) for E > E_break2

    parameters :
        K :
            desc : Normalization at 1 keV
            initial value : 1.0
            min : 1e-10
            max : 1e10
            delta : 0.1
        Gamma1 :
            desc : Photon index for E <= E_break1
            initial value : -1.5
            min : -10.0
            max : 10.0
            delta : 0.1
        E_break1 :
            desc : First break energy in keV
            initial value : 10.0
            min : 1e-20
            max : 1e3
            delta : 1.0
        Gamma2 :
            desc : Photon index for E_break1 < E <= E_break2
            initial value : -2.0
            min : -10.0
            max : 10.0
            delta : 0.1
        E_break2 :
            desc : Second break energy in keV
            initial value : 100.0
            min : 1e-20
            max : 1e4
            delta : 1.0
        Gamma3 :
            desc : Photon index for E > E_break2
            initial value : -2.5
            min : -10.0
            max : 10.0
            delta : 0.1
    """

    def _set_units(self, x_unit, y_unit):
        self.K.unit = y_unit
        self.Gamma1.unit = u.dimensionless_unscaled
        self.E_break1.unit = x_unit
        self.Gamma2.unit = u.dimensionless_unscaled
        self.E_break2.unit = x_unit
        self.Gamma3.unit = u.dimensionless_unscaled

    def evaluate(self, x, K, Gamma1, E_break1, Gamma2, E_break2, Gamma3):
        # Define the three segments of the broken power law
        segment1 = K * x**(-Gamma1) * (x <= E_break1)
        segment2 = (
            K
            * E_break1 ** (Gamma2 - Gamma1)
            * x**(-Gamma2)
            * (x > E_break1)
            * (x <= E_break2)
        )
        segment3 = (
            K
            * E_break1 ** (Gamma2 - Gamma1)
            * E_break2 ** (Gamma3 - Gamma2)
            * x**(-Gamma3)
            * (x > E_break2)
        )
        return segment1 + segment2 + segment3

# Instantiate the expfac component
expfac = ExpFac()

# Instantiate the bkn2pow component
bkn2pow = Bkn2Pow()

# Combine the components into a single model
spectral_model_Particle_Background = (
    constant
    * (
        gaussians[0]
        + gaussians[1]
        + gaussians[2]
        + gaussians[3]
        + gaussians[4]
        + gaussians[5]
        + gaussians[6]
        + gaussians[7]
        + gaussians[8]
        + gaussians[9]
        + gaussians[10]
        + gaussians[11]
        + gaussians[12]
        + gaussians[13]
    
    + expfac * (bkn2pow + powerlaw1 + powerlaw2))
)

combined_model = spectral_model1

# Create a point source with the spectral model
ModelPlusParticleBackground = PointSource("source", 0, 0, spectral_shape=combined_model)

# Create the final model
model_combined = Model(ModelPlusParticleBackground)

# Display the model
# model_combined.display(complete=True)


#***********************************************************************************************************************
#***********************************************************************************************************************
#***********************************************************************************************************************
#***********************************************************************************************************************

# Assign parameter values from the file

# Assign parameters to the constant component
# model_combined.source.spectrum.main.composite.k_1.value = parameters[0][0]  # value of the constant factor
# model_combined.source.spectrum.main.composite.k_1.delta = parameters[0][1]  # fit delta
# model_combined.source.spectrum.main.composite.k_1.bounds = (parameters[0][2], parameters[0][5])  # min and max
# model_combined.source.spectrum.main.composite.k_1.free = False  # fixed parameter
# # Assign parameters to the Gaussian components
# for i in range(14):  # 14 Gaussian components
#     model_combined.source.spectrum.main.composite[f"mu_{i+2}"].value = parameters[1 + i * 3][0]  # mu_i
#     model_combined.source.spectrum.main.composite[f"mu_{i+2}"].bounds = (parameters[1 + i * 3][2], parameters[1 + i * 3][5])

#     model_combined.source.spectrum.main.composite[f"sigma_{i+2}"].bounds = (parameters[2 + i * 3][2], parameters[2 + i * 3][5])
#     model_combined.source.spectrum.main.composite[f"sigma_{i+2}"].value = parameters[2 + i * 3][0]  # sigma_i
#     model_combined.source.spectrum.main.composite[f"sigma_{i+2}"].free = False  # fixed parameter

#     model_combined.source.spectrum.main.composite[f"F_{i+2}"].value = parameters[3 + i * 3][0]  # F_i
#     model_combined.source.spectrum.main.composite[f"F_{i+2}"].bounds = (parameters[3 + i * 3][2], parameters[3 + i * 3][5])

# # Assign parameters to the expfac component
# model_combined.source.spectrum.main.composite.A_16.value = parameters[-13][0]  # A_16
# model_combined.source.spectrum.main.composite.A_16.bounds = (parameters[-13][2], parameters[-13][5])

# model_combined.source.spectrum.main.composite.f_16.value = parameters[-12][0]  # f_16
# model_combined.source.spectrum.main.composite.f_16.bounds = (parameters[-12][2], parameters[-12][5])

# model_combined.source.spectrum.main.composite.E_c_16.value = parameters[-11][0]  # E_c_16
# model_combined.source.spectrum.main.composite.E_c_16.bounds = (parameters[-11][2], parameters[-11][5])

# # Assign parameters to the broken power law (bkn2pow)

# model_combined.source.spectrum.main.composite.Gamma1_17.value = parameters[-10][0] # Gamma1_17
# model_combined.source.spectrum.main.composite.Gamma1_17.bounds = (parameters[-10][2], parameters[-10][5])


# model_combined.source.spectrum.main.composite.E_break1_17.value = parameters[-9][0] # E_break1_17
# model_combined.source.spectrum.main.composite.E_break1_17.bounds = (parameters[-9][2], parameters[-9][5])


# model_combined.source.spectrum.main.composite.Gamma2_17.value = parameters[-8][0] # Gamma2_17
# model_combined.source.spectrum.main.composite.Gamma2_17.bounds = (parameters[-8][2], parameters[-8][5])


# model_combined.source.spectrum.main.composite.E_break2_17.value = parameters[-7][0] # E_break2_17
# model_combined.source.spectrum.main.composite.E_break2_17.bounds = (parameters[-7][2], parameters[-7][5])


# model_combined.source.spectrum.main.composite.Gamma3_17.value = parameters[-6][0] # Gamma3_17
# model_combined.source.spectrum.main.composite.Gamma3_17.bounds = (parameters[-6][2], parameters[-6][5])


# model_combined.source.spectrum.main.composite.K_17.bounds = (parameters[-5][2], parameters[-5][5])
# model_combined.source.spectrum.main.composite.K_17.value = parameters[-5][0]  # K_17

# # Assign parameters to the first power law (powerlaw1)
# model_combined.source.spectrum.main.composite.index_18.bounds = (parameters[-4][2], parameters[-4][5])
# model_combined.source.spectrum.main.composite.index_18.value = parameters[-4][0]  # index_18

# model_combined.source.spectrum.main.composite.K_18.bounds = (1e-20, parameters[-3][5])
# model_combined.source.spectrum.main.composite.K_18.value = parameters[-3][0]  # K_18

# # Assign parameters to the second power law (powerlaw2)
# model_combined.source.spectrum.main.composite.index_19.bounds = (parameters[-2][2], parameters[-2][5])
# model_combined.source.spectrum.main.composite.index_19.value = parameters[-2][0]  # index_19

# model_combined.source.spectrum.main.composite.K_19.bounds = (1e-20, parameters[-1][5])
# model_combined.source.spectrum.main.composite.K_19.value = 1e-20  # K_19

###################################################################################################################################################
###################################################################################################################################################
# We set the parameters of the APEC model and the Powerlaw ****Check with Marco's*** 

model_combined.source.spectrum.main.composite.NH_1 = 4.5e-2
model_combined.source.spectrum.main.composite.NH_1.free = False


model_combined.source.spectrum.main.composite.index_3.value = 1.46
model_combined.source.spectrum.main.composite.index_3.free = False


model_combined.source.spectrum.main.composite.kT_2.value = 0.2
model_combined.source.spectrum.main.composite.kT_2.free = False
model_combined.source.spectrum.main.composite.abund_2.value = 1.
model_combined.source.spectrum.main.composite.abund_2.free = False



model_combined.source.spectrum.main.composite.kT_4.value = 0.1
model_combined.source.spectrum.main.composite.kT_4.free = False
model_combined.source.spectrum.main.composite.abund_4.value = 1.
model_combined.source.spectrum.main.composite.abund_4.free = False

###################################################################################################################################################
###################################################################################################################################################
# print(model_combined.source.spectrum.main.composite)
# print("TbaBs Evaluated at 1.0 = ",TbAbs().evaluate_at(1.0))
# print(f"Model value at E={1.0} keV: {model_combined.source.spectrum.main.composite.evaluate_at(1.0)}")
# model_combined.display(complete=True)

ogip = OGIPLike("ogip", observation=LMC3deg_pha, response=LMC3deg_rmf,  arf_file=LMC3deg_arf)

ogip.set_active_measurements("0.3-1")


#***********************************************************************************************************************




# jl = JointLikelihood(model_combined,DataList(ogip))

# result = jl.fit()

# tbabs = TbAbs()
# tbabs.NH = model_combined.source.spectrum.main.composite.NH_1.value  # Use the same NH value
# print(f"TbAbs at E={1.0} keV: {tbabs(1.0)}")




spectrum_generator_1 = DispersionSpectrumLike.from_function(
    "s1",
    source_function=ModelPlusParticleBackground,
    response=OGIPResponse(LMC3deg_rmf),
)
fig = spectrum_generator_1.view_count_spectrum()
plt.show()
# powerlaw = Powerlaw()
# powerlaw.index = model_combined.source.spectrum.main.composite.index_3.value
# print(f"Powerlaw at E={1.0} keV: {powerlaw(1.0)}")
# fig = display_spectrum_model_counts(jl,data_color='blue',model_color='red')

# fig.set_size_inches(13,10)

# ax = fig.get_axes()[0]

# ax.set_xlim(left=0.2,right=11)
# ax.set_xlabel('Energy [keV]',fontsize=28)
# ax.set_ylabel('Rate [counts s$^{-1}$ keV$^{-1}$]',fontsize=28)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(left=0.2,right=1.1)

# plt.show()