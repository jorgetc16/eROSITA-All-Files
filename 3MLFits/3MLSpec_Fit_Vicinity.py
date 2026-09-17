import warnings

warnings.simplefilter("ignore")
import numpy as np
import matplotlib.pyplot as plt
np.seterr(all="ignore")
from threeML import *
from astromodels import *
from astromodels.functions.function import Function1D, FunctionMeta

LMC3deg_pha = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_120_SourceSpec_00001.fits"
LMC3deg_rmf = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_120_RMF_00001.fits"
LMC3deg_arf = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_120_ARF_00001.fits"
eROSITA_ParticleBackground_TM1 = "/home/jortecal/GitHub/eRosita/CookBook/TM1_FWC_v1.0.dat"

Background_and_DM = Quadratic() + Gaussian()


DM_Line = 3.10
DM_Sigma = 99.87e-3/(2*np.sqrt(2*np.log(2)))

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

gaussians = [Gaussian() for _ in range(14)]  # 14 Gaussian components


# Combine the components into a single model
Spec_Lines_Instrumental = (
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
)


# Create a point source with the spectral model
LinesInstrumental = PointSource("Lines_Inst", 0, 0, spectral_shape=Spec_Lines_Instrumental)

Bckg_DM = PointSource("Backg_DM", 0, 0, spectral_shape=Background_and_DM)

# Create the final model
model_combined = Model(Bckg_DM)
# Set the model to be additive  

# Display the model
# model_combined.display(complete=True)
# exit()
#***********************************************************************************************************************
#***********************************************************************************************************************
#***********************************************************************************************************************
#***********************************************************************************************************************

# Assign parameter values from the file

# Assign parameters to the Gaussian components
# for i in range(14):  # 14 Gaussian components
#     model_combined.sources["Lines_Inst"].spectrum.main.composite[f"mu_{i+1}"].value = parameters[1 + i * 3][0]  # mu_i
#     model_combined.sources["Lines_Inst"].spectrum.main.composite[f"mu_{i+1}"].bounds = (parameters[1 + i * 3][2], parameters[1 + i * 3][5])
#     model_combined.sources["Lines_Inst"].spectrum.main.composite[f"mu_{i+1}"].free = False  # fixed parameter

#     model_combined.sources["Lines_Inst"].spectrum.main.composite[f"sigma_{i+1}"].bounds = (parameters[2 + i * 3][2], parameters[2 + i * 3][5])
#     model_combined.sources["Lines_Inst"].spectrum.main.composite[f"sigma_{i+1}"].value = parameters[2 + i * 3][0]  # sigma_i
#     model_combined.sources["Lines_Inst"].spectrum.main.composite[f"sigma_{i+1}"].free = False  # fixed parameter

#     model_combined.sources["Lines_Inst"].spectrum.main.composite[f"F_{i+1}"].value = parameters[3 + i * 3][0]  # F_i
#     model_combined.sources["Lines_Inst"].spectrum.main.composite[f"F_{i+1}"].bounds = (parameters[3 + i * 3][2], parameters[3 + i * 3][5])

model_combined.sources["Backg_DM"].spectrum.main.composite["mu_2"].value = DM_Line
model_combined.sources["Backg_DM"].spectrum.main.composite["mu_2"].free = False

model_combined.sources["Backg_DM"].spectrum.main.composite["sigma_2"].value = DM_Sigma
model_combined.sources["Backg_DM"].spectrum.main.composite["sigma_2"].free = False

model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].free = True

# model_combined.display(complete=True)
ogip_arf = OGIPLike("ogip_arf", observation=LMC3deg_pha, response=LMC3deg_rmf,  arf_file=LMC3deg_arf)



Range_Min, Range_Max = DM_Line - 5*DM_Sigma, DM_Line + 5*DM_Sigma
ogip_arf.set_active_measurements(f"{Range_Min}-{Range_Max}")

# ogip.set_active_measurements("0.3-9.5")


#***********************************************************************************************************************




jl = JointLikelihood(model_combined,DataList(ogip_arf))


# Perform the fit
result, log_likelihood = jl.fit()

# Extract the desired values
F_2_value = model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].value

# Extract the 90% confidence level upper limit for F_2
errors = jl.get_errors()  # Default confidence level (likely 90%)
F_2_limit_90 = F_2_value + errors.loc["Backg_DM.spectrum.main.composite.F_2", "positive_error"]


# Save the results to a .txt file
output_file = "/home/jortecal/GitHub/eRosita/3MLFits/results.txt"
with open(output_file, "w") as f:
    f.write(f"F_2 value: {F_2_value}\n")
    f.write(f"90% C.L. upper limit on F_2: {F_2_limit_90}\n")
    f.write(f"-log(likelihood): {log_likelihood}\n")

print(f"Results saved to {output_file}")


# Define the range of F_2 values for the scan
F_2_values = np.linspace(0, 0.005, 50)  # Adjust the range and step as needed
chi_squared_values = []

# Perform the likelihood scan
for F_2 in F_2_values:
    # Fix F_2 to the current value
    model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].value = F_2
    model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].free = False  # Fix the parameter

    # Recalculate the likelihood
    jl = JointLikelihood(model_combined, DataList(ogip_arf))
    result, log_likelihood = jl.fit()

    # Debugging: Inspect the structure of log_likelihood
    print("Log likelihood structure:", log_likelihood)
    print("Type of log_likelihood:", type(log_likelihood))

    # Extract the total -log(likelihood) value from the DataFrame
    if isinstance(log_likelihood, pd.DataFrame):
        chi_squared_values.append(log_likelihood.loc["total", "-log(likelihood)"])
    else:
        raise ValueError("Unexpected format for log_likelihood. Check its structure.")

# Restore F_2 to be free
model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].free = True

# Save the results to a file
output_file = "/home/jortecal/GitHub/eRosita/3MLFits/chi_squared_scan.txt"
with open(output_file, "w") as f:
    f.write("F_2_value\tchi_squared\n")
    for F_2, chi2 in zip(F_2_values, chi_squared_values):
        f.write(f"{F_2}\t{chi2}\n")

print(f"Chi-squared scan results saved to {output_file}")

# Plot the chi-squared as a function of F_2
plt.figure(figsize=(10, 6))
plt.plot(F_2_values, chi_squared_values, marker="o", linestyle="-", color="blue")
plt.xlabel("F_2 Value", fontsize=14)
plt.ylabel("-log(Likelihood) or Chi-squared", fontsize=14)
plt.title("Likelihood Scan for F_2", fontsize=16)
plt.grid(True)
plt.show()


