import warnings

warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.seterr(all="ignore")
from threeML import *
from astromodels import *
from astromodels.functions.function import Function1D, FunctionMeta
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

LMC3deg_pha = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_120_SourceSpec_00001.fits"
LMC3deg_rmf = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_120_RMF_00001.fits"
LMC3deg_arf = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_120_ARF_00001.fits"
eROSITA_ParticleBackground_TM1 = "/home/jortecal/GitHub/eRosita/CookBook/TM1_FWC_v1.0.dat"

Background_and_DM = Quadratic() + Gaussian()

Bckg_DM = PointSource("Backg_DM", 0, 0, spectral_shape=Background_and_DM)

# Create the final model
model_combined = Model(Bckg_DM)

# Create the data 
LMCeROSITA= OGIPLike("LMCeROSITA", observation=LMC3deg_pha, response=LMC3deg_rmf,  arf_file=LMC3deg_arf)

# File containing DM_Line and DM_Sigma values
dm_lines_file = "/home/jortecal/GitHub/eRosita/3MLFits/DM_Lines_response_Full.txt"

# Read the DM_Line and DM_Sigma values from the file
dm_data = pd.read_csv(dm_lines_file, delimiter=',', header=None, names=["DM_Line", "DM_Sigma"])

# Prepare output file
output_file = "/home/jortecal/GitHub/eRosita/3MLFits/DM_Lines_F2_results.txt"
with open(output_file, "w") as f_out:
    f_out.write("DM_Line\tDM_Sigma\tF_2_at_2.71\n")

# Lists to store results for plotting
dm_lines = []
f2_values = []

# Iterate over each DM_Line and DM_Sigma
for _, row in dm_data.iterrows():
    DM_Line = row["DM_Line"]
    DM_Sigma = row["DM_Sigma"]*1e-3 / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

    print('*' * 50)
    print(f"Processing DM_Line = {DM_Line}, DM_Sigma = {DM_Sigma}")
    print('*' * 50)

    model_combined.sources["Backg_DM"].spectrum.main.composite["mu_2"].value = DM_Line
    model_combined.sources["Backg_DM"].spectrum.main.composite["mu_2"].free = False

    model_combined.sources["Backg_DM"].spectrum.main.composite["sigma_2"].value = DM_Sigma
    model_combined.sources["Backg_DM"].spectrum.main.composite["sigma_2"].free = False

    model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].free = True
    model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].bounds = (0, 100)

    # Create the data 
    LMCeROSITA= OGIPLike("LMCeROSITA", observation=LMC3deg_pha, response=LMC3deg_rmf,  arf_file=LMC3deg_arf)



    Range_Min, Range_Max = DM_Line - 5*DM_Sigma, DM_Line + 5*DM_Sigma
    LMCeROSITA.set_active_measurements(f"{Range_Min}-{Range_Max}")

    # Perform an initial fit with F_2 free
    jl_initial = JointLikelihood(model_combined, DataList(LMCeROSITA))
    result_initial, log_likelihood_initial = jl_initial.fit(quiet=True)

    chi_squared_min = log_likelihood_initial.loc["total", "-log(likelihood)"] * 2

    # Extract the best-fit value of F_2
    F_2_best_fit = model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].value
    print(f"Best-fit F_2 value: {F_2_best_fit:.15g}")


    # Fix F_2 to the current value
    if F_2_best_fit == 0:
        F_2_best_fit = 1e-2
    else:
        New_F_2 = F_2_best_fit*1e3
    model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].value = New_F_2
    model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].free = False  # Fix the parameter

    # Set initial bounds for F_2
    lower_bound = 0
    upper_bound = New_F_2

    # Ensure the initial upper bound results in Delta Chi-squared > 2.71
    while True:
        model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].value = upper_bound
        jl = JointLikelihood(model_combined, DataList(LMCeROSITA))
        result, log_likelihood = jl.fit(quiet=True)
        chi_squared = log_likelihood.loc["total", "-log(likelihood)"] * 2
        delta_chi_squared = chi_squared - chi_squared_min

        if delta_chi_squared > 2.71:
            break
        upper_bound *= 10  # Double the upper bound if Delta Chi-squared is too low


    while True:
        # Set F_2 to the midpoint of the current bounds
        New_F_2 = (lower_bound + upper_bound) / 2
        model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].value = New_F_2

        # Perform the fit
        jl = JointLikelihood(model_combined, DataList(LMCeROSITA))
        result, log_likelihood = jl.fit(quiet=True)
        chi_squared = log_likelihood.loc["total", "-log(likelihood)"] * 2
        delta_chi_squared = chi_squared - chi_squared_min

        # print(f"Current F_2: {New_F_2}, Chi-squared: {chi_squared}, Delta Chi-squared: {delta_chi_squared}")

        # Check if Delta Chi-squared is close to 2.71
        if np.abs(delta_chi_squared - 2.71) < 1e-2:  # Tolerance of 0.01
            F_2_bound = New_F_2
            break

        # Adjust bounds based on Delta Chi-squared
        if delta_chi_squared > 2.71:
            upper_bound = New_F_2  # Decrease the upper bound
        else:
            lower_bound = New_F_2  # Increase the lower bound

    # Save the result for this DM_Line and DM_Sigma
    with open(output_file, "a") as f_out:
        f_out.write(f"{DM_Line}\t{DM_Sigma}\t{F_2_bound}\n")

    # Store results for plotting
    dm_lines.append(DM_Line)
    f2_values.append(F_2_bound)

    print(f"Processed DM_Line = {DM_Line}, F_2_at_2.71 = {F_2_bound}")

print(f"Results saved to {output_file}")

# Plot F_2 as a function of DM_Line
plt.figure(figsize=(10, 6))
plt.plot(dm_lines, f2_values, marker="o", linestyle="-", color="blue")
plt.xlabel("DM Line (keV)", fontsize=14)
plt.ylabel("F_2 at Delta Chi^2 = 2.71", fontsize=14)
plt.title("F_2 as a Function of DM Line", fontsize=16)
plt.grid(True)
plt.show()


