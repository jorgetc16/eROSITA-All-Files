import numpy as np
import pandas as pd
from threeML import *
from astromodels import *
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Define the paths to your spectrum, ARF, and RMF files
spectrum_file_full_mask = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_020_SourceSpec_00001.fits"
arf_file_full_mask = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_020_ARF_00001.fits"
rmf_file_full_mask = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001_cheesemask_circle_masked/srctoolout_020_RMF_00001.fits"

spectrum_file_std_mask = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001/srctoolout_020_SourceSpec_00001.fits"
arf_file_std_mask = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001/srctoolout_020_ARF_00001.fits"
rmf_file_std_mask = "/home/jortecal/GitHub/eRosita/LMC3Deg_JorgePC/srctoolout_000_SourceProducts_00001/srctoolout_020_RMF_00001.fits"

# Load the spectrum using 3ML
spectrum_plugin_full_mask = OGIPLike("spectrum", observation=spectrum_file_full_mask, arf_file=arf_file_full_mask, response=rmf_file_full_mask)
spectrum_plugin_std_mask = OGIPLike("spectrum", observation=spectrum_file_std_mask, arf_file=arf_file_std_mask, response=rmf_file_std_mask)

# Plot the spectrum
spectrum_plugin_full_mask.view_count_spectrum()

plt.grid()
plt.title("LMC 3 degree masked spectrum")

spectrum_plugin_std_mask.view_count_spectrum()
plt.title("Standard Mask Spectrum")

plt.grid()
plt.show()

# File containing DM_Line and DM_Sigma values
dm_lines_file = "/home/jortecal/GitHub/eRosita/3MLFits/DM_Lines_response.txt"

# Read the DM_Line and DM_Sigma values from the file
dm_data = pd.read_csv(dm_lines_file, delim_whitespace=True, header=None, names=["DM_Line", "DM_Sigma"])

# Prepare output file
output_file = "/home/jortecal/GitHub/eRosita/3MLFits/DM_Lines_F2_results.txt"
with open(output_file, "w") as f_out:
    f_out.write("DM_Line\tDM_Sigma\tF_2_at_2.71\n")

# List to store results for plotting
dm_lines = []
f2_values = []

# Iterate over each DM_Line and DM_Sigma
for _, row in dm_data.iterrows():
    DM_Line = row["DM_Line"]
    DM_Sigma = row["DM_Sigma"] / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

    # Update the model with the current DM_Line and DM_Sigma
    model_combined.sources["Backg_DM"].spectrum.main.composite["mu_2"].value = DM_Line
    model_combined.sources["Backg_DM"].spectrum.main.composite["mu_2"].free = False

    model_combined.sources["Backg_DM"].spectrum.main.composite["sigma_2"].value = DM_Sigma
    model_combined.sources["Backg_DM"].spectrum.main.composite["sigma_2"].free = False

    # Perform an initial fit with F_2 free
    jl_initial = JointLikelihood(model_combined, DataList(LMCeROSITA))
    result_initial, log_likelihood_initial = jl_initial.fit(quiet=True)

    # Extract the best-fit value of F_2
    F_2_best_fit = model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].value

    # Extract the best-fit values of a_1, b_1, and c_1
    a_1_best_fit = model_combined.sources["Backg_DM"].spectrum.main.composite["a_1"].value
    b_1_best_fit = model_combined.sources["Backg_DM"].spectrum.main.composite["b_1"].value
    c_1_best_fit = model_combined.sources["Backg_DM"].spectrum.main.composite["c_1"].value

    # Fix a_1, b_1, and c_1 to their best-fit values
    model_combined.sources["Backg_DM"].spectrum.main.composite["a_1"].value = a_1_best_fit
    model_combined.sources["Backg_DM"].spectrum.main.composite["a_1"].free = False
    model_combined.sources["Backg_DM"].spectrum.main.composite["b_1"].value = b_1_best_fit
    model_combined.sources["Backg_DM"].spectrum.main.composite["b_1"].free = False
    model_combined.sources["Backg_DM"].spectrum.main.composite["c_1"].value = c_1_best_fit
    model_combined.sources["Backg_DM"].spectrum.main.composite["c_1"].free = False

    # Define the range of F_2 values for the scan
    F_2_values = np.linspace(0, 10 * F_2_best_fit, 50)

    # Perform the likelihood scan
    chi_squared_values = []

    for F_2 in F_2_values:
        # Fix F_2 to the current value
        model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].value = F_2
        model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].free = False

        # Recalculate the likelihood
        jl = JointLikelihood(model_combined, DataList(LMCeROSITA))
        result, log_likelihood = jl.fit(quiet=True)

        # Extract the total -log(likelihood) value from the DataFrame
        if isinstance(log_likelihood, pd.DataFrame):
            log_likelihood_total = log_likelihood.loc["total", "-log(likelihood)"]
        else:
            raise ValueError("Unexpected format for log_likelihood. Check its structure.")

        # Compute chi^2 = -2 * log(likelihood)
        chi_squared = 2 * log_likelihood_total
        chi_squared_values.append(chi_squared)

    # Restore F_2 to be free
    model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].free = True

    # Find the minimum chi^2 value
    chi_squared_min = log_likelihood_initial.loc["total", "-log(likelihood)"] * 2

    # Calculate Delta chi^2
    delta_chi_squared = [chi2 - chi_squared_min for chi2 in chi_squared_values]

    # Interpolate the Delta chi^2 function
    interp_func = interp1d(F_2_values, delta_chi_squared, kind="quadratic", fill_value="extrapolate")

    # Solve for F_2 where Delta chi^2 = 2.71
    def delta_chi_squared_eq(F_2):
        return interp_func(F_2) - 2.71

    # Use root_scalar to find the F_2 value
    solution = root_scalar(delta_chi_squared_eq, bracket=[F_2_values[0], F_2_values[-1]], method="brentq")
    F_2_at_2_71 = solution.root if solution.converged else None

    # Save the result for this DM_Line and DM_Sigma
    with open(output_file, "a") as f_out:
        f_out.write(f"{DM_Line}\t{DM_Sigma}\t{F_2_at_2_71}\n")

    # Store results for plotting
    dm_lines.append(DM_Line)
    f2_values.append(F_2_at_2_71)

    print(f"Processed DM_Line = {DM_Line}, DM_Sigma = {DM_Sigma}, F_2_at_2.71 = {F_2_at_2_71}")

print(f"Results saved to {output_file}")

# Plot F_2 as a function of DM_Line
plt.figure(figsize=(10, 6))
plt.plot(dm_lines, f2_values, marker="o", linestyle="-", color="blue")
plt.xlabel("DM Line (keV)", fontsize=14)
plt.ylabel("F_2 at Delta Chi^2 = 2.71", fontsize=14)
plt.title("F_2 as a Function of DM Line", fontsize=16)
plt.grid(True)
plt.show()
