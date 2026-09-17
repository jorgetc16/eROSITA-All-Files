import warnings

warnings.simplefilter("ignore")
import numpy as np
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


DM_Line = 4.1
DM_Sigma = 112e-3/(2*np.sqrt(2*np.log(2)))

Bckg_DM = PointSource("Backg_DM", 0, 0, spectral_shape=Background_and_DM)

# Create the final model
model_combined = Model(Bckg_DM)


model_combined.sources["Backg_DM"].spectrum.main.composite["mu_2"].value = DM_Line
model_combined.sources["Backg_DM"].spectrum.main.composite["mu_2"].free = False

model_combined.sources["Backg_DM"].spectrum.main.composite["sigma_2"].value = DM_Sigma
model_combined.sources["Backg_DM"].spectrum.main.composite["sigma_2"].free = False

model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].free = True
model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].bounds = (0, 10)

# Create the data 
LMCeROSITA= OGIPLike("LMCeROSITA", observation=LMC3deg_pha, response=LMC3deg_rmf,  arf_file=LMC3deg_arf)



Range_Min, Range_Max = DM_Line - 5*DM_Sigma, DM_Line + 5*DM_Sigma
LMCeROSITA.set_active_measurements(f"{Range_Min}-{Range_Max}")

# Perform an initial fit with F_2 free
jl_initial = JointLikelihood(model_combined, DataList(LMCeROSITA))
result_initial, log_likelihood_initial = jl_initial.fit()

# Extract the best-fit value of F_2
F_2_best_fit = model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].value
print(f"Best-fit F_2 value: {F_2_best_fit}")


# Define the range of F_2 values for the scan
F_2_values = np.linspace(0, 100 * F_2_best_fit, 50)  # Adjust the number of points as needed
print(f"F_2 values for scan: {F_2_values}")
# Perform the likelihood scan
chi_squared_values = []

for F_2 in F_2_values:
    # Fix F_2 to the current value
    model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].value = F_2
    model_combined.sources["Backg_DM"].spectrum.main.composite["F_2"].free = False  # Fix the parameter

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

# # Use root_scalar to find the F_2 value
# solution = root_scalar(delta_chi_squared_eq, bracket=[F_2_values[0], F_2_values[-1]], method="brentq")
# F_2_at_2_71 = solution.root if solution.converged else None

# # Save the results to a file
# output_file = "/home/jortecal/GitHub/eRosita/3MLFits/chi_squared_scan.txt"
# with open(output_file, "w") as f:
#     f.write("F_2_value\tchi_squared\tDelta_chi_squared\n")
#     for F_2, chi2, delta_chi2 in zip(F_2_values, chi_squared_values, delta_chi_squared):
#         f.write(f"{F_2}\t{chi2}\t{delta_chi2}\n")

# print(f"Chi-squared scan results saved to {output_file}")
# print(f"F_2 value where Delta chi^2 = 2.71: {F_2_at_2_71}")

# Plot the chi-squared as a function of F_2
plt.figure(figsize=(10, 6))
plt.plot(F_2_values, delta_chi_squared, marker="o", linestyle="-", color="blue", label="Delta chi^2")
plt.axhline(2.71, color="red", linestyle="--", label="Delta chi^2 = 2.71")
# if F_2_at_2_71 is not None:
#     plt.axvline(F_2_at_2_71, color="green", linestyle="--", label=f"F_2 at Delta chi^2 = 2.71: {F_2_at_2_71:.4f}")
plt.xlabel("F_2 Value", fontsize=14)
plt.ylabel("Delta chi^2", fontsize=14)
plt.title("Delta Chi^2 as a Function of F_2", fontsize=16)
plt.legend()
plt.grid(True)
plt.show()


