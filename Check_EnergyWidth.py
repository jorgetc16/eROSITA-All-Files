import numpy as np
import matplotlib.pyplot as plt

# Set plot style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

# Load data files
# File 1: CSV with space delimiter
data1 = np.loadtxt('/home/jortecal/GitHub/eRosita/Energy_width/Energ_Width_TM7.csv')
sorted_data1 = data1[np.argsort(data1[:, 0])]
energy1 = sorted_data1[:, 0]
width1 = sorted_data1[:, 1]

# File 2: TXT with comma delimiter
data2 = np.loadtxt('/home/jortecal/GitHub/eRosita/3MLFits/Energy_width.txt', delimiter=',')
energy2 = data2[:, 0]
width2 = data2[:, 1]
# print energy width for 1.444 keV interpolation
from scipy.interpolate import interp1d
interp_func = interp1d(energy2, width2, kind='linear', fill_value="extrapolate")
width_at_1_444 = interp_func(1.444)
print(f"Interpolated width at 1.444 keV: {width_at_1_444:.2f} eV")


interp_func = interp1d(energy1, width1, kind='linear', fill_value="extrapolate")
width_at_1_444 = interp_func(1.444)
print(f"Interpolated width at 1.444 keV: {width_at_1_444:.2f} eV")


# # Create plot
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot both datasets
# ax.plot(energy1, width1, 'o-', label='Energ_Width_TM1.csv', 
#         markersize=6, linewidth=1.5, color='blue')
# ax.plot(energy2, width2, 's-', label='Energy_width.txt (3MLFits)', 
#         markersize=5, linewidth=1.5, color='red', alpha=0.7)

# # Labels and formatting
# ax.set_xlabel('Energy [keV]', fontsize=14)
# ax.set_ylabel('Width [eV]', fontsize=14)
# ax.set_title('Energy Resolution Comparison', fontsize=16, weight='bold')
# ax.legend(fontsize=12, loc='best')
# ax.grid(True, alpha=0.3, linestyle='--')

# # Make it look nice
# ax.tick_params(which='both', direction='in', top=True, right=True)
# plt.tight_layout()

# # Save and show
# plt.savefig('energy_width_comparison.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('energy_width_comparison.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: energy_width_comparison.pdf")
# print("✓ Saved: energy_width_comparison.png")

# plt.show()

# # Print statistics
# print("\n" + "="*60)
# print("STATISTICS:")
# print("="*60)
# print(f"\nEnerg_Width_TM1.csv:")
# print(f"  Energy range: {energy1.min():.3f} - {energy1.max():.3f} keV")
# print(f"  Width range: {width1.min():.1f} - {width1.max():.1f} eV")
# print(f"  Number of points: {len(energy1)}")

# print(f"\nEnergy_width.txt (3MLFits):")
# print(f"  Energy range: {energy2.min():.3f} - {energy2.max():.3f} keV")
# print(f"  Width range: {width2.min():.1f} - {width2.max():.1f} eV")
# print(f"  Number of points: {len(energy2)}")
# print("="*60)