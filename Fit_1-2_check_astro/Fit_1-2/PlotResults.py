import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import chi2
from matplotlib import rc
import matplotlib.patheffects as path_effects

# print an array between 2 and 2.5 with step 0.01


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}' #package mathpazo siunitx
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') #serif
plt.rcParams['axes.linewidth'] = 2

# File path
# csv_file_TM1 = '/home/jortecal/GitHub/eRosita/JuliolTries/MultiResponse/Results/TM1/fit_summary.csv'
#csv_file_TM2 = '/home/jortecal/GitHub/eRosita/JuliolTries/MultiResponse/Results/TM2/fit_summary.csv'
#csv_file_TM3 = '/home/jortecal/GitHub/eRosita/JuliolTries/MultiResponse/Results/TM3/fit_summary.csv'
#csv_file_TM4 = '/home/jortecal/GitHub/eRosita/JuliolTries/MultiResponse/Results/TM4/fit_summary.csv'
#csv_file_TM5 = '/home/jortecal/GitHub/eRosita/JuliolTries/MultiResponse/Results/TM5/fit_summary.csv'
#csv_file_TM6 = '/home/jortecal/GitHub/eRosita/JuliolTries/MultiResponse/Results/TM6/fit_summary.csv'
#csv_file_TM7 = '/home/jortecal/GitHub/eRosita/JuliolTries/MultiResponse/Results/TM7/fit_summary.csv'

path='/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_global/Fit_1-2/output/'
csv_file_Prefit_TM1 = path+'TM1/fit_summary.csv'
csv_file_Prefit_TM2 = path+'TM2/fit_summary.csv'
csv_file_Prefit_TM3 = path+'TM3/fit_summary.csv'
csv_file_Prefit_TM4 = path+'TM4/fit_summary.csv'
csv_file_Prefit_TM5 = path+'TM5/fit_summary.csv'
csv_file_Prefit_TM6 = path+'TM6/fit_summary.csv'
csv_file_Prefit_TM7 = path+'TM7/fit_summary.csv'
Combined_Minimum_TS_File =  path+'CombinedProfiles/Combined_minTS.csv'


# Read CSV
# df_TM1 = pd.read_csv(csv_file_TM1)
#df_TM2 = pd.read_csv(csv_file_TM2)
#df_TM3 = pd.read_csv(csv_file_TM3)
#df_TM4 = pd.read_csv(csv_file_TM4)
#df_TM5 = pd.read_csv(csv_file_TM5)
#df_TM6 = pd.read_csv(csv_file_TM6)
#df_TM7 = pd.read_csv(csv_file_TM7)

df_Prefit_TM1 = pd.read_csv(csv_file_Prefit_TM1)
df_Prefit_TM2 = pd.read_csv(csv_file_Prefit_TM2)
df_Prefit_TM3 = pd.read_csv(csv_file_Prefit_TM3)
df_Prefit_TM4 = pd.read_csv(csv_file_Prefit_TM4)
df_Prefit_TM5 = pd.read_csv(csv_file_Prefit_TM5)
df_Prefit_TM6 = pd.read_csv(csv_file_Prefit_TM6)
df_Prefit_TM7 = pd.read_csv(csv_file_Prefit_TM7)

# Read Combined Minimum TS
df_Combined_Minimum_TS = pd.read_csv(Combined_Minimum_TS_File)
df_Combined_Minimum_TS.rename(columns={'Eline_keV': 'Eline (keV)'}, inplace=True)


# print(df_Combined_Minimum_TS.keys())
# # Set degrees of freedom for TS_astro (change as needed)
# vdof_astro_TM1 = df_TM1['nbins_as'] - df_TM1['nfit_as']
# vdof_dm_TM1 = df_TM1['nbins_dm'] - df_TM1['nfit_dm']

# # Calculate p-value
# df_TM1['p_value_as'] = chi2.sf(df_TM1['TS_as'], vdof_astro_TM1)
# df_TM1['p_value_dm'] = chi2.sf(df_TM1['TS_all'], vdof_dm_TM1)


# Set degrees of freedom for TS_astro (change as needed)
#vdof_astro_TM2 = df_TM2['nbins_as'] - df_TM2['nfit_as']
#vdof_dm_TM2 = df_TM2['nbins_dm'] - df_TM2['nfit_dm']

# Calculate p-value
#df_TM2['p_value_as'] = chi2.sf(df_TM2['TS_as'], vdof_astro_TM2)
#df_TM2['p_value_dm'] = chi2.sf(df_TM2['TS_all'], vdof_dm_TM2)

# Set degrees of freedom for TS_astro (change as needed)
    



#Eliminate points where some TM have no data 
# Get the original sets of energy lines for each TM
orig_elines_TM1 = set(df_Prefit_TM1['Eline (keV)'])
orig_elines_TM2 = set(df_Prefit_TM2['Eline (keV)'])
orig_elines_TM3 = set(df_Prefit_TM3['Eline (keV)'])
orig_elines_TM4 = set(df_Prefit_TM4['Eline (keV)'])
orig_elines_TM5 = set(df_Prefit_TM5['Eline (keV)'])
orig_elines_TM6 = set(df_Prefit_TM6['Eline (keV)'])
orig_elines_TM7 = set(df_Prefit_TM7['Eline (keV)'])

# Union of all lines
all_elines = (
    orig_elines_TM1 | orig_elines_TM2 | orig_elines_TM3 |
    orig_elines_TM4 | orig_elines_TM5 | orig_elines_TM6 | orig_elines_TM7
)


            
common_elines = (
    set(df_Prefit_TM1['Eline (keV)'])
    & set(df_Prefit_TM2['Eline (keV)'])
    & set(df_Prefit_TM3['Eline (keV)'])
    & set(df_Prefit_TM4['Eline (keV)'])
    & set(df_Prefit_TM5['Eline (keV)'])
    & set(df_Prefit_TM6['Eline (keV)'])
    & set(df_Prefit_TM7['Eline (keV)'])
    & set(df_Combined_Minimum_TS['Eline (keV)'])
)

# Filter and sort all DataFrames by common_elines and reset index
def filter_and_sort(df):
    return df[df['Eline (keV)'].isin(common_elines)].sort_values('Eline (keV)').reset_index(drop=True)

df_Prefit_TM1 = filter_and_sort(df_Prefit_TM1)
df_Prefit_TM2 = filter_and_sort(df_Prefit_TM2)
df_Prefit_TM3 = filter_and_sort(df_Prefit_TM3)
df_Prefit_TM4 = filter_and_sort(df_Prefit_TM4)
df_Prefit_TM5 = filter_and_sort(df_Prefit_TM5)
df_Prefit_TM6 = filter_and_sort(df_Prefit_TM6)
df_Prefit_TM7 = filter_and_sort(df_Prefit_TM7)
df_Combined_Minimum_TS = filter_and_sort(df_Combined_Minimum_TS)

# Now build df_Prefit_ALL
df_Prefit_ALL = pd.DataFrame({
    'Eline (keV)': df_Prefit_TM1['Eline (keV)'],
    'TS_as': df_Prefit_TM1['TS_as'] + df_Prefit_TM2['TS_as'] + df_Prefit_TM3['TS_as'] + df_Prefit_TM4['TS_as'] + df_Prefit_TM5['TS_as'] + df_Prefit_TM6['TS_as'] + df_Prefit_TM7['TS_as'],
    'TS_all': df_Prefit_TM1['TS_all'] + df_Prefit_TM2['TS_all'] + df_Prefit_TM3['TS_all'] + df_Prefit_TM4['TS_all'] + df_Prefit_TM5['TS_all'] + df_Prefit_TM6['TS_all'] + df_Prefit_TM7['TS_all'],
    'nbins_as': df_Prefit_TM1['nbins_as'] + df_Prefit_TM2['nbins_as'] + df_Prefit_TM3['nbins_as'] + df_Prefit_TM4['nbins_as'] + df_Prefit_TM5['nbins_as'] + df_Prefit_TM6['nbins_as'] + df_Prefit_TM7['nbins_as'],
    'nfit_as': df_Prefit_TM1['nfit_as'] + df_Prefit_TM2['nfit_as'] + df_Prefit_TM3['nfit_as'] + df_Prefit_TM4['nfit_as'] + df_Prefit_TM5['nfit_as'] + df_Prefit_TM6['nfit_as'] + df_Prefit_TM7['nfit_as']
})

# Check for NaN values
print(df_Prefit_ALL[df_Prefit_ALL['Eline (keV)'].isna()])

# Remove any rows with NaN in 'Eline (keV)'
df_Prefit_ALL = df_Prefit_ALL.dropna(subset=['Eline (keV)'])

# # Calculate p-value
# df_ALL['p_value_as'] = chi2.sf(df_ALL['TS_as'], df_ALL['nbins_as'] - df_ALL['nfit_as'])


# print TS_as for the lines 4.709 and 8.508 for each TM and for the ALL
# for index, row in df_Prefit_TM1.iterrows():
#     if row['Eline (keV)'] in [4.709, 8.508]:
#         print(f"TM1 - Eline (keV): {row['Eline (keV)']}, TS_as: {row['TS_as']}")
# for index, row in df_Prefit_TM2.iterrows():
#     if row['Eline (keV)'] in [4.709, 8.508]:
#         print(f"TM2 - Eline (keV): {row['Eline (keV)']}, TS_as: {row['TS_as']}")
# for index, row in df_Prefit_TM3.iterrows(): 
#     if row['Eline (keV)'] in [4.709, 8.508]:
#         print(f"TM3 - Eline (keV): {row['Eline (keV)']}, TS_as: {row['TS_as']}")    
# for index, row in df_Prefit_TM4.iterrows():
#     if row['Eline (keV)'] in [4.709, 8.508]:
#         print(f"TM4 - Eline (keV): {row['Eline (keV)']}, TS_as: {row['TS_as']}")
# for index, row in df_Prefit_TM5.iterrows():
#     if row['Eline (keV)'] in [4.709, 8.508]:
#         print(f"TM5 - Eline (keV): {row['Eline (keV)']}, TS_as: {row['TS_as']}")
# for index, row in df_Prefit_TM6.iterrows():
#     if row['Eline (keV)'] in [4.709, 8.508]:
#         print(f"TM6 - Eline (keV): {row['Eline (keV)']}, TS_as: {row['TS_as']}")
# for index, row in df_Prefit_TM7.iterrows():
#     if row['Eline (keV)'] in [4.709, 8.508]:
#         print(f"TM7 - Eline (keV): {row['Eline (keV)']}, TS_as: {row['TS_as']}")
# for index, row in df_Prefit_ALL.iterrows():
#     if row['Eline (keV)'] in [4.709, 8.508]:
#         print(f"ALL - Eline (keV): {row['Eline (keV)']}, TS_as: {row['TS_as']}")
# exit()

# Calculate p-value
df_Prefit_ALL['p_value_as'] = chi2.sf(df_Combined_Minimum_TS['TS_at0'], df_Prefit_ALL['nbins_as'] - df_Prefit_ALL['nfit_as'])



# print the p-values that are below 0.05 and the corresponding Eline (keV)

# print("TM1 p-values below 0.05:")
# for index, row in df_Prefit_TM1.iterrows():
#     if row['p_value_as'] < 0.05:
#         print(f"Eline (keV): {row['Eline (keV)']}, p-value: {row['p_value_as']}")

# print("TM2 p-values below 0.05:")
# for index, row in df_Prefit_TM2.iterrows():
#     if row['p_value_as'] < 0.05:
#         print(f"Eline (keV): {row['Eline (keV)']}, p-value: {row['p_value_as']}")

# print("TM3 p-values below 0.05:")
# for index, row in df_Prefit_TM3.iterrows():
#     if row['p_value_as'] < 0.05:
#         print(f"Eline (keV): {row['Eline (keV)']}, p-value: {row['p_value_as']}")

# print("TM4 p-values below 0.05:")
# for index, row in df_Prefit_TM4.iterrows():
#     if row['p_value_as'] < 0.05:
#         print(f"Eline (keV): {row['Eline (keV)']}, p-value: {row['p_value_as']}")   

# print("TM5 p-values below 0.05:")
# for index, row in df_Prefit_TM5.iterrows():
#     if row['p_value_as'] < 0.05:
#         print(f"Eline (keV): {row['Eline (keV)']}, p-value: {row['p_value_as']}")   

# print("TM6 p-values below 0.05:")
# for index, row in df_Prefit_TM6.iterrows():
#     if row['p_value_as'] < 0.05:
#         print(f"Eline (keV): {row['Eline (keV)']}, p-value: {row['p_value_as']}")   

# print("TM7 p-values below 0.05:")
# for index, row in df_Prefit_TM7.iterrows():
#     if row['p_value_as'] < 0.05:
#         print(f"Eline (keV): {row['Eline (keV)']}, p-value: {row['p_value_as']}")   

# # Plot the data p-value

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()

# # plt.plot(df_TM1['Eline (keV)'], df_TM1['p_value_as'], label='TM1', marker='o', linestyle='-', markersize=2)
# # plt.plot(df_Prefit_TM1['Eline (keV)'], df_Prefit_TM1['p_value_as'], label='TM1 Pre-fit', marker='o', linestyle='-', markersize=2)
# # plt.plot(df_TM2['Eline (keV)'], df_TM2['p_value_as'], label='TM2', marker='o',  markersize=2)
# # plt.plot(df_Prefit_TM2['Eline (keV)'], df_Prefit_TM2['p_value_as'], label='TM2 Pre-fit', marker='o', linestyle='--', markersize=2)
# # plt.plot(df_TM3['Eline (keV)'], df_TM3['p_value_as'], label='TM3', marker='o',  markersize=2)
# # plt.plot(df_Prefit_TM3['Eline (keV)'], df_Prefit_TM3['p_value_as'], label='TM3 Pre-fit', marker='o', linestyle='--', markersize=2)
# # plt.plot(df_TM4['Eline (keV)'], df_TM4['p_value_as'], label='TM4', marker='s', markersize=2)
# # plt.plot(df_Prefit_TM4['Eline (keV)'], df_Prefit_TM4['p_value_as'], label='TM4 Pre-fit', marker='o', linestyle='--', markersize=2)
# # plt.plot(df_TM5['Eline (keV)'], df_TM5['p_value_as'], label='TM5', marker='1', linestyle='-', markersize=2)
# # plt.plot(df_Prefit_TM5['Eline (keV)'], df_Prefit_TM5['p_value_as'], label='TM5 Pre-fit', marker='o', linestyle='--', markersize=2)
# # plt.plot(df_TM6['Eline (keV)'], df_TM6['p_value_as'], label='TM6', marker='x', linestyle='--', markersize=2)
# # plt.plot(df_Prefit_TM6['Eline (keV)'], df_Prefit_TM6['p_value_as'], label='TM6 Pre-fit', marker='o', linestyle='--', markersize=2)
# # plt.plot(df_TM7['Eline (keV)'], df_TM7['p_value_as'], label='TM7', marker='D', markersize=2)
# # plt.plot(df_Prefit_TM7['Eline (keV)'], df_Prefit_TM7['p_value_as'], label='TM7 Pre-fit', marker='o', linestyle='--', markersize=2)
plt.plot(df_Prefit_ALL['Eline (keV)'], df_Prefit_ALL['p_value_as'], marker='o', linestyle='-', markersize=2)


print(" p-values ", df_Prefit_ALL['p_value_as'])
print(" TS ", df_Combined_Minimum_TS['TS_at0'])
print(" dof ", df_Prefit_ALL['nbins_as'] - df_Prefit_ALL['nfit_as'])

print(" energies ", df_Prefit_ALL['Eline (keV)'])

#horizontal line at p_value = 0.05
plt.axhline(y=0.05, color='black', linestyle='--', label='p-value = 0.05')



plt.xlabel(r'E (keV)',fontsize=30)
plt.ylabel(r'p-value',fontsize=30)
plt.xlim(1.0, 2.3)
plt.ylim(1e-3, 1)
plt.xscale('log')
plt.yscale('log')


ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in', labelsize=22,width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('p-value2_pm20.pdf')

# Plot the data DeltaTS

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()

# plt.plot(df_TM1['Eline (keV)'], df_TM1['p_value_as'], label='TM1', marker='o', linestyle='-', markersize=2)
# plt.plot(df_Prefit_TM1['Eline (keV)'], df_Prefit_TM1['p_value_as'], label='TM1 Pre-fit', marker='o', linestyle='-', markersize=2)
# plt.plot(df_TM2['Eline (keV)'], df_TM2['p_value_as'], label='TM2', marker='o',  markersize=2)
# plt.plot(df_Prefit_TM2['Eline (keV)'], df_Prefit_TM2['p_value_as'], label='TM2 Pre-fit', marker='o', linestyle='--', markersize=2)
# plt.plot(df_TM3['Eline (keV)'], df_TM3['p_value_as'], label='TM3', marker='o',  markersize=2)
# plt.plot(df_Prefit_TM3['Eline (keV)'], df_Prefit_TM3['p_value_as'], label='TM3 Pre-fit', marker='o', linestyle='--', markersize=2)
# plt.plot(df_TM4['Eline (keV)'], df_TM4['p_value_as'], label='TM4', marker='s', markersize=2)
# plt.plot(df_Prefit_TM4['Eline (keV)'], df_Prefit_TM4['p_value_as'], label='TM4 Pre-fit', marker='o', linestyle='--', markersize=2)
# plt.plot(df_TM5['Eline (keV)'], df_TM5['p_value_as'], label='TM5', marker='1', linestyle='-', markersize=2)
# plt.plot(df_Prefit_TM5['Eline (keV)'], df_Prefit_TM5['p_value_as'], label='TM5 Pre-fit', marker='o', linestyle='--', markersize=2)
# plt.plot(df_TM6['Eline (keV)'], df_TM6['p_value_as'], label='TM6', marker='x', linestyle='--', markersize=2)
# plt.plot(df_Prefit_TM6['Eline (keV)'], df_Prefit_TM6['p_value_as'], label='TM6 Pre-fit', marker='o', linestyle='--', markersize=2)
# plt.plot(df_TM7['Eline (keV)'], df_TM7['p_value_as'], label='TM7', marker='D', markersize=2)
# plt.plot(df_Prefit_TM7['Eline (keV)'], df_Prefit_TM7['p_value_as'], label='TM7 Pre-fit', marker='o', linestyle='--', markersize=2)
# plt.plot(df_Prefit_ALL['Eline (keV)'], df_Prefit_ALL['TS_as']-df_Prefit_ALL['TS_all'], label='ALL TMs', marker='o', linestyle='--', markersize=2)
plt.plot(df_Combined_Minimum_TS['Eline (keV)'], df_Combined_Minimum_TS['TS_at0']-df_Combined_Minimum_TS['min_TS'], marker='o', linestyle='-', markersize=2)
plt.axhline(y=9, color='red', linestyle=':', label='3$\sigma$')
plt.axhline(y=25, color='red', linestyle='--', label='5$\sigma$')



print("E ",df_Combined_Minimum_TS['Eline (keV)'])
print(" DeltaTS ", df_Combined_Minimum_TS['TS_at0']-df_Combined_Minimum_TS['min_TS'])



plt.xlabel(r'E (keV)',fontsize=30)
plt.ylabel(r'$\Delta$TS',fontsize=30)
# plt.ylim(1e-3, 1)
plt.xscale('log')
# plt.yscale('log')

ax1.tick_params(which='major',direction='in',width=1, labelsize=22,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=15)
plt.xlim(1.0, 2.3)
plt.ylim(-1, 40)
plt.tight_layout()
# plt.show()
plt.savefig('DeltaTS2_pm20.pdf')

# Print all the energy lines with a Delta_TS = df_Prefit_ALL['TS_as']-df_Combined_Minimum_TS['min_TS'] larger than 9 
print(df_Combined_Minimum_TS[df_Combined_Minimum_TS['TS_at0']-df_Combined_Minimum_TS['min_TS'] > 9]['Eline (keV)'])


