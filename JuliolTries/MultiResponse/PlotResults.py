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
csv_file_Prefit_TM1 = '/home/jortecal/GitHub/eRosita/Full_Range_24_dblpwl/Results/TM1_summary.csv'
csv_file_Prefit_TM2 = '/home/jortecal/GitHub/eRosita/Full_Range_24_dblpwl/Results/TM2_summary.csv'
csv_file_Prefit_TM3 = '/home/jortecal/GitHub/eRosita/Full_Range_24_dblpwl/Results/TM3_summary.csv'
csv_file_Prefit_TM4 = '/home/jortecal/GitHub/eRosita/Full_Range_24_dblpwl/Results/TM4_summary.csv'
csv_file_Prefit_TM5 = '/home/jortecal/GitHub/eRosita/Full_Range_24_dblpwl/Results/TM5_summary.csv'
csv_file_Prefit_TM6 = '/home/jortecal/GitHub/eRosita/Full_Range_24_dblpwl/Results/TM6_summary.csv'
csv_file_Prefit_TM7 = '/home/jortecal/GitHub/eRosita/Full_Range_24_dblpwl/Results/TM7_summary.csv'

Combined_Minimum_TS_File =  '/home/jortecal/GitHub/eRosita/Full_Range_24_dblpwl/Results/CombinedProfiles/Combined_minTS.csv'


csv_file_TM1 = '/home/jortecal/GitHub/eRosita/2to9FixedLines/Results/TM1_summary.csv'
csv_file_TM2 = '/home/jortecal/GitHub/eRosita/2to9FixedLines/Results/TM2_summary.csv'
csv_file_TM3 = '/home/jortecal/GitHub/eRosita/2to9FixedLines/Results/TM3_summary.csv'
csv_file_TM4 = '/home/jortecal/GitHub/eRosita/2to9FixedLines/Results/TM4_summary.csv'
csv_file_TM5 = '/home/jortecal/GitHub/eRosita/2to9FixedLines/Results/TM5_summary.csv'
csv_file_TM6 = '/home/jortecal/GitHub/eRosita/2to9FixedLines/Results/TM6_summary.csv'
csv_file_TM7 = '/home/jortecal/GitHub/eRosita/2to9FixedLines/Results/TM7_summary.csv'

Combined_Minimum_TS_File_1 =  '/home/jortecal/GitHub/eRosita/2to9FixedLines/Results/CombinedProfiles/Combined_minTS.csv'

# Read CSV
df_TM1 = pd.read_csv(csv_file_TM1)
df_TM2 = pd.read_csv(csv_file_TM2)
df_TM3 = pd.read_csv(csv_file_TM3)
df_TM4 = pd.read_csv(csv_file_TM4)
df_TM5 = pd.read_csv(csv_file_TM5)
df_TM6 = pd.read_csv(csv_file_TM6)
df_TM7 = pd.read_csv(csv_file_TM7)

df_Prefit_TM1 = pd.read_csv(csv_file_Prefit_TM1)
df_Prefit_TM2 = pd.read_csv(csv_file_Prefit_TM2)
df_Prefit_TM3 = pd.read_csv(csv_file_Prefit_TM3)
df_Prefit_TM4 = pd.read_csv(csv_file_Prefit_TM4)
df_Prefit_TM5 = pd.read_csv(csv_file_Prefit_TM5)
df_Prefit_TM6 = pd.read_csv(csv_file_Prefit_TM6)
df_Prefit_TM7 = pd.read_csv(csv_file_Prefit_TM7)

# Read Combined Minimum TS
df_Combined_Minimum_TS_1 = pd.read_csv(Combined_Minimum_TS_File_1)
df_Combined_Minimum_TS_1.rename(columns={'Eline_keV': 'Eline (keV)'}, inplace=True)


# Read Combined Minimum TS
df_Combined_Minimum_TS = pd.read_csv(Combined_Minimum_TS_File)
df_Combined_Minimum_TS.rename(columns={'Eline_keV': 'Eline (keV)'}, inplace=True)


# print(df_Combined_Minimum_TS.keys())
# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_TM1 = df_TM1['nbins_as'] - df_TM1['nfit_as']
vdof_dm_TM1 = df_TM1['nbins_dm'] - df_TM1['nfit_dm']

# Calculate p-value
df_TM1['p_value_as'] = chi2.sf(df_TM1['TS_as'], vdof_astro_TM1)
df_TM1['p_value_dm'] = chi2.sf(df_TM1['TS_all'], vdof_dm_TM1)


# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_TM2 = df_TM2['nbins_as'] - df_TM2['nfit_as']
vdof_dm_TM2 = df_TM2['nbins_dm'] - df_TM2['nfit_dm']

# Calculate p-value
df_TM2['p_value_as'] = chi2.sf(df_TM2['TS_as'], vdof_astro_TM2)
df_TM2['p_value_dm'] = chi2.sf(df_TM2['TS_all'], vdof_dm_TM2)

# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_TM3 = df_TM3['nbins_as'] - df_TM3['nfit_as']
vdof_dm_TM3 = df_TM3['nbins_dm'] - df_TM3['nfit_dm']    

# Calculate p-value
df_TM3['p_value_as'] = chi2.sf(df_TM3['TS_as'], vdof_astro_TM3)
df_TM3['p_value_dm'] = chi2.sf(df_TM3['TS_all'], vdof_dm_TM3)   

# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_TM4 = df_TM4['nbins_as'] - df_TM4['nfit_as']
vdof_dm_TM4 = df_TM4['nbins_dm'] - df_TM4['nfit_dm']

# Calculate p-value
df_TM4['p_value_as'] = chi2.sf(df_TM4['TS_as'], vdof_astro_TM4)
df_TM4['p_value_dm'] = chi2.sf(df_TM4['TS_all'], vdof_dm_TM4)

# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_TM5 = df_TM5['nbins_as'] - df_TM5['nfit_as']
vdof_dm_TM5 = df_TM5['nbins_dm'] - df_TM5['nfit_dm']   
# Calculate p-value
df_TM5['p_value_as'] = chi2.sf(df_TM5['TS_as'], vdof_astro_TM5)
df_TM5['p_value_dm'] = chi2.sf(df_TM5['TS_all'], vdof_dm_TM5)     

# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_TM6 = df_TM6['nbins_as'] - df_TM6['nfit_as']
vdof_dm_TM6 = df_TM6['nbins_dm'] - df_TM6['nfit_dm']    

# Calculate p-value
df_TM6['p_value_as'] = chi2.sf(df_TM6['TS_as'], vdof_astro_TM6)
df_TM6['p_value_dm'] = chi2.sf(df_TM6['TS_all'], vdof_dm_TM6)

# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_TM7 = df_TM7['nbins_as'] - df_TM7['nfit_as']
vdof_dm_TM7 = df_TM7['nbins_dm'] - df_TM7['nfit_dm']    

# Calculate p-value
df_TM7['p_value_as'] = chi2.sf(df_TM7['TS_as'], vdof_astro_TM7)
df_TM7['p_value_dm'] = chi2.sf(df_TM7['TS_all'], vdof_dm_TM7)       



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
common_elines_1 = (
    set(df_TM1['Eline (keV)'])
    & set(df_TM2['Eline (keV)'])
    & set(df_TM3['Eline (keV)'])
    & set(df_TM4['Eline (keV)'])
    & set(df_TM5['Eline (keV)'])
    & set(df_TM6['Eline (keV)'])
    & set(df_TM7['Eline (keV)'])
    & set(df_Combined_Minimum_TS_1['Eline (keV)'])
)
# Filter and sort all DataFrames by common_elines and reset index
def filter_and_sort(df, common_elines=common_elines):
    return df[df['Eline (keV)'].isin(common_elines)].sort_values('Eline (keV)').reset_index(drop=True)

df_Prefit_TM1 = filter_and_sort(df_Prefit_TM1, common_elines)
df_Prefit_TM2 = filter_and_sort(df_Prefit_TM2, common_elines)
df_Prefit_TM3 = filter_and_sort(df_Prefit_TM3, common_elines)
df_Prefit_TM4 = filter_and_sort(df_Prefit_TM4, common_elines)
df_Prefit_TM5 = filter_and_sort(df_Prefit_TM5, common_elines)
df_Prefit_TM6 = filter_and_sort(df_Prefit_TM6, common_elines)
df_Prefit_TM7 = filter_and_sort(df_Prefit_TM7, common_elines)
df_Combined_Minimum_TS = filter_and_sort(df_Combined_Minimum_TS, common_elines)

# Filter TM dataframes by common_elines BEFORE creating df_ALL
df_TM1 = filter_and_sort(df_TM1, common_elines_1)
df_TM2 = filter_and_sort(df_TM2, common_elines_1)
df_TM3 = filter_and_sort(df_TM3, common_elines_1)
df_TM4 = filter_and_sort(df_TM4, common_elines_1)
df_TM5 = filter_and_sort(df_TM5, common_elines_1)
df_TM6 = filter_and_sort(df_TM6, common_elines_1)
df_TM7 = filter_and_sort(df_TM7, common_elines_1)
df_Combined_Minimum_TS_1 = filter_and_sort(df_Combined_Minimum_TS_1, common_elines_1)

# Filter out energies below 2.21 keV for the 2-9 keV case
min_energy = 2.28
df_TM1 = df_TM1[df_TM1['Eline (keV)'] >= min_energy]
df_TM2 = df_TM2[df_TM2['Eline (keV)'] >= min_energy]
df_TM3 = df_TM3[df_TM3['Eline (keV)'] >= min_energy]
df_TM4 = df_TM4[df_TM4['Eline (keV)'] >= min_energy]
df_TM5 = df_TM5[df_TM5['Eline (keV)'] >= min_energy]
df_TM6 = df_TM6[df_TM6['Eline (keV)'] >= min_energy]
df_TM7 = df_TM7[df_TM7['Eline (keV)'] >= min_energy]
df_Combined_Minimum_TS_1 = df_Combined_Minimum_TS_1[df_Combined_Minimum_TS_1['Eline (keV)'] >= min_energy]


df_Prefit_TM1 = df_Prefit_TM1[df_Prefit_TM1['Eline (keV)'] <= min_energy]
df_Prefit_TM2 = df_Prefit_TM2[df_Prefit_TM2['Eline (keV)'] <= min_energy]
df_Prefit_TM3 = df_Prefit_TM3[df_Prefit_TM3['Eline (keV)'] <= min_energy]
df_Prefit_TM4 = df_Prefit_TM4[df_Prefit_TM4['Eline (keV)'] <= min_energy]
df_Prefit_TM5 = df_Prefit_TM5[df_Prefit_TM5['Eline (keV)'] <= min_energy]
df_Prefit_TM6 = df_Prefit_TM6[df_Prefit_TM6['Eline (keV)'] <= min_energy]
df_Prefit_TM7 = df_Prefit_TM7[df_Prefit_TM7['Eline (keV)'] <= min_energy]
df_Combined_Minimum_TS = df_Combined_Minimum_TS[df_Combined_Minimum_TS['Eline (keV)'] <= min_energy]
# Now build df_ALL with filtered data
df_ALL = pd.DataFrame({
    'Eline (keV)': df_TM1['Eline (keV)'],
    'TS_as': df_TM1['TS_as'] + df_TM2['TS_as'] + df_TM3['TS_as'] + df_TM4['TS_as'] + df_TM5['TS_as'] + df_TM6['TS_as'] + df_TM7['TS_as'],
    'TS_all': df_TM1['TS_all'] + df_TM2['TS_all'] + df_TM3['TS_all'] + df_TM4['TS_all'] + df_TM5['TS_all'] + df_TM6['TS_all'] + df_TM7['TS_all'],
    'nbins_as': df_TM1['nbins_as'] + df_TM2['nbins_as'] + df_TM3['nbins_as'] + df_TM4['nbins_as'] + df_TM5['nbins_as'] + df_TM6['nbins_as'] + df_TM7['nbins_as'],
    'nfit_as': df_TM1['nfit_as'] + df_TM2['nfit_as'] + df_TM3['nfit_as'] + df_TM4['nfit_as'] + df_TM5['nfit_as'] + df_TM6['nfit_as'] + df_TM7['nfit_as']
})


# Now build df_Prefit_ALL
df_Prefit_ALL = pd.DataFrame({
    'Eline (keV)': df_Prefit_TM1['Eline (keV)'],
    'TS_as': df_Prefit_TM1['TS_as'] + df_Prefit_TM2['TS_as'] + df_Prefit_TM3['TS_as'] + df_Prefit_TM4['TS_as'] + df_Prefit_TM5['TS_as'] + df_Prefit_TM6['TS_as'] + df_Prefit_TM7['TS_as'],
    'TS_all': df_Prefit_TM1['TS_all'] + df_Prefit_TM2['TS_all'] + df_Prefit_TM3['TS_all'] + df_Prefit_TM4['TS_all'] + df_Prefit_TM5['TS_all'] + df_Prefit_TM6['TS_all'] + df_Prefit_TM7['TS_all'],
    'nbins_as': df_Prefit_TM1['nbins_as'] + df_Prefit_TM2['nbins_as'] + df_Prefit_TM3['nbins_as'] + df_Prefit_TM4['nbins_as'] + df_Prefit_TM5['nbins_as'] + df_Prefit_TM6['nbins_as'] + df_Prefit_TM7['nbins_as'],
    'nfit_as': df_Prefit_TM1['nfit_as'] + df_Prefit_TM2['nfit_as'] + df_Prefit_TM3['nfit_as'] + df_Prefit_TM4['nfit_as'] + df_Prefit_TM5['nfit_as'] + df_Prefit_TM6['nfit_as'] + df_Prefit_TM7['nfit_as']
})

# Check for NaN values
# print(df_Prefit_ALL[df_Prefit_ALL['Eline (keV)'].isna()])

# Remove any rows with NaN in 'Eline (keV)'
df_Prefit_ALL = df_Prefit_ALL.dropna(subset=['Eline (keV)'])


# Calculate p-value
df_Prefit_ALL['p_value_as'] = chi2.sf(df_Combined_Minimum_TS['TS_at0'], df_Prefit_ALL['nbins_as'] - df_Prefit_ALL['nfit_as'])

df_ALL['p_value_as'] = chi2.sf(df_Combined_Minimum_TS_1['TS_at0'], df_ALL['nbins_as'] - df_ALL['nfit_as'])


min_pval = 0.05
df_Combined_Minimum_TS = df_Combined_Minimum_TS[df_Prefit_ALL['p_value_as'] >= min_pval]
df_Prefit_ALL = df_Prefit_ALL[df_Prefit_ALL['p_value_as'] >= min_pval]


# # Plot the data p-value

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()


# plt.plot(df_Prefit_ALL['Eline (keV)'], df_Prefit_ALL['p_value_as'], marker='', linestyle='-.', markersize=4, label=r'Sliding Window')
# plt.plot(df_ALL['Eline (keV)'], df_ALL['p_value_as'], marker='', linestyle='--', markersize=5, label='Full Range')
plt.plot(np.concatenate((df_Prefit_ALL['Eline (keV)'],df_ALL['Eline (keV)'])), np.concatenate((df_Prefit_ALL['p_value_as'],df_ALL['p_value_as'])), marker='', linestyle='-', markersize=3, color='dodgerblue')


# print("P-value at 1.080", df_Prefit_ALL[df_Prefit_ALL['Eline (keV)'] == 1.080]['p_value_as'].values[0])º
#horizontal line at p_value = 0.05
# plt.axhline(y=0.05, color='black', linestyle='--', label='p-value = 0.05')

# print the points where p-value is below 0.05
for index, row in df_ALL.iterrows():
    if row['p_value_as'] < 0.05:
        print(f"Eline (keV): {row['Eline (keV)']}, p-value: {row['p_value_as']}")

plt.xlabel(r'E (keV)',fontsize=30)
plt.ylabel(r'p-value',fontsize=30)
plt.ylim(1e-3, 1)
plt.xscale('log')
plt.yscale('log')


ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in', labelsize=22,width=1,length=7,top=True,right=True,pad=10)
ax1.set_xticks([1, 2, 4, 6, 9])
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
plt.xlim(1, 9)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
# plt.legend(fontsize=15)
plt.tight_layout()

plt.savefig('p-value.pdf')
# Plot the data DeltaTS

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()

# plt.plot(df_Combined_Minimum_TS['Eline (keV)'], np.sqrt(df_Combined_Minimum_TS['TS_at0']-df_Combined_Minimum_TS['min_TS']), marker='', linestyle='-.', markersize=4, label=r'Sliding Window')
# plt.plot(df_Combined_Minimum_TS_1['Eline (keV)'], np.sqrt(df_Combined_Minimum_TS_1['TS_at0']-df_Combined_Minimum_TS_1['min_TS']), marker='', linestyle='--', markersize=5,label='Full Range')

# Load IB line energies
ib_lines = np.loadtxt('/home/jortecal/GitHub/eRosita/IBLines/IBLine_TM3.txt', usecols=0)

# Plot the data DeltaTS

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()

plt.plot(np.concatenate((df_Combined_Minimum_TS['Eline (keV)'],df_Combined_Minimum_TS_1['Eline (keV)'])), np.concatenate((np.sqrt(df_Combined_Minimum_TS['TS_at0']-df_Combined_Minimum_TS['min_TS']),np.sqrt(df_Combined_Minimum_TS_1['TS_at0']-df_Combined_Minimum_TS_1['min_TS']))), color='dodgerblue', marker='', markersize=3)

for e in ib_lines:
    plt.axvline(x=e, color='gray', linewidth=2, alpha=0.8, zorder=0, linestyle='--', label='IB Lines' if e == ib_lines[0] else "")  # Add label only for the first line

# plt.axhline(y=3, color='red', linestyle='-.', label=r'3$\sigma$')
# plt.axhline(y=5, color='red', linestyle='--', label=r'5$\sigma$')
# plt.axhline(y=3, color='red', linestyle='-.', label=r'3$\sigma$')
# plt.axhline(y=5, color='red', linestyle='--', label=r'5$\sigma$')







plt.xlabel(r'Energy [keV]',fontsize=30)
plt.ylabel(r'$\sqrt{ \mathrm{TS}}$',fontsize=30)
# plt.ylim(1e-3, 1)
plt.xscale('log')
# plt.yscale('log')

ax1.tick_params(which='major',direction='in',width=1, labelsize=22,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
ax1.set_xticks([1, 2, 4, 6, 9])
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=15)
plt.xlim(1, 9)
plt.ylim(-.1, 7)
plt.tight_layout()
# plt.show()
plt.savefig('DeltaTS.pdf')

df_boundsNew = np.loadtxt('/home/jortecal/GitHub/eRosita/FINALRESULTS/2to9FixedLines/Results/CombinedProfiles/CombinedBounds.dat')
df_boundsOld = np.loadtxt('/home/jortecal/GitHub/eRosita/Full_Range_24_dblpwl/Results/CombinedProfiles/CombinedBounds.dat')

# Filter: keep only energies >= 2.21 keV for the "New" bounds
_min_E_new = 2.21
df_boundsNew = df_boundsNew[df_boundsNew[:, 0] >= _min_E_new]
df_boundsOld = df_boundsOld[df_boundsOld[:, 0] <= _min_E_new]
min_energy = 1.15
df_boundsOld = df_boundsOld[df_boundsOld[:, 0] >= min_energy]

bounds_together = np.concatenate((df_boundsOld,df_boundsNew), axis=0)

# Plot the data bounds
fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()
# plt.plot(df_boundsNew[:,0], df_boundsNew[:,1], label=r'Sliding Window', marker='', linestyle='-.', markersize=4)
# plt.plot(df_boundsOld[:,0], df_boundsOld[:,1], label=r'Full Range', marker='', linestyle='--', markersize=4)
plt.plot(bounds_together[:,0], bounds_together[:,1], color='dodgerblue', marker='', markersize=3, linestyle='-')
plt.xlabel(r'E (keV)',fontsize=30)
plt.ylabel(r'Upper Bound (ph cm$^{-2}$ s$^{-1}$)',fontsize=30)
plt.xscale('log')
plt.yscale('log')
ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in', labelsize=22,width=1,length=7,top=True,right=True,pad=10)
ax1.set_xticks([1, 2, 4, 6, 9])
ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
plt.xlim(1, 9)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=15)
plt.tight_layout()  
# plt.show()
# plt.savefig('UpperBounds_1to2_SlidingWindow_Comparison.pdf')