import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib import rc
import matplotlib.patheffects as path_effects

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('font', weight='bold')

# File path
csv_file_XSpec_M2 = '/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/XSpec/Model2/fit_summary.csv'
csv_file_Minuit_M2 = '/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/Minuit/Model2/fit_summary.csv'
csv_file_XSpec_M1 = '/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/XSpec/Model1/fit_summary.csv'
csv_file_Minuit_M1 = '/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/Minuit/Model1/fit_summary.csv'
csv_file_XSpec_M3 = '/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/XSpec/Model3/fit_summary.csv'
csv_file_Minuit_M3 = '/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/Minuit/Model3/fit_summary.csv'
csv_file_XSpec_M4 = '/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/XSpec/Model4/fit_summary.csv'
csv_file_XSpec_M5 = '/home/jortecal/GitHub/eRosita/3MLFits/FitJune/Results/XSpec/Model5/fit_summary.csv'


# Read CSV
df_XSpec_M1 = pd.read_csv(csv_file_XSpec_M1)
df_Minuit_M1 = pd.read_csv(csv_file_Minuit_M1)
df_XSpec_M2 = pd.read_csv(csv_file_XSpec_M2)
df_Minuit_M2 = pd.read_csv(csv_file_Minuit_M2)
df_XSpec_M3 = pd.read_csv(csv_file_XSpec_M3)
df_Minuit_M3 = pd.read_csv(csv_file_Minuit_M3)
df_XSpec_M4 = pd.read_csv(csv_file_XSpec_M4)
df_XSpec_M5 = pd.read_csv(csv_file_XSpec_M5)


# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_XSpec_M1 = df_XSpec_M1['nbins_as'] - df_XSpec_M1['nfit_as']
vdof_dm_XSpec_M1 = df_XSpec_M1['nbins_dm'] - df_XSpec_M1['nfit_dm']

# Calculate p-value
df_XSpec_M1['p_value_as'] = chi2.sf(df_XSpec_M1['TS_as'], vdof_astro_XSpec_M1)
df_XSpec_M1['p_value_dm'] = chi2.sf(df_XSpec_M1['TS_all'], vdof_dm_XSpec_M1)

# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_XSpec_M2 = df_XSpec_M2['nbins_as'] - df_XSpec_M2['nfit_as']
vdof_dm_XSpec_M2 = df_XSpec_M2['nbins_dm'] - df_XSpec_M2['nfit_dm']

# Calculate p-value
df_XSpec_M2['p_value_as'] = chi2.sf(df_XSpec_M2['TS_as'], vdof_astro_XSpec_M2)
df_XSpec_M2['p_value_dm'] = chi2.sf(df_XSpec_M2['TS_all'], vdof_dm_XSpec_M2)


# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_XSpec_M3 = df_XSpec_M3['nbins_as'] - df_XSpec_M3['nfit_as']
vdof_dm_XSpec_M3 = df_XSpec_M3['nbins_dm'] - df_XSpec_M3['nfit_dm']

# Calculate p-value
df_XSpec_M3['p_value_as'] = chi2.sf(df_XSpec_M3['TS_as'], vdof_astro_XSpec_M3)
df_XSpec_M3['p_value_dm'] = chi2.sf(df_XSpec_M3['TS_all'], vdof_dm_XSpec_M3)

# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_Minuit_M1 = df_Minuit_M1['nbins_as'] - df_Minuit_M1['nfit_as']
vdof_dm_Minuit_M1 = df_Minuit_M1['nbins_dm'] - df_Minuit_M1['nfit_dm']

# Calculate p-value
df_Minuit_M1['p_value_as'] = chi2.sf(df_Minuit_M1['TS_as'], vdof_astro_Minuit_M1)
df_Minuit_M1['p_value_dm'] = chi2.sf(df_Minuit_M1['TS_all'], vdof_dm_Minuit_M1)

# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_Minuit_M2 = df_Minuit_M2['nbins_as'] - df_Minuit_M2['nfit_as']
vdof_dm_Minuit_M2 = df_Minuit_M2['nbins_dm'] - df_Minuit_M2['nfit_dm']

# Calculate p-value
df_Minuit_M2['p_value_as'] = chi2.sf(df_Minuit_M2['TS_as'], vdof_astro_Minuit_M2)
df_Minuit_M2['p_value_dm'] = chi2.sf(df_Minuit_M2['TS_all'], vdof_dm_Minuit_M2)


# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_Minuit_M3 = df_Minuit_M3['nbins_as'] - df_Minuit_M3['nfit_as']
vdof_dm_Minuit_M3 = df_Minuit_M3['nbins_dm'] - df_Minuit_M3['nfit_dm']

# Calculate p-value
df_Minuit_M3['p_value_as'] = chi2.sf(df_Minuit_M3['TS_as'], vdof_astro_Minuit_M3)
df_Minuit_M3['p_value_dm'] = chi2.sf(df_Minuit_M3['TS_all'], vdof_dm_Minuit_M3)


# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_XSpec_M4 = df_XSpec_M4['nbins_as'] - df_XSpec_M4['nfit_as']
vdof_dm_XSpec_M4 = df_XSpec_M4['nbins_dm'] - df_XSpec_M4['nfit_dm']

# Calculate p-value
df_XSpec_M4['p_value_as'] = chi2.sf(df_XSpec_M4['TS_as'], vdof_astro_XSpec_M4)
df_XSpec_M4['p_value_dm'] = chi2.sf(df_XSpec_M4['TS_all'], vdof_dm_XSpec_M4)

# Set degrees of freedom for TS_astro (change as needed)
vdof_astro_XSpec_M5 = df_XSpec_M5['nbins_as'] - df_XSpec_M5['nfit_as']
vdof_dm_XSpec_M5 = df_XSpec_M5['nbins_dm'] - df_XSpec_M5['nfit_dm']

# Calculate p-value
df_XSpec_M5['p_value_as'] = chi2.sf(df_XSpec_M5['TS_as'], vdof_astro_XSpec_M5)
df_XSpec_M5['p_value_dm'] = chi2.sf(df_XSpec_M5['TS_all'], vdof_dm_XSpec_M5)




# # Plot the data
# fig, ax = plt.subplots()
# fig.set_size_inches(21.5, 13.5)

# plt.plot(df_XSpec_M1['Eline (keV)'], df_XSpec_M1['ΔTS'], label='XSpec Model 1', marker='o', markersize=2)
# plt.plot(df_Minuit_M1['Eline (keV)'], df_Minuit_M1['ΔTS'], label='Minuit Model 1', marker='*', markersize=2)

# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30) 

# plt.xlabel(r'E (keV)',fontsize=30)
# plt.ylabel(r'$\Delta$TS',fontsize=30)

# plt.xscale('log')
# plt.ylim(0, 30)

# plt.legend(fontsize=30, loc='upper right')
# plt.show()

# # Plot the data
# fig, ax = plt.subplots()
# fig.set_size_inches(21.5, 13.5)

# plt.plot(df_XSpec_M2['Eline (keV)'], df_XSpec_M2['ΔTS'], label='XSpec Model 2', marker='o', markersize=2)
# plt.plot(df_Minuit_M2['Eline (keV)'], df_Minuit_M2['ΔTS'], label='Minuit Model 2', marker='*', markersize=2)

# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30) 

# plt.xlabel(r'E (keV)',fontsize=30)
# plt.ylabel(r'$\Delta$TS',fontsize=30)

# plt.xscale('log')
# plt.ylim(0, 30)

# plt.legend(fontsize=30, loc='upper right')
# plt.show()

# Plot the data
fig, ax = plt.subplots()
fig.set_size_inches(21.5, 13.5)

plt.plot(df_XSpec_M1['Eline (keV)'], df_XSpec_M1['ΔTS'], label='XSpec Model 1', marker='o', linestyle='-.', markersize=2)
plt.plot(df_XSpec_M2['Eline (keV)'], df_XSpec_M2['ΔTS'], label='XSpec Model 2', marker='*', linestyle='--', markersize=2) 
plt.plot(df_XSpec_M3['Eline (keV)'], df_XSpec_M3['ΔTS'], label='XSpec Model 3', marker='s', linestyle=':', markersize=2)
plt.plot(df_XSpec_M4['Eline (keV)'], df_XSpec_M4['ΔTS'], label='XSpec Model 4', marker='1', markersize=2)
plt.plot(df_XSpec_M5['Eline (keV)'], df_XSpec_M5['ΔTS'], label='XSpec Model 5', marker='x', markersize=2)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30) 

plt.xlabel(r'E (keV)',fontsize=30)
plt.ylabel(r'$\Delta$TS',fontsize=30)

plt.xscale('log')
plt.ylim(0, 30)

plt.legend(fontsize=20)
plt.show()

# Plot the data
fig, ax = plt.subplots()
fig.set_size_inches(21.5, 13.5)

plt.plot(df_XSpec_M1['Eline (keV)'], df_XSpec_M1['A_95'], label='XSpec Model 1', marker='o', linestyle='-.', markersize=2)
plt.plot(df_XSpec_M2['Eline (keV)'], df_XSpec_M2['A_95'], label='XSpec Model 2', marker='*', linestyle='--', markersize=2) 
plt.plot(df_XSpec_M3['Eline (keV)'], df_XSpec_M3['A_95'], label='XSpec Model 3', marker='s', linestyle=':', markersize=2)
plt.plot(df_XSpec_M4['Eline (keV)'], df_XSpec_M4['A_95'], label='XSpec Model 4', marker='1', markersize=2)
plt.plot(df_XSpec_M5['Eline (keV)'], df_XSpec_M5['A_95'], label='XSpec Model 5', marker='x', markersize=2)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30) 

plt.xlabel(r'E (keV)',fontsize=30)
plt.ylabel(r'$A_{95}$',fontsize=30)

plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-6, 1e-1)

plt.legend(fontsize=20)
plt.show()


# # Plot the data
# fig, ax = plt.subplots()
# fig.set_size_inches(21.5, 13.5)

# plt.plot(df_Minuit_M1['Eline (keV)'], df_Minuit_M1['ΔTS'], label='Minuit Model 1', marker='o', markersize=2)
# plt.plot(df_Minuit_M2['Eline (keV)'], df_Minuit_M2['ΔTS'], label='Minuit Model 2', marker='*', markersize=2)

# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30) 

# plt.xlabel(r'E (keV)',fontsize=30)
# plt.ylabel(r'$\Delta$TS',fontsize=30)

# plt.xscale('log')
# plt.ylim(0, 30)

# plt.legend(fontsize=30, loc='upper right')
# plt.show()


# # Plot the data
# fig, ax = plt.subplots()
# fig.set_size_inches(21.5, 13.5)

# plt.plot(df_Minuit_M3['Eline (keV)'], df_Minuit_M3['ΔTS'], label='Minuit Model 3', marker='o', markersize=2)
# plt.plot(df_XSpec_M3['Eline (keV)'], df_XSpec_M3['ΔTS'], label='XSpec Model 3', marker='*', markersize=2)

# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30) 

# plt.xlabel(r'E (keV)',fontsize=30)
# plt.ylabel(r'$\Delta$TS',fontsize=30)

# plt.xscale('log')
# plt.ylim(0, 30)

# plt.legend(fontsize=30, loc='upper right')
# plt.show()

# # Plot the data
# fig, ax = plt.subplots()
# fig.set_size_inches(21.5, 13.5)

# plt.plot(df_XSpec_M1['Eline (keV)'], df_XSpec_M1['p_value_as'], label='XSpec Model 1', marker='o', markersize=2)
# plt.plot(df_Minuit_M1['Eline (keV)'], df_Minuit_M1['p_value_as'], label='Minuit Model 1', marker='*', markersize=2)

# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30) 

# plt.xlabel(r'E (keV)',fontsize=30)
# plt.ylabel(r'p-value',fontsize=30)
# plt.ylim(1e-15, 1)
# plt.xscale('log')
# plt.yscale('log')


# plt.legend(fontsize=30, loc='upper right')
# plt.show()


# # Plot the data
# fig, ax = plt.subplots()
# fig.set_size_inches(21.5, 13.5)

# plt.plot(df_XSpec_M2['Eline (keV)'], df_XSpec_M2['p_value_as'], label='XSpec Model 2', marker='o', markersize=2)
# plt.plot(df_Minuit_M2['Eline (keV)'], df_Minuit_M2['p_value_as'], label='Minuit Model 2', marker='*', markersize=2)

# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30) 

# plt.xlabel(r'E (keV)',fontsize=30)
# plt.ylabel(r'p-value',fontsize=30)
# plt.ylim(1e-15, 1)
# plt.xscale('log')
# plt.yscale('log')


# plt.legend(fontsize=30, loc='upper right')
# plt.show()



# # Plot the data
# fig, ax = plt.subplots()
# fig.set_size_inches(21.5, 13.5)

# plt.plot(df_Minuit_M1['Eline (keV)'], df_Minuit_M1['p_value_as'], label='Minuit Model 1', marker='o', markersize=2)
# plt.plot(df_Minuit_M2['Eline (keV)'], df_Minuit_M2['p_value_as'], label='Minuit Model 2', marker='*', markersize=2)

# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30) 

# plt.xlabel(r'E (keV)',fontsize=30)
# plt.ylabel(r'p-value',fontsize=30)
# plt.ylim(1e-15, 1)
# plt.xscale('log')
# plt.yscale('log')


# plt.legend(fontsize=30, loc='upper right')
# plt.show()

# Plot the data
fig, ax = plt.subplots()
fig.set_size_inches(21.5, 13.5)

plt.plot(df_XSpec_M1['Eline (keV)'], df_XSpec_M1['p_value_as'], label='XSpec Model 1', marker='o', linestyle='-.', markersize=2)
plt.plot(df_XSpec_M2['Eline (keV)'], df_XSpec_M2['p_value_as'], label='XSpec Model 2', marker='*', linestyle='--', markersize=2)
plt.plot(df_XSpec_M3['Eline (keV)'], df_XSpec_M3['p_value_as'], label='XSpec Model 3', marker='s', linestyle=':', markersize=2)
plt.plot(df_XSpec_M4['Eline (keV)'], df_XSpec_M4['p_value_as'], label='XSpec Model 4', marker='1', markersize=2)
plt.plot(df_XSpec_M5['Eline (keV)'], df_XSpec_M5['p_value_as'], label='XSpec Model 5', marker='x', markersize=2)
hline = plt.axhline(y=0.05, color='r', linestyle='--', label='p-value = 0.05')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30) 

plt.xlabel(r'E (keV)',fontsize=30)
plt.ylabel(r'p-value',fontsize=30)
plt.ylim(1e-15, 1)
plt.xscale('log')
plt.yscale('log')


plt.legend(fontsize=20)
plt.show()
# Plot the data
# fig, ax = plt.subplots()
# fig.set_size_inches(21.5, 13.5)

# plt.plot(df_Minuit_M3['Eline (keV)'], df_Minuit_M3['p_value_as'], label='Minuit Model 3', marker='o', markersize=2)
# plt.plot(df_XSpec_M3['Eline (keV)'], df_XSpec_M3['p_value_as'], label='XSpec Model 3', marker='*', markersize=2)

# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30) 

# plt.xlabel(r'E (keV)',fontsize=30)
# plt.ylabel(r'p-value',fontsize=30)
# plt.ylim(1e-15, 1)
# plt.xscale('log')
# plt.yscale('log')


# plt.legend(fontsize=30, loc='upper right')
# plt.show()
# Plot p-value 