import h5py
import pandas as pd
import numpy as np
import glob
import re
import os
import sys
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

def summarize_fit_results(folder):
    result_rows = []
    files = sorted(glob.glob(os.path.join(folder, "fit_*.h5")))
    if not files:
        print("No HDF5 files found in", folder)
        return
    Eline_array = []
    TBabsNH2 = []
    TAPEC = []
    normAPEC = []
    TPLgamma = []
    for file in files:
        try:
            match = re.search(r"_E_(\d+\.\d+)\.h5", file)
            Eline = float(match.group(1)) if match else np.nan
            Eline_array = np.append(Eline_array,Eline)
            
            with h5py.File(file, "r") as f:
                grp = f[f"E_{Eline:.3f}"]
                dm_grp = grp["dm"]

                # Read raw parameter array (no names)
                dm_param_values = dm_grp["params"][:]
                dm_param_values = dm_param_values.astype(float)
                TBabsNH2 = np.append(TBabsNH2,dm_param_values[0])
                TAPEC = np.append(TAPEC,dm_param_values[2])
                normAPEC = np.append(normAPEC,dm_param_values[5])
                TPLgamma = np.append(TPLgamma,dm_param_values[11])

#Array: first mw and then pb
#mw: TABS, APEC(T,abund,z,norm),PL(PhoIndex,norm), GaussianDM(LineE,Sigma,norm)
#pb: PL(PhoIndex,norm) + Gaussian_IB(LineE,Sigma,norm)


        except Exception as e:
            print(f"Error reading {file}: {e}")

    #df = pd.DataFrame(result_rows).sort_values("Eline (keV)")
    #print(df.to_string(index=False))

    #if output_csv:
    #    df.to_csv(output_csv, index=False)
    #    print(f"\nSummary saved to {output_csv}")
    
    return Eline_array, TBabsNH2, TAPEC, normAPEC, TPLgamma


#Eline, TBabsNH2,TAPEC, TPLgamma = summarize_fit_results("/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_windows/Tests/1-2keV/output/TM1")
#print("Eline ",Eline)
#print("TBabsNH2 ",TBabsNH2)

# TM numbers you want to include
TM_list = [1, 2, 3, 4, 5, 6, 7]

all_Eline = []
all_TBabsNH2 = []
all_TAPEC = []
all_normAPEC = []
all_TPLgamma = []

# Loop over folders
for tm in TM_list:
    folder = f"/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_windows/1-2keV_fixednHDM/output/TM{tm}"
    print(f"Processing {folder}")

    Eline, TBabsNH2, TAPEC, normAPEC, TPLgamma = summarize_fit_results(folder)
    
    all_Eline.append(Eline)
    all_TBabsNH2.append(TBabsNH2)
    all_TAPEC.append(TAPEC)
    all_normAPEC.append(normAPEC)
    all_TPLgamma.append(TPLgamma)


print(all_Eline," shape ",all_TBabsNH2)

fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()


plt.axhline(y=0.01,color='gray',linestyle='dotted')
plt.axhline(y=0.8,color='gray',linestyle='dotted')

#plt.plot(Eline,TBabsNH2, marker='o', linestyle='-', markersize=2)
for tm, Eline, TBabsNH2 in zip(TM_list, all_Eline, all_TBabsNH2):
    plt.plot(Eline, TBabsNH2, marker='o', linestyle='-', markersize=2, label=f"TM{tm}")


plt.xlabel(r'E (keV)',fontsize=30)
plt.ylabel(r'TBABS nH2',fontsize=30)
# plt.ylim(1e-3, 1)
plt.xscale('log')
plt.yscale('log')

ax1.tick_params(which='major',direction='in',width=1, labelsize=22,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=15)
plt.xlim(1.0, 2.)
# plt.ylim(-1, 37)
plt.tight_layout()
# plt.show()
plt.savefig('TBABS_nH2.pdf')





fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()
plt.axhline(y=0.05,color='gray',linestyle='dotted')
plt.axhline(y=3.0,color='gray',linestyle='dotted')

#plt.plot(Eline,TAPEC, marker='o', linestyle='-', markersize=2)
for tm, Eline, TAPEC in zip(TM_list, all_Eline, all_TAPEC):
    plt.plot(Eline, TAPEC, marker='o', linestyle='-', markersize=2, label=f"TM{tm}")


plt.xlabel(r'E (keV)',fontsize=30)
plt.ylabel(r'TAPEC',fontsize=30)
# plt.ylim(1e-3, 1)
plt.xscale('log')
#plt.yscale('log')

ax1.tick_params(which='major',direction='in',width=1, labelsize=22,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=15)
plt.xlim(1.0, 2.)
# plt.ylim(-1, 37)
plt.tight_layout()
# plt.show()
plt.savefig('TAPEC.pdf')



fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()
plt.axhline(y=60,color='gray',linestyle='dotted')
plt.axhline(y=2.0,color='gray',linestyle='dotted')

#plt.plot(Eline,TAPEC, marker='o', linestyle='-', markersize=2)
for tm, Eline, normAPEC in zip(TM_list, all_Eline, all_normAPEC):
    plt.plot(Eline, normAPEC, marker='o', linestyle='-', markersize=2, label=f"TM{tm}")


plt.xlabel(r'E (keV)',fontsize=30)
plt.ylabel(r'norm APEC',fontsize=30)
# plt.ylim(1e-3, 1)
plt.xscale('log')
plt.yscale('log')

ax1.tick_params(which='major',direction='in',width=1, labelsize=22,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=15)
plt.xlim(1.0, 2.)
plt.ylim(1e-3, 70)
plt.tight_layout()
# plt.show()
plt.savefig('normAPEC.pdf')




fig = plt.figure(figsize=(8,7))
ax1 = plt.subplot()
plt.axhline(y=-2.5,color='gray',linestyle='dotted')
plt.axhline(y=4.5,color='gray',linestyle='dotted')

#plt.plot(Eline,TPLgamma, marker='o', linestyle='-', markersize=2)
for tm, Eline, TPLgamma in zip(TM_list, all_Eline, all_TPLgamma):
    plt.plot(Eline, TPLgamma, marker='o', linestyle='-', markersize=2, label=f"TM{tm}")
    
plt.xlabel(r'E (keV)',fontsize=30)
plt.ylabel(r'TPLgamma',fontsize=30)
# plt.ylim(1e-3, 1)
plt.xscale('log')
#plt.yscale('log')

ax1.tick_params(which='major',direction='in',width=1, labelsize=22,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in', labelsize=22, width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=15)
plt.xlim(1.0, 2.)
# plt.ylim(-1, 37)
plt.tight_layout()
# plt.show()
plt.savefig('TPLgamma.pdf')

