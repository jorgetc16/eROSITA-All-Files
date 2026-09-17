import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
import numpy as np

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}' #package mathpazo siunitx
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') #serif
plt.rcParams['axes.linewidth'] = 2


def extract_profile_bounds(folder):
    vEline = []
    vbounds = []
    files = sorted(glob.glob(os.path.join(folder, "fit_*.h5")))
    if not files:
        print("No HDF5 files found in", folder)
        return

    for file in files:
        try:
            match = re.search(r"_E_(\d+\.\d+)\.h5", file)
            Eline = float(match.group(1)) if match else np.nan

            with h5py.File(file, "r") as f:
                grp = f[f"E_{Eline:.3f}"]
                dm_grp = grp["dm"]
                
                profile = grp["profile"]

                A_vals = profile["A_values"][:]
                TS_vals = profile["TS_values"][:]
                TS_all = dm_grp["TS_all"][()]

                delta_TS = TS_vals - TS_all
                #print("A_vals ",A_vals)
                #print("delta_TS ",delta_TS)
                x = np.log10(A_vals)
                y = delta_TS - TSth
                vroots = np.pow(10.,MTroots(x,y))
                if(len(vroots)>0):
                    bound = np.max(vroots)
                else:
                    bound = 0.
                vEline.append(Eline)
                vbounds.append(bound)

        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    Eline = np.array(Eline)
    vbounds = np.array(vbounds)
    return  (vEline, vbounds)
    
###########################################
# File path
path = '/Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_test_3deg/Fitpoly/Fit/XSpec/Model4/TMsingle_rebinned/'
csv_file_XSpec_MT1 = path+'TM1/fit_summary.csv'
csv_file_XSpec_MT2 = path+'TM2/fit_summary.csv'
csv_file_XSpec_MT3 = path+'TM3/fit_summary.csv'
csv_file_XSpec_MT4 = path+'TM4/fit_summary.csv'
csv_file_XSpec_MT5 = path+'TM5/fit_summary.csv'
csv_file_XSpec_MT6 = path+'TM6/fit_summary.csv'
csv_file_XSpec_MT7 = path+'TM7/fit_summary.csv'


csv_file_XSpec = [csv_file_XSpec_MT6]#,csv_file_XSpec_MT2,csv_file_XSpec_MT3,csv_file_XSpec_MT4,csv_file_XSpec_MT5,csv_file_XSpec_MT6,csv_file_XSpec_MT7]
nTM = len(csv_file_XSpec)
print("nTM ",nTM)

for i in range(0,len(csv_file_XSpec)):
    df_XSpec = pd.read_csv(csv_file_XSpec[i])
    nvalidp = len(df_XSpec)
    print("number of valid points ",nvalidp)
######################################################################################

######################################################################################

color_TM = 'lightgray'#'mediumseagreen'

# Plot TS/ndf vs Eline
plt.figure()
ax1 = plt.subplot()
#
for i in range(0,nTM):
    df_XSpec = pd.read_csv(csv_file_XSpec[i])
    vEline = df_XSpec['Eline (keV)']
    vTS_as = df_XSpec['TS_as']
    vdof_as = df_XSpec['nbins_as'] - df_XSpec['nfit_as']
    ax1.plot(vEline, vTS_as/vdof_as, color=color_TM,label='')
#
#ax1.plot(vEline[0], vTS_as_tot/vdof_as_tot, color='black',label='')
#
ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$E_{\rm line}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$TS_{\rm as}/dof$', size=24)
#ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(0, 5)
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig('TS_as_xdof.pdf')



# Plot p-value vs Eline
plt.figure()
ax1 = plt.subplot()
for i in range(0,nTM):
    df_XSpec = pd.read_csv(csv_file_XSpec[i])
    vEline = df_XSpec['Eline (keV)']
    vTS_as = df_XSpec['TS_as']
    vdof_as = df_XSpec['nbins_as'] - df_XSpec['nfit_as']
    vp_value_as = chi2.sf(vTS_as, vdof_as)
    print(np.column_stack((vEline, vp_value_as, vTS_as)))
    print("************************")
    ax1.plot(vEline, vp_value_as, color=color_TM,label='')

#
#ax1.plot(vEline[0], vp_value_as_tot, color='black',label='')
#
ax1.axhline(y=2.7e-3, color='black',linestyle=':',linewidth='0.5')
ax1.axhline(y=5.7e-7, color='black',linestyle=':',linewidth='0.5')
#
ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$E_{\rm line}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'p-value', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(1e-15,1.)
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig('p-value.pdf')




# Plot DeltaTS vs Eline
plt.figure()
ax1 = plt.subplot()
for i in range(0,nTM):
    df_XSpec = pd.read_csv(csv_file_XSpec[i])
    vEline = df_XSpec['Eline (keV)']
    vΔTS = df_XSpec['ΔTS']
    ax1.plot(vEline, vΔTS, color=color_TM,label='')
ax1.axhline(y=9, color='black',linestyle=':',linewidth='0.5')
ax1.axhline(y=25, color='black',linestyle=':',linewidth='0.5')
#
ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$E_{\rm line}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$\Delta TS$', size=24)
#ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(0, 30)
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig('DeltaTS.pdf')





plt.figure()
ax1 = plt.subplot()
#
for i in range(0,nTM):
    df_XSpec = pd.read_csv(csv_file_XSpec[i])
    vEline = df_XSpec['Eline (keV)']
    vA95 = df_XSpec['A_95']
    ax1.plot(vEline, vA95, color=color_TM,label='')

#
ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$E_{\rm line}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$\Phi\, [{\rm cm}^2 {\rm s}^{-1}]$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')
#ax1.set_ylim(1e-15,1.)
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig('ULimFlux.pdf')





'''
vnbins_as = np.zeros((nTM, nvalidp))
vnbins_dm = np.zeros((nTM, nvalidp))
vnfit_as = np.zeros((nTM, nvalidp))
vnfit_dm = np.zeros((nTM, nvalidp))
vTS_as = np.zeros((nTM, nvalidp))
vTS_dm = np.zeros((nTM, nvalidp))
vEline = np.zeros((nTM, nvalidp))
vdof_as = np.zeros((nTM, nvalidp))
vdof_dm = np.zeros((nTM, nvalidp))
vp_value_as = np.zeros((nTM, nvalidp))
vp_value_dm = np.zeros((nTM, nvalidp))
vA95 = np.zeros((nTM, nvalidp))


for i in range(0,nTM):
    df_XSpec = pd.read_csv(csv_file_XSpec[i])
    #print(np.shape(df_XSpec['nbins_as']))
    #print(np.shape(vnbins_as[i]))
    vnbins_as[i] = df_XSpec['nbins_as']
    vnbins_dm[i] = df_XSpec['nbins_dm']
    vnfit_as[i] = df_XSpec['nfit_as']
    vnfit_dm[i] = df_XSpec['nfit_dm']
    vTS_as[i] = df_XSpec['TS_as']
    vTS_dm[i] = df_XSpec['TS_all']
    vEline[i] = df_XSpec['Eline (keV)']
    vA95[i] = df_XSpec['A_95']
    #
    vdof_as[i] = vnbins_as[i] - vnfit_as[i]
    vdof_dm[i] = vnbins_dm[i] - vnfit_dm[i]
    vp_value_as[i] = chi2.sf(vTS_as[i], vdof_as[i])
    vp_value_dm[i] = chi2.sf(vTS_dm[i], vdof_dm[i])


vTS_as_tot = np.zeros(nvalidp)
vTS_dm_tot = np.zeros(nvalidp)
vdof_as_tot = np.zeros(nvalidp)
vdof_dm_tot = np.zeros(nvalidp)

for i in range(0,nTM):
    vTS_as_tot = vTS_as_tot + vTS_as[i]
    vTS_dm_tot = vTS_dm_tot + vTS_dm[i]
    vdof_as_tot = vdof_as_tot + vdof_as[i]
    vdof_dm_tot = vdof_dm_tot + vdof_dm[i]

vp_value_as_tot = chi2.sf(vTS_as_tot, vdof_as_tot)
vp_value_dm_tot = chi2.sf(vTS_dm_tot, vdof_dm_tot)

######################################################################################

color_TM = 'lightgray'#'mediumseagreen'

# Plot TS/ndf vs Eline
plt.figure()
ax1 = plt.subplot()
#
for i in range(0,nTM):
    ax1.plot(vEline[i], vTS_as[i]/vdof_as[i], color=color_TM,label='')
#
#ax1.plot(vEline[0], vTS_as_tot/vdof_as_tot, color='black',label='')
#
ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$E_{\rm line}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$TS_{\rm as}/dof$', size=24)
#ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(0, 5)
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig('TS_as_xdof.pdf')





# Plot p-value vs Eline
plt.figure()
ax1 = plt.subplot()
for i in range(0,nTM):
    ax1.plot(vEline[i], vp_value_as[i], color=color_TM,label='')

#
#ax1.plot(vEline[0], vp_value_as_tot, color='black',label='')
#
ax1.axhline(y=2.7e-3, color='black',linestyle=':',linewidth='0.5')
ax1.axhline(y=5.7e-7, color='black',linestyle=':',linewidth='0.5')
#
ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$E_{\rm line}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'p-value', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim(1e-15,1.)
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig('p-value.pdf')



plt.figure()
ax1 = plt.subplot()
#
for i in range(0,nTM):
    ax1.plot(vEline[i], vA95[i], color=color_TM,label='')

#
ax1.tick_params(which='major',direction='in',width=1,length=10,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='y', direction='in',width=1,length=7,top=True,right=True,pad=10)
ax1.tick_params(which='minor',axis='x', direction='in',width=1,length=7,top=True,right=True,pad=10)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax1.set_xlabel(r'$E_{\rm line}\, [{\rm keV}]$', size=24)
ax1.set_ylabel(r'$\Phi\, [{\rm cm}^2 {\rm s}^{-1}]$', size=24)
ax1.set_yscale('log')
ax1.set_xscale('log')
#ax1.set_ylim(1e-15,1.)
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig('ULimFlux.pdf')

'''

'''
ii=0
jj=130
#print(vdof_as[ii]-vdof_dm[ii])
print(vEline[ii][jj])
print(vTS_as[ii][jj])
print(vnbins_as[ii][jj])
print(vnfit_as[ii][jj])
print(vdof_as[ii][jj])
print(vp_value_as[ii][jj])
'''

