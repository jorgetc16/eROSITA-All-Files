import numpy as np

data_corr = np.loadtxt("/home/jortecal/GitHub/eRosita/RefereeChecks/nH/Marco_nH/CorrectionFactor.txt",skiprows=1)

vE_corr = data_corr[:,0] #keV
vcorr = data_corr[:,1]


print(vE_corr)
print(vcorr)

Eline = 5

original_flux_bound = 1e-5
corrfactor = np.interp(Eline,vE_corr,vcorr)

new_flux_bound = original_flux_bound*corrfactor

print("Eline ",Eline)
print("original_flux_bound ",original_flux_bound)
print("new_flux_bound ",new_flux_bound)
print("corrfactor ",corrfactor)

