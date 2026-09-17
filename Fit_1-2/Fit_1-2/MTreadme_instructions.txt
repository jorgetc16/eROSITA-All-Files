


python3 Results_read.py

python3 Results_read.py combine /Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_global/Fit_1-2/output/ /Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_global/Fit_1-2/output/CombinedProfiles/


python3 CombinedBounds.py /Users/marcotaoso/Documents/2024/eROSITA/xSpec/LMC_5deg/Fit_global/Fit_1-2/output/CombinedProfiles/

python3 PlotResults.py



Show combined profile plot
python3 CombinedBounds.py output/CombinedProfiles 1.261



# show combined profile and profile of all TMs
python3 Results_read.py plot_all output/ 1.208




TYPEFIT=1 NSTEPS=100 MODEL=1 MAX_PROCS=7 ./RunFitParallelLow.sh








Lo corres con "python AnalysisPipeline.py /path/Otuput -o ./Results"