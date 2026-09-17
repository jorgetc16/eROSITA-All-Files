#!/bin/bash

export OMP_NUM_THREADS=7
#erbox images="image_LMC_selected_region_3deg_rebin40.fits" boxlist="boxlist_LMC_selected_region_3deg_rebin40.fits" emin=500 emax=10000 expimages="expmap_LMC_selected_region_3deg_rebin40.fits" detmasks="detmask_LMC_selected_region_3deg_rebin40.fits" bkgima_flag=N ecf=1
#erbackmap image="image_LMC_selected_region_3deg_rebin40.fits" expimage="expmap_LMC_selected_region_3deg_rebin40.fits" boxlist="boxlist_LMC_selected_region_3deg_rebin40.fits" detmask="detmask_LMC_selected_region_3deg_rebin40.fits" bkgimage="bkg_map_LMC_selected_region_3deg_rebin40.fits" emin=500 emax=10000 cheesemask="cheesemask_LMC_selected_region_3deg_rebin40.fits" cheesemask_flag="Y" maxcut=0.7
srctool eventfiles="LMC_selected_region_3deg.fits" exttype="TOPHAT" srcreg="cheesemask_LMC_selected_region_3deg_rebin40.fits" backreg="NONE" extpars=10800 srccoord="fk5;80.89417,-69.75611" todo="SPEC ARF RMF" tstep=2 xgrid=4 psftype="NONE"  tarball="tar.gz"

