#!/bin/sh

# eROSITA generated download script (Jeremy Sanders 2023)
# Please report problems on https://erosita-forum.mpe.mpg.de/
#
# Note: requires curl installed
#
# If given a directory argument, write to that directory.  Otherwise,
# asks user which directory to write to (current or elsewhere).

set -e

root="https://erosita.mpe.mpg.de/dr1/erodat/data/download/"

if ! command -v curl >/dev/null 2>&1 ; then
    echo >&2 "Error: cannot find curl. Please install."
    exit 1
fi

download () {
    fpath="$1"
    url="${root}${fpath}"
    echo "Downloading $url"

    dir="$(dirname "$fpath")"
    base="$(basename "$fpath")"
    mkdir -p "$dir"
    (
        cd "$dir"
        if test -e "$base"; then zflag="-z $base"; else zflag=; fi # do not download if unchanged
        curl $zflag --fail -R -O "$url"
    )
    echo
}

echo "eROSITA data download script"
echo "============================"
if [ "$#" -gt 0 ]; then
    first="$(echo "$1" | head -c 1)"
    if [ "$first" = "-" ] || [ "$#" -gt 1 ]; then
        echo >&2 "$0 [output directory]"
        exit 1
    fi
    outdir="$1"
else
    echo "Press [Enter] to download to the current directory, enter a directory name, or press Ctrl+C to cancel"
    printf "> "
    read -r outdir
    if [ "$outdir" = "" ]; then
        outdir="$(pwd)"
    fi
fi

cd "$outdir"

download 153/079/EXP_010/em01_079153_020_EventList_c010.fits.gz
download 153/085/EXP_010/em01_085153_020_EventList_c010.fits.gz
download 156/066/EXP_010/em01_066156_020_EventList_c010.fits.gz
download 156/073/EXP_010/em01_073156_020_EventList_c010.fits.gz
download 156/080/EXP_010/em01_080156_020_EventList_c010.fits.gz
download 156/087/EXP_010/em01_087156_020_EventList_c010.fits.gz
download 156/093/EXP_010/em01_093156_020_EventList_c010.fits.gz
download 159/067/EXP_010/em01_067159_020_EventList_c010.fits.gz
download 159/074/EXP_010/em01_074159_020_EventList_c010.fits.gz
download 159/082/EXP_010/em01_082159_020_EventList_c010.fits.gz
download 159/090/EXP_010/em01_090159_020_EventList_c010.fits.gz
download 159/098/EXP_010/em01_098159_020_EventList_c010.fits.gz
download 162/068/EXP_010/em01_068162_020_EventList_c010.fits.gz
download 162/077/EXP_010/em01_077162_020_EventList_c010.fits.gz
download 162/086/EXP_010/em01_086162_020_EventList_c010.fits.gz
download 162/095/EXP_010/em01_095162_020_EventList_c010.fits.gz
download 165/069/EXP_010/em01_069165_020_EventList_c010.fits.gz
download 165/079/EXP_010/em01_079165_020_EventList_c010.fits.gz
download 165/090/EXP_010/em01_090165_020_EventList_c010.fits.gz
