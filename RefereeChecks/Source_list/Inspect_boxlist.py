"""
Quick inspection of an erbox output boxlist FITS file.
Usage: python inspect_boxlist.py boxlist_comb_LMC_rad5deg_rebin80.fits
"""
import sys
import pandas as pd
from astropy.io import fits
from astropy.table import Table

path = sys.argv[1] if len(sys.argv) > 1 else "boxlist_comb_LMC_rad5deg_rebin80.fits"

with fits.open(path) as hdul:
    hdul.info()
    data = Table(hdul[1].data)

print(f"\nNumber of candidate sources detected by erbox: {len(data)}")
print("\nColumns available:", data.colnames)

# Basic stats useful for the referee reply
print(f"\nDetection likelihood (LIKE) range: {data['like'].min():.1f} - {data['like'].max():.1f}")
print(f"Flux range [erg/cm^2/s]: {data['flux'].min():.2e} - {data['flux'].max():.2e}")

# Report on the structure: id_band and box_size columns
band_info = data['id_band'] if 'id_band' in data.colnames else None
box_info = data['box_size'] if 'box_size' in data.colnames else None
if band_info is not None:
    print(f"\nUnique id_band values: {set(band_info)}")
if box_info is not None:
    print(f"Unique box_size values: {set(box_info)}")

# Keep only unique sources: if id_band == 0 exists (combined band), use that.
# Otherwise, keep only the rows with the smallest box_size (native resolution).
# Then deduplicate by RA/Dec to remove the per-nruns multi-row structure.
if band_info is not None and 0 in set(band_info):
    combined = data[data['id_band'] == 0]
    print(f"\nFiltered to id_band=0: {len(combined)} rows.")
elif box_info is not None:
    min_box = box_info.min()
    combined = data[data['box_size'] == min_box]
    print(f"\nFiltered to box_size={min_box}: {len(combined)} rows.")
else:
    combined = data
    print(f"\nNo id_band/box_size columns; using all {len(combined)} rows.")

# Drop duplicate RA/Dec positions (keep first occurrence)
n_before = len(combined)
df = combined.to_pandas()
df = df.drop_duplicates(subset=['ra', 'dec'], keep='first')
combined = Table.from_pandas(df)
print(f"Deduplicated by RA/Dec: {len(combined)} unique sources "
      f"(removed {n_before - len(combined)} duplicate rows).")

# Save a plain RA/DEC/LIKE/FLUX table for easy cross-matching (e.g. with astroquery)
cols_to_save = ['ra', 'dec', 'like', 'flux', 'rate']
for col in ['id_band', 'box_size']:
    if col in combined.colnames:
        cols_to_save.append(col)
out = combined[cols_to_save]
out.write("/home/jortecal/GitHub/eRosita/RefereeChecks/Source_list/boxlist_summary.csv", format="csv", overwrite=True)
print("\nWrote boxlist_summary.csv with RA, DEC, LIKE, FLUX, RATE for cross-matching.")