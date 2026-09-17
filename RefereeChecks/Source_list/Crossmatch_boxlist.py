"""
Cross-match the erbox source list (boxlist_summary.csv) against SIMBAD
using CDS XMatch (not SIMBAD TAP, whose async endpoint is unreliable).

Requires: pip install astroquery --break-system-packages

Usage: python crossmatch_boxlist.py boxlist_summary.csv
"""
import sys
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import Table
from astroquery.xmatch import XMatch

CHUNK_SIZE = 10000  # XMatch handles larger uploads reliably

path = sys.argv[1] if len(sys.argv) > 1 else "boxlist_summary.csv"
df = pd.read_csv(path)
print(f"Loaded {len(df)} rows from {path}")

# Deduplicate: if id_band column exists, keep only id_band == 0 (combined band).
# Otherwise drop duplicate RA/Dec positions (from multi-run erbox output).
if 'id_band' in df.columns:
    n_before = len(df)
    df = df[df['id_band'] == 0]
    print(f"Filtered to id_band=0: {len(df)} sources (removed {n_before - len(df)} duplicate rows).")
elif 'box_size' in df.columns:
    n_before = len(df)
    min_box = df['box_size'].min()
    df = df.sort_values('box_size').drop_duplicates(subset=['ra', 'dec'], keep='first')
    print(f"Deduplicated keeping box_size={min_box}: {len(df)} sources (removed {n_before - len(df)} rows).")
else:
    n_before = len(df)
    df = df.drop_duplicates(subset=['ra', 'dec'])
    print(f"Deduplicated by RA/Dec: {len(df)} sources (removed {n_before - len(df)} rows).")

MATCH_RADIUS_ARCSEC = 15.0  # 15 arcsec

df = df.reset_index(drop=True)
df.insert(0, "src_id", df.index)

# Build an astropy Table for XMatch (needs ra, dec columns in degrees)
input_table = Table.from_pandas(df[["src_id", "ra", "dec"]])

# Split into chunks to avoid timeouts
n_chunks = int(np.ceil(len(df) / CHUNK_SIZE))
print(f"\nCross-matching {len(df)} sources in {n_chunks} chunks"
      f" of up to {CHUNK_SIZE} via CDS XMatch...")

all_results = []
for k in range(n_chunks):
    i0 = k * CHUNK_SIZE
    i1 = min(i0 + CHUNK_SIZE, len(df))
    chunk = input_table[i0:i1]

    try:
        res = XMatch.query(
            cat1=chunk,
            cat2='simbad',
            max_distance=MATCH_RADIUS_ARCSEC * u.arcsec,
            colRA1='ra',
            colDec1='dec',
        )
    except Exception as exc:
        print(f"  Chunk {k+1}/{n_chunks} (sources {i0}-{i1}): FAILED — {exc}")
        continue

    n_hits = len(res) if res is not None else 0
    print(f"  Chunk {k+1}/{n_chunks} (sources {i0}-{i1}): {n_hits} SIMBAD matches")
    if res is not None and len(res) > 0:
        # Keep only relevant columns: our src_id + SIMBAD type + angDist
        # (angDist is kept explicitly so we can guarantee the CLOSEST match is
        # retained below, rather than relying on XMatch's undocumented result
        # ordering when deduplicating to one row per source.)
        res_df = res.to_pandas()
        cols_to_keep = ['src_id', 'angDist'] + [c for c in res_df.columns
                                     if c.startswith('main_type') or c == 'otype'
                                     or c.startswith('obj_type')]
        all_results.append(res_df[cols_to_keep].copy())

if not all_results:
    print("\nERROR: No cross-match results obtained."
          " Check https://xmatch.unistra.fr for CDS XMatch service status.")
    sys.exit(1)

result_df = pd.concat(all_results, ignore_index=True)

print(f"\nXMatch returned columns: {list(result_df.columns)}")

# XMatch typically returns 'main_type' (scalars) alongside 'otype' (lists).
# Use main_type if available, otherwise try otype.
if 'main_type' in result_df.columns:
    result_df['otype'] = result_df['main_type'].astype(str)
elif 'otype' in result_df.columns:
    # otype may be a multi-valued column; flatten it
    def _flatten(val):
        if isinstance(val, (list, tuple, np.ndarray)):
            return str(val.flat[0]) if len(val) > 0 else 'Unknown'
        return str(val)
    result_df['otype'] = result_df['otype'].apply(_flatten)
else:
    result_df['otype'] = 'MATCHED'

# Drop any leftover helper columns except src_id, angDist and otype
keep_cols = ['src_id', 'angDist', 'otype']
result_df = result_df[keep_cols]

result_df.to_csv("/home/jortecal/GitHub/eRosita/RefereeChecks/Source_list/boxlist_crossmatch.csv", index=False)

# Keep only one row per source: explicitly sort by angDist first so the
# CLOSEST SIMBAD match is guaranteed to be the one kept, regardless of the
# order XMatch happened to return rows in.
result_df_sorted = result_df.sort_values(['src_id', 'angDist'])
unique_df = result_df_sorted.drop_duplicates("src_id", keep='first')

matched_ids = set(unique_df["src_id"])
unmatched = df[~df["src_id"].isin(matched_ids)]
print(f"\nUnidentified sources (no SIMBAD match within {MATCH_RADIUS_ARCSEC:.0f}\"):"
      f" {len(unmatched)} / {len(df)}")

print("\nBreakdown of identified object types (one row per source, closest match kept):")
vc = unique_df["otype"].value_counts()
print(vc)

# Also save the full breakdown to a CSV for reference
vc.to_csv("/home/jortecal/GitHub/eRosita/RefereeChecks/Source_list/boxlist_otype_breakdown.csv", header=["count"])
print("\nWrote full breakdown to boxlist_otype_breakdown.csv")