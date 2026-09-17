#!/usr/bin/env python
"""
analyze_like.py

Analyzes the 'like' column from boxlist_summary.csv, split by whether
a SIMBAD counterpart was found (boxlist_crossmatch.csv).

- Histograms of LIKE values (identified vs unidentified)
- Cumulative counts above key thresholds for each category
- Basic statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
csv_path = "/home/jortecal/GitHub/eRosita/RefereeChecks/Source_list/boxlist_summary.csv"
xmatch_path = "/home/jortecal/GitHub/eRosita/RefereeChecks/Source_list/boxlist_crossmatch.csv"

df = pd.read_csv(csv_path)

# Filter to combined band (id_band == 0) as done in Crossmatch_boxlist.py
if 'id_band' in df.columns:
    df = df[df['id_band'] == 0].copy()
    print(f"Filtered to id_band=0: {len(df)} sources")

df = df.reset_index(drop=True)
df.insert(0, "src_id", df.index)  # row index matches the src_id in crossmatch

# Load crossmatch results
df_xm = pd.read_csv(xmatch_path)
matched_ids = set(df_xm["src_id"].unique())

# Split: identified (in crossmatch) vs unidentified
mask_id = df["src_id"].isin(matched_ids)
like_id   = df.loc[mask_id, "like"].values
like_unid = df.loc[~mask_id, "like"].values

n_id   = len(like_id)
n_unid = len(like_unid)
n_tot  = n_id + n_unid

print(f"Identified   (SIMBAD match): {n_id:6d}  ({100*n_id/n_tot:.1f}%)")
print(f"Unidentified (no match)  : {n_unid:6d}  ({100*n_unid/n_tot:.1f}%)")
print(f"Total                    : {n_tot:6d}")
print()
for label, arr in [("Identified", like_id), ("Unidentified", like_unid), ("ALL", df["like"].values)]:
    print(f"{label:>14s}  min={arr.min():8.4f}  max={arr.max():8.4f}"
          f"  mean={arr.mean():8.4f}  median={np.median(arr):8.4f}")

# ---------------------------------------------------------------------------
# 2. Cumulative counts above thresholds
# ---------------------------------------------------------------------------
thresholds = [6, 8, 10, 12, 15, 20, 25, 30, 50, 100, 200]

print(f"\n{'LIKE >':>8s}  {'Identified':>10s} {'%':>7s}  {'Unidentified':>12s} {'%':>7s}  {'ALL':>8s} {'%':>7s}")
print("-" * 72)
for thr in thresholds:
    n_id_p   = np.sum(like_id   > thr)
    n_unid_p = np.sum(like_unid > thr)
    n_all_p  = n_id_p + n_unid_p
    print(f"{thr:8.1f}  {n_id_p:10d} {100*n_id_p/n_id:6.2f}%"
          f"  {n_unid_p:12d} {100*n_unid_p/n_unid:6.2f}%"
          f"  {n_all_p:8d} {100*n_all_p/n_tot:6.2f}%")

# ---------------------------------------------------------------------------
# 3. Histogram  (overlaid: identified vs unidentified)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = {'Identified': 'steelblue', 'Unidentified': 'darkorange'}
bins_lin = np.linspace(like_id.min(), like_id.max(), 70)

for ax, use_log, title_suffix in [(axes[0], False, "linear bins"),
                                   (axes[1], True,  "log bins")]:
    if use_log:
        bins = np.logspace(np.log10(max(min(like_id.min(), like_unid.min()), 1)),
                           np.log10(max(like_id.max(), like_unid.max())), 60)
    else:
        bins = bins_lin

    ax.hist(like_id,   bins=bins, color=colors['Identified'],
            edgecolor='white', alpha=0.7, label='Identified (SIMBAD)')
    ax.hist(like_unid, bins=bins, color=colors['Unidentified'],
            edgecolor='white', alpha=0.55, label='Unidentified')

    if use_log:
        ax.set_xscale('log')
    for thr in thresholds:
        ax.axvline(thr, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('LIKE')
    ax.set_ylabel('Number of sources')
    ax.set_title(f'LIKE distribution ({title_suffix})')
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig('like_histogram.pdf')
print("\nSaved like_histogram.pdf")

# ---------------------------------------------------------------------------
# 4. Cumulative distribution  (N with LIKE > x)
# ---------------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(8, 6))

for label, arr, ls in [('Identified', like_id, '-'),
                         ('Unidentified', like_unid, '--')]:
    sorted_like = np.sort(arr)[::-1]
    cum_N = np.arange(1, len(arr) + 1)
    ax.step(sorted_like, cum_N, where='pre', linewidth=2,
            linestyle=ls, label=label)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('LIKE threshold', fontsize=13)
ax.set_ylabel('Number of sources with LIKE > threshold', fontsize=13)
ax.set_title('Cumulative distribution')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

fig2.tight_layout()
fig2.savefig('like_cumulative.pdf')
print("Saved like_cumulative.pdf")

# ---------------------------------------------------------------------------
# 5. Identified fraction vs LIKE threshold
# ---------------------------------------------------------------------------
fig3, ax = plt.subplots(figsize=(7, 5))

thr_fine = np.linspace(0, 100, 200)
frac_id = []
n_above = []
for thr in thr_fine:
    n_above_id = np.sum(like_id > thr)
    n_above_unid = np.sum(like_unid > thr)
    tot = n_above_id + n_above_unid
    frac_id.append(n_above_id / tot if tot > 0 else np.nan)
    n_above.append(tot)

frac_id = np.array(frac_id)
n_above = np.array(n_above)

ax.plot(thr_fine, frac_id * 100, color='darkviolet', linewidth=2)
ax.set_xlabel('LIKE threshold', fontsize=13)
ax.set_ylabel('Identified fraction [%]', fontsize=13)
ax.set_title('Fraction of sources with SIMBAD counterpart\namong those with LIKE > threshold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

# Add a secondary axis showing N above threshold
ax2 = ax.twiny()
# Only keep points where N is meaningful
mask_n = n_above > 10
ax2.plot(n_above[mask_n], frac_id[mask_n] * 100, visible=False)  # dummy for scale
ax2.set_xscale('log')
ax2.set_xlabel('N with LIKE > threshold', fontsize=11, color='gray')

fig3.tight_layout()
fig3.savefig('like_identified_fraction.pdf')
print("Saved like_identified_fraction.pdf")

plt.show()
