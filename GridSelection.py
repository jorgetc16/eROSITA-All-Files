#!/usr/bin/env python3
"""
grid_from_widths.py

Reads a two-column whitespace table:
    energy_keV   fwhm_eV

Builds a resolution-aware energy grid between Emin and Emax,
stepping by dE = f_fwhm * FWHM(E), where FWHM(E) is interpolated
from the input table (converted from eV -> keV).

Prints a space-separated list of energies to stdout.

Usage:
    ./grid_from_widths.py widths_table.txt --emin 2.0 --emax 9.0 --f_fwhm 0.5 [--mode full|sampled] [--n_sample N]

Notes:
 - Requires numpy.
 - If --mode sampled is used, exactly N energies are selected evenly from the generated full grid.
"""
import argparse
import numpy as np
import sys
from pathlib import Path

def read_table(path):
    arr = np.loadtxt(path,delimiter=",")
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError("Widths table must have two columns: energy_keV fwhm_eV")
        arr = arr.reshape((1,2))
    E_tab = arr[:,0].astype(float)         # keV
    fwhm_tab_eV = arr[:,1].astype(float)   # eV
    return E_tab, fwhm_tab_eV

def build_grid(Emin, Emax, f_fwhm, E_tab, fwhm_tab_eV, nmax=10000):
    # convert FWHM to keV
    fwhm_tab = (fwhm_tab_eV / 1000.0)
    # build a linear interpolant for FWHM(E)
    def fwhm_interp(E):
        # numpy.interp works for scalar or array
        return np.interp(E, E_tab, fwhm_tab, left=fwhm_tab[0], right=fwhm_tab[-1])
    Egrid = []
    E = Emin
    count = 0
    while E <= Emax and count < nmax:
        fwhm_local = float(fwhm_interp(E))
        if fwhm_local <= 0:
            fwhm_local = 1e-6
        dE = max(1e-6, f_fwhm * fwhm_local)
        Egrid.append(E)
        E = E + dE
        count += 1
    if count >= nmax:
        print("# WARNING: reached nmax in build_grid; increase nmax", file=sys.stderr)
    return np.array(Egrid), fwhm_interp

def effective_nind(Egrid, fwhm_interp):
    E = Egrid
    fwhm = fwhm_interp(E)
    fwhm = np.maximum(fwhm, 1e-8)
    integral = np.trapz(1.0 / fwhm, E)
    return integral

def main():
    p = argparse.ArgumentParser()
    p.add_argument("widths_table", help="two-col table: energy_keV  fwhm_eV")
    p.add_argument("--emin", type=float, default=2.0)
    p.add_argument("--emax", type=float, default=9.0)
    p.add_argument("--f_fwhm", type=float, default=0.5,
                   help="step size as fraction of local FWHM (e.g. 0.5)")
    p.add_argument("--mode", choices=("full","sampled"), default="full")
    p.add_argument("--n_sample", type=int, default=200)
    p.add_argument("--nmax", type=int, default=10000)
    args = p.parse_args()

    pfile = Path(args.widths_table)
    if not pfile.exists():
        print("ERROR: widths table not found:", args.widths_table, file=sys.stderr)
        sys.exit(2)

    E_tab, fwhm_tab = read_table(args.widths_table)
    Egrid, fwhm_interp = build_grid(args.emin, args.emax, args.f_fwhm, E_tab, fwhm_tab, nmax=args.nmax)
    nind = effective_nind(Egrid, fwhm_interp)

    # diagnostics to stderr
    print("# Generated grid from {:.3f} to {:.3f} keV".format(args.emin, args.emax), file=sys.stderr)
    print("# f_fwhm = {:.3f}".format(args.f_fwhm), file=sys.stderr)
    print("# grid points = {}".format(len(Egrid)), file=sys.stderr)
    print("# approx independent trials (integral dE/FWHM) = {:.2f}".format(nind), file=sys.stderr)

    if args.mode == "full":
        out = Egrid
    else:
        N = int(args.n_sample)
        if N >= len(Egrid):
            out = Egrid
        else:
            idx = np.round(np.linspace(0, len(Egrid)-1, N)).astype(int)
            out = Egrid[idx]

    # print energies to stdout (space separated)
    sys.stdout.write(" ".join(["{:.6f}".format(e) for e in out]))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
