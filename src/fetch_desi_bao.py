#!/usr/bin/env python3
"""
fetch_desi_bao.py — Download DESI DR1 BAO data for DSI analysis.

Tries three sources in order:
  1. DESI public server (full correlation function)
  2. Zenodo deposits (correlation function from papers)
  3. CosmoSIS compressed BAO (distance measurements only)

Usage:
    python src/fetch_desi_bao.py
"""

import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

# ── Configuration ──────────────────────────────────────────
DATA_DIR = Path("data/desi")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# DESI DR1 public data base URL
DESI_BASE = "https://data.desi.lbl.gov/public/dr1"

# Clustering products VAC
CLUSTERING_BASE = f"{DESI_BASE}/vac/full-shape-bao-clustering"

# Tracer types and redshift bins (from DESI 2024 II)
# Reference: arXiv:2411.12020
TRACERS = {
    "LRG": {
        "zbins": ["0.4-0.6", "0.6-0.8", "0.8-1.1"],
        "label": "Luminous Red Galaxies",
    },
    "ELG_LOPnotqso": {
        "zbins": ["0.8-1.1", "1.1-1.6"],
        "label": "Emission Line Galaxies",
    },
    "QSO": {
        "zbins": ["0.8-2.1"],
        "label": "Quasars",
    },
    "BGS_BRIGHT-21.5": {
        "zbins": ["0.1-0.4"],
        "label": "Bright Galaxy Survey",
    },
}

# Preferred tracer/bin for BAO analysis (highest S/N BAO detection)
PREFERRED_TRACER = "LRG"
PREFERRED_ZBIN = "0.6-0.8"


def download_file(url, dest, label=""):
    """Download a file, return True on success."""
    print(f"  Trying: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fibonacci-cosmology/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            with open(dest, "wb") as f:
                f.write(data)
            size_kb = len(data) / 1024
            print(f"  ✅ Downloaded {label}: {dest.name} ({size_kb:.1f} KB)")
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  ❌ Failed: {e}")
        return False


def try_desi_public_server():
    """
    Attempt to download correlation functions from the DESI public server.

    The directory structure for clustering products is:
      /public/dr1/vac/full-shape-bao-clustering/v1/
        {tracer}/xi/post/
          xiell_..._{tracer}_GCcomb_{zbin}_default_FKP_lin4_...dat

    References:
      - DESI 2024 II: arXiv:2411.12020
      - https://data.desi.lbl.gov/doc/releases/dr1/vac/full-shape-bao-clustering/
    """
    print("\n" + "=" * 60)
    print("SOURCE 1: DESI Public Server")
    print("=" * 60)

    # Try multiple possible URL patterns (the exact structure varies)
    version_dirs = ["v1.2", "v1.1", "v1", ""]
    recon_types = ["post", "prerecon", ""]
    cap_types = ["GCcomb", "NGC", "SGC"]

    tracer = PREFERRED_TRACER
    zbin = PREFERRED_ZBIN
    success = False

    for version in version_dirs:
        if success:
            break

        base = f"{CLUSTERING_BASE}/{version}" if version else CLUSTERING_BASE

        # Try to find the correlation function file
        for recon in recon_types:
            if success:
                break

            for cap in cap_types:
                if success:
                    break

                # Several common filename patterns
                patterns = [
                    f"xiell_{tracer}_{cap}_z{zbin}_default_FKP_lin4_s20-200_{"post_" if recon == "post" else ""}recon.dat",
                    f"xi024_{tracer}_{cap}_z{zbin}_default_FKP_lin4_njack0_nran18_split20.txt",
                    f"xiell_{tracer}_{cap}_z{zbin}.txt",
                ]

                # Try subdirectory patterns
                subdirs = [
                    f"{tracer}/xi/{recon}/" if recon else f"{tracer}/xi/",
                    f"correlation_function/{tracer}/",
                    f"{tracer}/",
                    "",
                ]

                for subdir in subdirs:
                    if success:
                        break
                    for pattern in patterns:
                        url = f"{base}/{subdir}{pattern}"
                        dest = DATA_DIR / f"xi_{tracer}_{cap}_z{zbin}.dat"
                        if download_file(url, dest, f"{tracer} z={zbin} ξ(s)"):
                            success = True
                            break

    if not success:
        # Try listing the directory to see what's actually there
        print("\n  Attempting to list directory contents...")
        list_url = f"{CLUSTERING_BASE}/"
        try:
            req = urllib.request.Request(list_url, headers={"User-Agent": "fibonacci-cosmology/1.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode("utf-8", errors="replace")
                # Extract links from directory listing
                import re
                links = re.findall(r'href="([^"]+)"', html)
                links = [l for l in links if not l.startswith("?") and not l.startswith("/")]
                if links:
                    print(f"  Found {len(links)} items in clustering directory:")
                    for link in links[:20]:
                        print(f"    {link}")
                    print("  → Please navigate manually and update the URL patterns.")
                else:
                    print("  Directory listing returned no useful links.")
        except Exception as e:
            print(f"  Could not list directory: {e}")

    return success


def try_zenodo():
    """
    Download BAO data from Zenodo deposits associated with DESI papers.

    Zenodo record 16883053 contains multi-survey BAO compilation
    including DESI DR1 correlation functions.
    """
    print("\n" + "=" * 60)
    print("SOURCE 2: Zenodo Deposits")
    print("=" * 60)

    zenodo_urls = [
        # Multi-survey BAO compilation (ref 58)
        {
            "url": "https://zenodo.org/records/16883053/files/desi_dr1_lrg_xi_monopole.csv",
            "dest": "xi_LRG_z0.6-0.8_zenodo.csv",
            "label": "Zenodo BAO compilation - LRG xi",
        },
        # DESI H0 measurement supplement (ref 59)
        {
            "url": "https://zenodo.org/records/17686137/files/xi_post_recon_LRG_z0.6-0.8.dat",
            "dest": "xi_LRG_z0.6-0.8_zenodo_h0.dat",
            "label": "Zenodo H0 paper - LRG xi",
        },
    ]

    for item in zenodo_urls:
        dest = DATA_DIR / item["dest"]
        if download_file(item["url"], dest, item["label"]):
            return True

    # The Zenodo filenames may not match exactly. Try the main record page.
    print("\n  Zenodo exact filenames may differ.")
    print("  Manual download: https://zenodo.org/records/16883053")
    print("  Look for files containing 'lrg' and 'xi' or 'correlation'")
    return False


def try_compressed_bao():
    """
    Download compressed BAO distance measurements from CosmoSIS or
    construct from the published DESI 2024 VI paper values.

    These are D_M/r_d, D_H/r_d measurements at 7 effective redshifts.
    Reference: arXiv:2404.03002 (DESI 2024 VI), Tables 1-2.
    """
    print("\n" + "=" * 60)
    print("SOURCE 3: Compressed BAO (DESI 2024 VI, Table 1)")
    print("=" * 60)

    # ── Published values from DESI 2024 VI (arXiv:2404.03002) ──
    # These are the official BAO distance measurements.
    # z_eff, tracer, DM/rd, σ(DM/rd), DH/rd, σ(DH/rd)
    # Some bins report DV/rd instead of separate DM, DH.

    bao_data = {
        "description": "DESI DR1 BAO distance measurements (arXiv:2404.03002)",
        "reference": "DESI 2024 VI: Cosmological Constraints from BAO",
        "columns": {
            "z_eff": "effective redshift",
            "tracer": "galaxy tracer type",
            "DM_over_rd": "transverse comoving distance / sound horizon",
            "DH_over_rd": "Hubble distance / sound horizon (= c / (H(z) * rd))",
            "DV_over_rd": "volume-averaged distance / sound horizon",
            "sigma_DM": "1σ error on DM/rd",
            "sigma_DH": "1σ error on DH/rd",
            "sigma_DV": "1σ error on DV/rd",
        },
        "measurements": [
            {
                "z_eff": 0.295,
                "tracer": "BGS",
                "DV_over_rd": 7.93,
                "sigma_DV": 0.15,
            },
            {
                "z_eff": 0.510,
                "tracer": "LRG1",
                "DM_over_rd": 13.62,
                "sigma_DM": 0.25,
                "DH_over_rd": 20.98,
                "sigma_DH": 0.61,
            },
            {
                "z_eff": 0.706,
                "tracer": "LRG2",
                "DM_over_rd": 16.85,
                "sigma_DM": 0.32,
                "DH_over_rd": 20.08,
                "sigma_DH": 0.60,
            },
            {
                "z_eff": 0.930,
                "tracer": "LRG3+ELG1",
                "DM_over_rd": 21.71,
                "sigma_DM": 0.28,
                "DH_over_rd": 17.88,
                "sigma_DH": 0.35,
            },
            {
                "z_eff": 1.317,
                "tracer": "ELG2",
                "DM_over_rd": 27.79,
                "sigma_DM": 0.69,
                "DH_over_rd": 13.82,
                "sigma_DH": 0.42,
            },
            {
                "z_eff": 1.491,
                "tracer": "QSO",
                "DV_over_rd": 26.07,
                "sigma_DV": 0.67,
            },
            {
                "z_eff": 2.330,
                "tracer": "Lya",
                "DM_over_rd": 39.71,
                "sigma_DM": 0.94,
                "DH_over_rd": 8.52,
                "sigma_DH": 0.17,
            },
        ],
        # Correlation coefficients between DM and DH in the same bin
        # (approximate, from DESI 2024 VI Table 2)
        "correlations": {
            "LRG1": -0.42,
            "LRG2": -0.37,
            "LRG3+ELG1": -0.44,
            "ELG2": -0.39,
            "Lya": -0.45,
        },
    }

    # Save as JSON
    json_file = DATA_DIR / "desi_dr1_bao_distances.json"
    with open(json_file, "w") as f:
        json.dump(bao_data, f, indent=2)
    print(f"  ✅ Saved: {json_file}")

    # Also save as simple text table for quick inspection
    txt_file = DATA_DIR / "desi_dr1_bao_distances.txt"
    with open(txt_file, "w") as f:
        f.write("# DESI DR1 BAO distance measurements\n")
        f.write("# Reference: arXiv:2404.03002 (DESI 2024 VI)\n")
        f.write("# z_eff  tracer         DM/rd  σ(DM)  DH/rd  σ(DH)  DV/rd  σ(DV)\n")
        for m in bao_data["measurements"]:
            f.write(f"{m['z_eff']:6.3f}  {m['tracer']:<14s} "
                    f"{m.get('DM_over_rd', 0):6.2f}  "
                    f"{m.get('sigma_DM', 0):5.2f}  "
                    f"{m.get('DH_over_rd', 0):6.2f}  "
                    f"{m.get('sigma_DH', 0):5.2f}  "
                    f"{m.get('DV_over_rd', 0):6.2f}  "
                    f"{m.get('sigma_DV', 0):5.2f}\n")
    print(f"  ✅ Saved: {txt_file}")

    # ── Also create a numpy-friendly version ──
    # Extract DV/rd for ALL bins (compute DV from DM, DH where needed)
    # DV(z) = [z * DM(z)^2 * DH(z)]^(1/3)
    z_arr = []
    dv_arr = []
    dv_err = []

    for m in bao_data["measurements"]:
        z = m["z_eff"]
        z_arr.append(z)

        if "DV_over_rd" in m and m["DV_over_rd"] > 0:
            dv_arr.append(m["DV_over_rd"])
            dv_err.append(m["sigma_DV"])
        else:
            # Compute DV from DM and DH
            dm = m["DM_over_rd"]
            dh = m["DH_over_rd"]
            dv = (z * dm**2 * dh) ** (1.0 / 3.0)
            # Error propagation (ignoring correlation for rough estimate)
            ddv_ddm = (2.0 / 3.0) * dv / dm
            ddv_ddh = (1.0 / 3.0) * dv / dh
            sig = np.sqrt((ddv_ddm * m["sigma_DM"])**2 +
                          (ddv_ddh * m["sigma_DH"])**2)
            dv_arr.append(dv)
            dv_err.append(sig)

    np_file = DATA_DIR / "desi_dr1_dv_over_rd.txt"
    header = ("# DESI DR1: DV/rd vs z_eff\n"
              "# Derived from arXiv:2404.03002\n"
              "# z_eff    DV/rd    sigma(DV/rd)\n")
    with open(np_file, "w") as f:
        f.write(header)
        for z, dv, sig in zip(z_arr, dv_arr, dv_err):
            f.write(f"{z:.4f}  {dv:.4f}  {sig:.4f}\n")
    print(f"  ✅ Saved: {np_file}")

    return True


def try_cosmosis_github():
    """
    Try downloading DESI BAO data from CosmoSIS standard library on GitHub.
    """
    print("\n  Trying CosmoSIS GitHub for data vectors...")

    # The CosmoSIS DESI DR1 module contains the data vectors
    cosmosis_urls = [
        "https://raw.githubusercontent.com/joezuntz/cosmosis-standard-library/main/likelihood/desi_dr1/desi_bao_data.txt",
        "https://raw.githubusercontent.com/joezuntz/cosmosis-standard-library/main/likelihood/desi_dr1/desi_bao_cov.txt",
        "https://raw.githubusercontent.com/joezuntz/cosmosis-standard-library/main/likelihood/desi_dr1_arxiv/desi_bao_data.txt",
        "https://raw.githubusercontent.com/joezuntz/cosmosis-standard-library/main/likelihood/desi_dr1_arxiv/desi_bao_cov.txt",
    ]

    for url in cosmosis_urls:
        fname = url.split("/")[-1]
        # Prefix with cosmosis_ to avoid name conflicts
        dest = DATA_DIR / f"cosmosis_{fname}"
        download_file(url, dest, f"CosmoSIS {fname}")

    # Check if we got anything
    cosmosis_files = list(DATA_DIR.glob("cosmosis_*.txt"))
    if cosmosis_files:
        print(f"  ✅ Got {len(cosmosis_files)} CosmoSIS files")
        return True
    return False


def summarize():
    """Print summary of what's in the data directory."""
    print("\n" + "=" * 60)
    print("SUMMARY: Contents of data/desi/")
    print("=" * 60)

    files = sorted(DATA_DIR.iterdir())
    if not files or (len(files) == 1 and files[0].name == "README.md"):
        print("  ⚠️  No data files found!")
        print("  Manual download options:")
        print("    1. https://data.desi.lbl.gov/public/dr1/vac/full-shape-bao-clustering/")
        print("    2. https://zenodo.org/records/16883053")
        print("    3. Re-run this script (it saves published values as fallback)")
        return

    total_size = 0
    for f in files:
        if f.is_file() and f.name != "README.md":
            size = f.stat().st_size
            total_size += size
            print(f"  {f.name:<50s} {size/1024:8.1f} KB")

    print(f"\n  Total: {total_size/1024:.1f} KB in {len(files)} files")

    # Check what tests we can run
    has_xi = any("xi" in f.name.lower() for f in files)
    has_dv = any("dv" in f.name.lower() or "distance" in f.name.lower() for f in files)
    has_cosmosis = any("cosmosis" in f.name.lower() for f in files)

    print("\n  Available analyses:")
    if has_xi:
        print("    ✅ Full ξ(r) → Lomb-Scargle in ln(r) space")
    if has_dv or has_cosmosis:
        print("    ✅ Compressed BAO → log-periodic test in ln(z) space")
    if not has_xi:
        print("    ⚠️  No full ξ(r) found — only compressed BAO available")
        print("       (This is still useful: test DV/rd for log-periodic deviations from ΛCDM)")


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("DESI DR1 BAO Data Fetcher")
    print("=" * 60)

    # Source 1: Try DESI public server for full correlation function
    got_xi = try_desi_public_server()

    # Source 2: Try Zenodo deposits
    if not got_xi:
        got_xi = try_zenodo()

    # Source 3: Always save compressed BAO (from published paper values)
    # This is guaranteed to work — no download needed
    got_compressed = try_compressed_bao()

    # Source 4: Try CosmoSIS for official data vectors
    got_cosmosis = try_cosmosis_github()

    # Summary
    summarize()

    if not got_xi:
        print("\n" + "-" * 60)
        print("NOTE: Could not download full ξ(r) correlation function.")
        print("The compressed BAO data (DV/rd at 7 redshifts) is saved and")
        print("sufficient for a log-periodic test in ln(z) space.")
        print()
        print("To get the full ξ(r), try manually browsing:")
        print("  https://data.desi.lbl.gov/public/dr1/vac/full-shape-bao-clustering/")
        print("and look for files containing 'xi' and 'LRG' and 'z0.6-0.8'.")
        print("-" * 60)
