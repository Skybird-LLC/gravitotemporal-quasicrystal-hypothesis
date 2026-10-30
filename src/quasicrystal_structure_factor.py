"""
quasicrystal_structure_factor.py
================================
Computes S(k) for a 3D icosahedral quasicrystal (Ammann-Kramer tiling)
via cut-and-project from Z^6.

Generates testable predictions for φ-structured spacetime.

Requirements: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product as iproduct
from scipy.signal import find_peaks

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════
PHI    = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)

# ═══════════════════════════════════════════════════════════════
# 1. ICOSAHEDRAL STAR VECTORS
# ═══════════════════════════════════════════════════════════════
# Six vectors to vertices of a regular icosahedron (one per antipodal pair).
# Icosahedron vertices: (0, ±1, ±φ) and cyclic permutations.
raw_vectors = np.array([
    [ 0,  1,  PHI],
    [ 0, -1,  PHI],
    [ 1,  PHI, 0 ],
    [-1,  PHI, 0 ],
    [ PHI, 0,  1 ],
    [-PHI, 0,  1 ],
], dtype=np.float64)

# Normalize to unit vectors — these form a tight frame: Σ v⊗v = 2·I₃
e_par = raw_vectors / np.linalg.norm(raw_vectors, axis=1, keepdims=True)

# Perpendicular-space basis: orthogonal complement in R^6 via SVD
U_full, _, _ = np.linalg.svd(e_par, full_matrices=True)
e_perp = U_full[:, 3:]   # (6, 3) — orthonormal basis of E_perp

# Sanity checks
print("Tight-frame check (should be 2·I₃):")
print(np.round(e_par.T @ e_par, 6))
print(f"Orthogonality |e_par^T · e_perp|_max = "
      f"{np.max(np.abs(e_par.T @ e_perp)):.2e}")

# ═══════════════════════════════════════════════════════════════
# 2. GENERATE RECIPROCAL LATTICE VECTORS
# ═══════════════════════════════════════════════════════════════
H_MAX = 4  # |h_i| ≤ H_MAX.  4 → 531K vectors, 5 → 1.77M (slower but finer)
print(f"\nGenerating lattice vectors with |h_i| ≤ {H_MAX} ...")
H = np.array(list(iproduct(range(-H_MAX, H_MAX + 1), repeat=6)),
             dtype=np.float64)
H = H[np.any(H != 0, axis=1)]   # remove the origin
print(f"  {len(H):,} non-zero vectors")

# Project into physical (parallel) and internal (perpendicular) space
G_par  = H @ e_par     # (N, 3) — physical-space wavevectors
G_perp = H @ e_perp    # (N, 3) — perp-space components

k_par  = np.linalg.norm(G_par,  axis=1)
k_perp = np.linalg.norm(G_perp, axis=1)

# ═══════════════════════════════════════════════════════════════
# 3. WINDOW FUNCTION → BRAGG PEAK INTENSITIES
# ═══════════════════════════════════════════════════════════════
R_W = 1.0   # acceptance-domain radius (perp-space units)

def sphere_ft(k, R):
    """Normalized FT of indicator function of a 3D ball of radius R."""
    x  = k * R
    out = np.ones_like(x)
    m   = x > 1e-12
    xm  = x[m]
    out[m] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / xm**3
    return out

intensity = sphere_ft(k_perp, R_W) ** 2

# ═══════════════════════════════════════════════════════════════
# 4. RADIALLY AVERAGED S(k)
# ═══════════════════════════════════════════════════════════════
K_MAX  = 8.0
N_BINS = 4000
k_edges = np.linspace(0, K_MAX, N_BINS + 1)
k_cen   = 0.5 * (k_edges[:-1] + k_edges[1:])

S_k, _ = np.histogram(k_par, bins=k_edges, weights=intensity)

# ═══════════════════════════════════════════════════════════════
# 5. DIRECTIONAL S(k) ALONG A 5-FOLD AXIS
# ═══════════════════════════════════════════════════════════════
axis5  = e_par[0]                           # first star vector = a 5-fold axis
proj5  = G_par @ axis5                      # scalar projection
cos_a  = proj5 / (k_par + 1e-30)

cone   = (np.abs(cos_a) > np.cos(np.radians(5))) & (k_par > 0.01)
k5, I5 = np.abs(proj5[cone]), intensity[cone]

S_5f, _ = np.histogram(k5, bins=k_edges, weights=I5)

# ═══════════════════════════════════════════════════════════════
# 6. PEAK ANALYSIS — φ-RATIO COUNTING
# ═══════════════════════════════════════════════════════════════
pk_idx, _ = find_peaks(S_k, height=0.005 * S_k.max(),
                       distance=5, prominence=0.002 * S_k.max())
pk_k = k_cen[pk_idx]
pk_I = S_k[pk_idx]
order = np.argsort(pk_I)[::-1]
pk_k, pk_I = pk_k[order], pk_I[order]

# Pairwise ratios among top N peaks
N_TOP  = min(30, len(pk_k))
ratios = []
for i in range(N_TOP):
    for j in range(i + 1, N_TOP):
        if pk_k[j] > 0.05:
            r = pk_k[i] / pk_k[j]
            if r > 1:
                ratios.append(r)
ratios = np.array(ratios)

print("\n── φ-ratio hits among top-30 peak pairs (±3%) ──")
print(f"{'Power':>7s}  {'Target':>8s}  {'Hits':>6s}")
for n in range(1, 6):
    tgt  = PHI ** n
    hits = np.sum(np.abs(ratios - tgt) / tgt < 0.03)
    print(f"  φ^{n}     {tgt:8.4f}  {hits:6d}")

# ═══════════════════════════════════════════════════════════════
# 7. FOUR-PANEL FIGURE
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('3D Icosahedral Quasicrystal — Structure Factor & φ-Scaling',
             fontsize=14, fontweight='bold')

# ── (a) Radial S(k) ──
ax = axes[0, 0]
ax.plot(k_cen, S_k, 'k-', lw=0.5)
if len(pk_k):
    k0 = pk_k[0]
    for n in range(-2, 7):
        kn = k0 * PHI ** n
        if 0 < kn < K_MAX:
            ax.axvline(kn, color='goldenrod', alpha=0.35, ls='--', lw=0.7)
    ax.axvline(k0, color='crimson', alpha=0.8, lw=1.2,
               label=f'$k_0 = {k0:.3f}$')
ax.set(xlabel=r'$k$  (quasilattice units)', ylabel=r'$S(k)$',
       title=r'(a)  Radially averaged $S(k)$')
ax.legend(fontsize=9)

# ── (b) 5-fold axis ──
ax = axes[0, 1]
ax.plot(k_cen, S_5f, 'steelblue', lw=0.6)
pk5_idx, _ = find_peaks(S_5f, height=0.01 * max(S_5f.max(), 1e-10),
                        distance=5)
if len(pk5_idx):
    top5 = k_cen[pk5_idx[np.argsort(S_5f[pk5_idx])[::-1]]]
    if len(top5):
        for n in range(-2, 7):
            kn = top5[0] * PHI ** n
            if 0 < kn < K_MAX:
                ax.axvline(kn, color='goldenrod', alpha=0.35, ls='--', lw=0.7)
ax.set(xlabel=r'$k$  (quasilattice units)',
       ylabel=r'$S(k)$  along 5-fold axis',
       title=r'(b)  5-fold axis — cleaner $\phi$-hierarchy')

# ── (c) log_φ scale ──
ax = axes[1, 0]
pos = k_cen > 0.05
lk  = np.log(k_cen[pos]) / LN_PHI
ax.plot(lk, S_k[pos], 'k-', lw=0.5)
for n in range(-4, 12):
    ax.axvline(n, color='goldenrod', alpha=0.15, ls=':', lw=0.6)
ax.set(xlabel=r'$\log_\phi\,k$', ylabel=r'$S(k)$',
       title=r'(c)  $\log_\phi$ scale — periodicity $\Leftrightarrow$ DSI')

# ── (d) Pairwise ratio histogram ──
ax = axes[1, 1]
if len(ratios):
    ax.hist(ratios, bins=120, range=(1, 5), color='steelblue',
            edgecolor='none', alpha=0.7)
    for n in range(1, 4):
        ax.axvline(PHI ** n, color='crimson', ls='--', lw=1.2,
                   label=rf'$\phi^{n} = {PHI**n:.3f}$')
    ax.set(xlabel='Peak-position ratio', ylabel='Count',
           title=r'(d)  Pairwise peak ratios — $\phi^n$ clustering')
    ax.legend(fontsize=8)

plt.tight_layout()
outpath = 'results/figures/icosahedral_quasicrystal_Sk.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Figure saved: {outpath}")

# ═══════════════════════════════════════════════════════════════
# 8. EXPORTABLE DSI PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def dsi_power_modulation(k, k0, epsilon, phase=0.0):
    """
    Log-periodic modulation of the matter power spectrum.

        δP(k)/P(k) = ε · cos(2π log_φ(k/k₀) + φ₀)

    Parameters
    ----------
    k       : array  — wavenumber [h/Mpc]
    k0      : float  — fundamental DSI scale [h/Mpc]
    epsilon : float  — modulation amplitude
    phase   : float  — phase offset [radians]
    """
    return epsilon * np.cos(2 * np.pi * np.log(k / k0) / LN_PHI + phase)


def dsi_correlation_modulation(r, r0, epsilon, phase=0.0):
    """
    Log-periodic modulation of the 2-point correlation function.

        δξ(r)/ξ(r) = ε · cos(2π log_φ(r/r₀) + ψ₀)
    """
    return epsilon * np.cos(2 * np.pi * np.log(r / r0) / LN_PHI + phase)


# Quick demo: BAO-range prediction
print("\n── Log-periodic prediction for BAO separations ──")
r_bao   = np.array([50, 80, 100, 130, 150])   # Mpc/h
r0_demo = 100.0   # Mpc/h — to be fit
eps_demo = 0.02

print(f"  r₀ = {r0_demo} Mpc/h,  ε = {eps_demo}")
for r in r_bao:
    mod = dsi_correlation_modulation(r, r0_demo, eps_demo)
    print(f"    r = {r:4d} Mpc/h  →  δξ/ξ = {mod:+.4f}")
