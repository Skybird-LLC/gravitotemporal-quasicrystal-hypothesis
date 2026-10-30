"""
LIGO DSI Validation Suite
==========================
Three critical checks on the Test B >3σ result:
  1) H1-only vs L1-only stacking (detector independence)
  2) Spectral line masking (remove known instrumental features)
  3) Off-source null test (pure noise, no merger)

Run after ligo_dsi_search.py --strain completes.
"""

import numpy as np
from scipy.signal import lombscargle, welch
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ─── Constants ───
PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
K_PHI = 1.0 / LN_PHI
K_PHI_HARMONICS = [K_PHI * m for m in range(1, 5)]

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / 'data' / 'ligo_cache'
RESULTS_DIR = BASE_DIR / 'results'
FIG_DIR = RESULTS_DIR / 'figures'

# Known LIGO spectral lines to mask (Hz)
# Sources: LIGO T2100058, aLIGO noise budget documents
KNOWN_LINES_HZ = [
    # Power mains harmonics
    60, 120, 180, 240, 300, 360, 420, 480, 540, 600,
    660, 720, 780, 840, 900, 960,
    # Calibration lines (approximate, vary by observing run)
    35.9, 36.7, 331.9, 1083.7,
    # Violin modes (approximate fundamentals)
    500, 501, 502, 1000, 1001, 1002,
    # Known mechanical resonances
    9.0, 9.5, 13.0, 14.0, 21.0, 22.0,
]
LINE_HALF_WIDTH_HZ = 1.0  # mask ± this width around each line


def load_cached_ringdowns():
    """Load all cached ringdown files, tagged by detector."""
    ringdowns = {'H1': [], 'L1': []}

    for npz_file in sorted(CACHE_DIR.glob('*_ringdown.npz')):
        name = npz_file.stem  # e.g., "GW150914_H1_ringdown"
        parts = name.rsplit('_', 2)

        if len(parts) >= 3:
            event_name = parts[0]
            detector = parts[1]
        elif len(parts) == 2:
            event_name = parts[0]
            detector = 'H1' if 'H1' in name else 'L1' if 'L1' in name else None
        else:
            continue

        if detector not in ('H1', 'L1'):
            continue

        data = np.load(str(npz_file))
        ringdowns[detector].append({
            'event': event_name,
            'times': data['times'],
            'strain': data['strain'],
            'sample_rate': float(data['sample_rate']),
            'file': str(npz_file),
        })

    print(f"  Loaded ringdowns: H1={len(ringdowns['H1'])}, "
          f"L1={len(ringdowns['L1'])}")
    return ringdowns


def compute_psd_residual(strain, sample_rate, mask_lines=False):
    """
    Compute detrended PSD residual in ln(f) space.

    Parameters
    ----------
    mask_lines : bool
        If True, zero out known LIGO instrumental lines.
    """
    nperseg = min(len(strain), 256)
    if nperseg < 16:
        return None, None, None

    freqs, psd = welch(strain, fs=sample_rate, nperseg=nperseg,
                       noverlap=nperseg // 2)

    # Restrict to 30 Hz – Nyquist/4
    f_min, f_max = 30.0, sample_rate / 4
    mask = (freqs > f_min) & (freqs < f_max) & (psd > 0)

    if mask_lines:
        for line_f in KNOWN_LINES_HZ:
            line_mask = np.abs(freqs - line_f) < LINE_HALF_WIDTH_HZ
            mask = mask & ~line_mask

    if np.sum(mask) < 20:
        return None, None, None

    freqs_cut = freqs[mask]
    psd_cut = psd[mask]

    ln_f = np.log(freqs_cut)
    ln_psd = np.log10(psd_cut)

    # Detrend
    coeffs = np.polyfit(ln_f, ln_psd, 3)
    trend = np.polyval(coeffs, ln_f)
    residuals = ln_psd - trend

    return ln_f, residuals, freqs_cut


def run_periodogram(ln_f, residuals, n_freqs=3000, f_max=15.0):
    """Run Lomb-Scargle on the residual."""
    y = residuals - np.mean(residuals)
    freqs = np.linspace(0.1, f_max, n_freqs)
    angular = 2 * np.pi * freqs
    power = lombscargle(ln_f, y, angular, normalize=True)
    return freqs, power


def stack_periodograms(spectra_list, mask_lines=False):
    """Stack periodograms across a list of ringdown data dicts."""
    all_powers = []
    common_freqs = None

    for rd in spectra_list:
        ln_f, resid, _ = compute_psd_residual(
            rd['strain'], rd['sample_rate'], mask_lines=mask_lines
        )
        if ln_f is None:
            continue

        freqs, power = run_periodogram(ln_f, resid)
        all_powers.append(power)

        if common_freqs is None:
            common_freqs = freqs

    if len(all_powers) < 3:
        return None, None, 0

    stacked = np.mean(all_powers, axis=0)
    return common_freqs, stacked, len(all_powers)


def bootstrap_threshold(spectra_list, mask_lines=False, n_boot=3000, seed=42):
    """Compute significance thresholds via shuffled stacking."""
    rng = np.random.default_rng(seed)

    # Precompute all residuals
    all_ln_f = []
    all_resid = []
    for rd in spectra_list:
        ln_f, resid, _ = compute_psd_residual(
            rd['strain'], rd['sample_rate'], mask_lines=mask_lines
        )
        if ln_f is not None:
            all_ln_f.append(ln_f)
            all_resid.append(resid - np.mean(resid))

    if len(all_resid) < 3:
        return {}

    freqs_ls = np.linspace(0.1, 15.0, 3000)
    angular = 2 * np.pi * freqs_ls

    null_max = np.zeros(n_boot)
    for i in range(n_boot):
        stack = np.zeros(len(freqs_ls))
        for ln_f, resid in zip(all_ln_f, all_resid):
            y_shuf = rng.permutation(resid)
            p = lombscargle(ln_f, y_shuf, angular, normalize=True)
            stack += p
        stack /= len(all_resid)
        null_max[i] = stack.max()

    return {
        '1sigma': float(np.percentile(null_max, 68.27)),
        '2sigma': float(np.percentile(null_max, 95.45)),
        '3sigma': float(np.percentile(null_max, 99.73)),
    }


def read_harmonics(freqs, power, sig_levels):
    """Extract peak info near each DSI harmonic."""
    results = []
    for m in range(1, 5):
        k_target = m * K_PHI
        mask = np.abs(freqs - k_target) < 0.5
        if not np.any(mask):
            results.append({
                'm': m, 'k_target': k_target,
                'k_found': None, 'power': None, 'sigma': '—'
            })
            continue
        idx = np.where(mask)[0][np.argmax(power[mask])]
        k_found = freqs[idx]
        p_found = power[idx]

        if p_found > sig_levels.get('3sigma', 999):
            sigma = '>3σ'
        elif p_found > sig_levels.get('2sigma', 999):
            sigma = '>2σ'
        elif p_found > sig_levels.get('1sigma', 999):
            sigma = '>1σ'
        else:
            sigma = '<1σ'

        offset_pct = abs(k_found - k_target) / k_target * 100
        results.append({
            'm': m, 'k_target': round(k_target, 3),
            'k_found': round(float(k_found), 3),
            'power': round(float(p_found), 4),
            'sigma': sigma, 'offset_pct': round(offset_pct, 1),
        })
    return results


def fetch_off_source_strain(events, detector='H1', offset_sec=600,
                            duration=0.5):
    """
    Download strain from times AWAY from mergers (pure noise).
    Uses GPS time + offset (10 minutes after each event).
    """
    try:
        from gwpy.timeseries import TimeSeries
    except ImportError:
        print("    ⚠ gwpy not installed — cannot fetch off-source data")
        return []

    off_source = []
    for ev in events:
        gps = ev.get('gps')
        if gps is None:
            continue

        t_off = gps + offset_sec
        cache_file = CACHE_DIR / f"offsource_{ev['name']}_{detector}.npz"

        if cache_file.exists():
            data = np.load(str(cache_file))
            off_source.append({
                'event': ev['name'] + '_offsource',
                'times': data['times'],
                'strain': data['strain'],
                'sample_rate': float(data['sample_rate']),
            })
            continue

        try:
            strain = TimeSeries.fetch_open_data(
                detector, t_off, t_off + duration + 4
            )
            strain = strain.bandpass(30, 1024)
            t = strain.times.value - t_off
            mask = (t >= 0) & (t <= duration)
            h = strain.value[mask]
            sr = float(strain.sample_rate.value)

            np.savez(str(cache_file), times=t[mask], strain=h, sample_rate=sr)
            off_source.append({
                'event': ev['name'] + '_offsource',
                'times': t[mask],
                'strain': h,
                'sample_rate': sr,
            })
        except Exception:
            continue

    print(f"  Off-source segments loaded: {len(off_source)}")
    return off_source


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  LIGO DSI VALIDATION SUITE")
    print("  Checking Test B >3σ result for instrumental artifacts")
    print("=" * 70)

    # Load cached data
    print("\n[1/4] Loading cached ringdown data...")
    ringdowns = load_cached_ringdowns()

    all_spectra = ringdowns['H1'] + ringdowns['L1']
    if len(all_spectra) < 6:
        print("  ⚠ Too few cached ringdowns. Run ligo_dsi_search.py --strain first.")
        return

    # ─── VALIDATION 1: DETECTOR SPLIT ───
    print("\n" + "─" * 70)
    print("  VALIDATION 1: H1-only vs L1-only")
    print("─" * 70)

    results_v1 = {}
    for det in ['H1', 'L1']:
        if len(ringdowns[det]) < 3:
            print(f"  {det}: too few spectra ({len(ringdowns[det])})")
            continue

        print(f"\n  Stacking {det} ({len(ringdowns[det])} spectra)...")
        freqs, power, n = stack_periodograms(ringdowns[det])

        if freqs is None:
            continue

        print(f"  Bootstrap ({det})...", end='', flush=True)
        sig = bootstrap_threshold(ringdowns[det], n_boot=3000)
        print(" done.")
        print(f"  Thresholds: 1σ={sig['1sigma']:.4f}, "
              f"2σ={sig['2sigma']:.4f}, 3σ={sig['3sigma']:.4f}")

        harmonics = read_harmonics(freqs, power, sig)

        print(f"\n  {det} harmonics:")
        print(f"  {'m':>3} {'k_target':>8} {'k_found':>8} "
              f"{'power':>8} {'σ':>6} {'offset':>8}")
        for h in harmonics:
            if h['k_found'] is not None:
                print(f"  {h['m']:>3} {h['k_target']:>8} {h['k_found']:>8} "
                      f"{h['power']:>8} {h['sigma']:>6} "
                      f"{h['offset_pct']:>7.1f}%")

        results_v1[det] = {
            'n_spectra': n,
            'sig_levels': sig,
            'harmonics': harmonics,
            'freqs': freqs,
            'power': power,
        }

    # Check consistency
    if 'H1' in results_v1 and 'L1' in results_v1:
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │  DETECTOR CONSISTENCY CHECK              │")
        print("  └─────────────────────────────────────────┘")
        for m in range(1, 5):
            h1 = results_v1['H1']['harmonics'][m-1]
            l1 = results_v1['L1']['harmonics'][m-1]
            k_h1 = h1.get('k_found', '—')
            k_l1 = l1.get('k_found', '—')
            s_h1 = h1.get('sigma', '—')
            s_l1 = l1.get('sigma', '—')

            if isinstance(k_h1, float) and isinstance(k_l1, float):
                delta = abs(k_h1 - k_l1)
                consistent = "✓" if delta < 0.3 else "✗"
            else:
                delta = None
                consistent = "?"

            print(f"  m={m}: H1 k={k_h1} ({s_h1}) | "
                  f"L1 k={k_l1} ({s_l1}) | "
                  f"Δk={'%.3f' % delta if delta else '—'} {consistent}")

    # ─── VALIDATION 2: LINE MASKING ───
    print("\n" + "─" * 70)
    print("  VALIDATION 2: Known spectral line masking")
    print("─" * 70)

    print(f"  Masking {len(KNOWN_LINES_HZ)} known lines "
          f"(±{LINE_HALF_WIDTH_HZ} Hz each)...")
    freqs_masked, power_masked, n_masked = stack_periodograms(
        all_spectra, mask_lines=True
    )

    if freqs_masked is not None:
        print("  Bootstrap (masked)...", end='', flush=True)
        sig_masked = bootstrap_threshold(
            all_spectra, mask_lines=True, n_boot=3000
        )
        print(" done.")

        harmonics_masked = read_harmonics(freqs_masked, power_masked, sig_masked)

        print(f"\n  Masked harmonics:")
        print(f"  {'m':>3} {'k_target':>8} {'k_found':>8} "
              f"{'power':>8} {'σ':>6} {'offset':>8}")
        for h in harmonics_masked:
            if h['k_found'] is not None:
                print(f"  {h['m']:>3} {h['k_target']:>8} {h['k_found']:>8} "
                      f"{h['power']:>8} {h['sigma']:>6} "
                      f"{h['offset_pct']:>7.1f}%")
    else:
        harmonics_masked = []
        sig_masked = {}

    # ─── VALIDATION 3: OFF-SOURCE NULL ───
    print("\n" + "─" * 70)
    print("  VALIDATION 3: Off-source null test")
    print("─" * 70)

    # Need event list for GPS times
    from ligo_dsi_search import _fallback_catalog
    events = _fallback_catalog()

    off_h1 = fetch_off_source_strain(events, 'H1', offset_sec=600)
    off_l1 = fetch_off_source_strain(events, 'L1', offset_sec=600)
    off_all = off_h1 + off_l1

    if len(off_all) >= 6:
        print(f"  Stacking {len(off_all)} off-source segments...")
        freqs_off, power_off, n_off = stack_periodograms(off_all)

        if freqs_off is not None:
            print("  Bootstrap (off-source)...", end='', flush=True)
            sig_off = bootstrap_threshold(off_all, n_boot=3000)
            print(" done.")

            harmonics_off = read_harmonics(freqs_off, power_off, sig_off)

            print(f"\n  Off-source harmonics (should all be <1σ):")
            print(f"  {'m':>3} {'k_target':>8} {'k_found':>8} "
                  f"{'power':>8} {'σ':>6}")
            for h in harmonics_off:
                if h['k_found'] is not None:
                    print(f"  {h['m']:>3} {h['k_target']:>8} "
                          f"{h['k_found']:>8} {h['power']:>8} "
                          f"{h['sigma']:>6}")
    else:
        print("  ⚠ Could not fetch enough off-source data.")
        freqs_off, power_off, harmonics_off, sig_off = None, None, [], {}

    # ─── PLOTTING ───
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('LIGO DSI Validation Suite', fontsize=14, fontweight='bold')

    # Panel 1: H1 vs L1
    ax = axes[0, 0]
    for det, color in [('H1', 'steelblue'), ('L1', 'coral')]:
        if det in results_v1:
            ax.plot(results_v1[det]['freqs'], results_v1[det]['power'],
                    '-', color=color, lw=0.8, label=f'{det} ({results_v1[det]["n_spectra"]} spectra)')
    for m in range(1, 5):
        ax.axvline(m * K_PHI, color='gold', lw=1.5 if m == 1 else 0.8, alpha=0.7)
    ax.set_xlabel('k (cycles per unit ln f)')
    ax.set_ylabel('Stacked Lomb-Scargle Power')
    ax.set_title('Validation 1: H1 vs L1 (detector independence)')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.15)

    # Panel 2: Unmasked vs masked
    ax = axes[0, 1]
    if freqs_masked is not None:
        # Load original for comparison
        freqs_orig, power_orig, _ = stack_periodograms(all_spectra, mask_lines=False)
        if freqs_orig is not None:
            ax.plot(freqs_orig, power_orig, '-', color='gray', lw=0.8,
                    alpha=0.5, label='Original (all lines)')
        ax.plot(freqs_masked, power_masked, '-', color='darkgreen', lw=0.8,
                label='Lines masked')
        for m in range(1, 5):
            ax.axvline(m * K_PHI, color='gold', lw=1.5 if m == 1 else 0.8, alpha=0.7)
    ax.set_xlabel('k (cycles per unit ln f)')
    ax.set_ylabel('Stacked Lomb-Scargle Power')
    ax.set_title('Validation 2: Effect of line masking')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.15)

    # Panel 3: Off-source null
    ax = axes[1, 0]
    if freqs_off is not None:
        ax.plot(freqs_off, power_off, '-', color='dimgray', lw=0.8)
        for m in range(1, 5):
            ax.axvline(m * K_PHI, color='gold', lw=1.5 if m == 1 else 0.8, alpha=0.7)
        if sig_off:
            ax.axhline(sig_off['3sigma'], color='red', ls='--', lw=1,
                       label=f'3σ: {sig_off["3sigma"]:.4f}')
            ax.axhline(sig_off['1sigma'], color='green', ls='--', lw=1,
                       label=f'1σ: {sig_off["1sigma"]:.4f}')
        ax.legend(fontsize=9)
    ax.set_xlabel('k (cycles per unit ln f)')
    ax.set_ylabel('Stacked Lomb-Scargle Power')
    ax.set_title('Validation 3: Off-source (pure noise — no merger)')
    ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.15)

    # Panel 4: Decision matrix
    ax = axes[1, 1]
    ax.axis('off')

    decision_text = """
    DECISION MATRIX
    ═══════════════════════════════════════════════

    IF H1 ≈ L1 peaks:       → Detector-independent ✓
    IF H1 ≠ L1 peaks:       → Instrumental artifact ✗

    IF masking kills peaks:  → Spectral lines ✗
    IF peaks survive:        → Not from known lines ✓

    IF off-source has peaks: → Detector noise ✗
    IF off-source is flat:   → Signal needs merger ✓

    ═══════════════════════════════════════════════
    ALL THREE must pass for a credible detection.
    """
    ax.text(0.05, 0.95, decision_text, transform=ax.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig_path = FIG_DIR / 'ligo_dsi_validation.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    print(f"\n  Validation plot saved to {fig_path}")
    plt.show()

    # ─── SAVE ───
    save_data = {
        'validation_1_detector_split': {
            det: {
                'n_spectra': r['n_spectra'],
                'sig_levels': r['sig_levels'],
                'harmonics': r['harmonics'],
            }
            for det, r in results_v1.items()
        },
        'validation_2_line_masking': {
            'sig_levels': sig_masked,
            'harmonics': harmonics_masked if isinstance(
                harmonics_masked, list) else [],
        },
        'validation_3_off_source': {
            'sig_levels': sig_off if sig_off else {},
            'harmonics': harmonics_off if isinstance(
                harmonics_off, list) else [],
        },
    }

    out_file = RESULTS_DIR / 'ligo_dsi_validation_results.json'
    with open(str(out_file), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved to {out_file}")


if __name__ == '__main__':
    main()
