"""
LIGO Discrete Scale Invariance (DSI) Search
=============================================
Searches for golden-ratio log-periodic signatures in gravitational wave data.

Two independent tests:
  A) QNM frequency ratio analysis across the GWTC catalog
  B) Ringdown power spectrum analysis in ln(f) space, stacked across events

Tests the prediction:
    f(φ) ≈ φ^{-n} · cos(2π ln(r) / ln(φ))

which should produce peaks at k_m = m/ln(φ) for m = 1, 2, 3, ...
in the Lomb-Scargle periodogram of residuals vs ln(f).

Requirements:
    pip install gwpy gwosc qnm numpy scipy matplotlib
"""

import numpy as np
from scipy.signal import lombscargle, welch
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import json
import os
import warnings
from pathlib import Path
from collections import defaultdict

# ══════════════════════════════════════════════════════════════
# 0. CONSTANTS
# ══════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
K_PHI = 1.0 / LN_PHI              # ≈ 2.078
K_PHI_HARMONICS = [K_PHI * m for m in range(1, 5)]

# Solar mass in seconds (geometric units: G = c = 1)
M_SUN_SEC = 4.925491025543576e-06  # M_sun * G / c^3

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'results'
FIG_DIR = RESULTS_DIR / 'figures'
CACHE_DIR = BASE_DIR / 'data' / 'ligo_cache'

for d in [RESULTS_DIR, FIG_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. FETCH GWTC CATALOG
# ══════════════════════════════════════════════════════════════

def fetch_catalog(min_snr=8.0, max_events=None):
    """
    Fetch BBH events from GWTC with parameter estimation results.

    Returns list of dicts with: name, mass_1, mass_2, final_mass,
    final_spin, luminosity_distance, network_snr, gps_time.
    """
    try:
        from gwosc import datasets as gwosc_ds
    except ImportError:
        print("  ⚠ gwosc package not installed. Using fallback catalog.")
        return _fallback_catalog()

    print("\n[1/6] Fetching GWTC catalog from GWOSC...")

    events = []

    # Modern gwosc API exposes GPS/event listing, but not PE posteriors.
    # We use it to discover available events and match onto local PE fallback table.
    fallback_by_name = {ev['name']: ev for ev in _fallback_catalog()}

    for cat_name in ['GWTC-1-confident', 'GWTC-2.1-confident', 'GWTC-3-confident']:
        try:
            event_names = gwosc_ds.find_datasets(type='events', catalog=cat_name)
            print(f"    {cat_name}: {len(event_names)} events")
        except Exception as exc:
            print(f"    Could not fetch {cat_name}: {exc}")
            continue

        for event_name in event_names:
            if event_name not in fallback_by_name:
                continue

            base = dict(fallback_by_name[event_name])
            base['catalog'] = cat_name

            # Prefer live GWOSC GPS when available
            try:
                gps_live = gwosc_ds.event_gps(event_name)
                if gps_live is not None:
                    base['gps'] = float(gps_live)
            except Exception:
                pass

            snr_val = base.get('snr', None)
            if snr_val is not None and snr_val < min_snr:
                continue

            events.append(base)

    if len(events) == 0:
        print("  ⚠ GWOSC reachable but no matched PE rows found; using fallback catalog.")
        return _fallback_catalog()

    # Deduplicate (same event may appear in multiple catalogs)
    seen = set()
    unique_events = []
    for ev in events:
        # Use GPS time rounded to nearest second as dedup key
        key = round(ev['gps'])
        if key not in seen:
            seen.add(key)
            unique_events.append(ev)

    # Sort by SNR (highest first)
    unique_events.sort(key=lambda x: x.get('snr', 0) or 0, reverse=True)

    if max_events:
        unique_events = unique_events[:max_events]

    print(f"    Total unique BBH events with PE results: {len(unique_events)}")

    return unique_events


def _fallback_catalog():
    """
    Hardcoded high-SNR events from GWTC-3 in case gwosc package fails.
    Source: GWTC-3 catalog paper Table I.
    """
    return [
        {'name': 'GW150914', 'gps': 1126259462.4,
         'final_mass': 62.0, 'final_spin': 0.69, 'snr': 24.4,
         'mass_1': 35.6, 'mass_2': 30.6},
        {'name': 'GW151226', 'gps': 1135136350.6,
         'final_mass': 20.5, 'final_spin': 0.74, 'snr': 13.1,
         'mass_1': 13.7, 'mass_2': 7.7},
        {'name': 'GW170104', 'gps': 1167559936.6,
         'final_mass': 48.9, 'final_spin': 0.66, 'snr': 13.0,
         'mass_1': 30.8, 'mass_2': 20.0},
        {'name': 'GW170814', 'gps': 1186741861.5,
         'final_mass': 53.2, 'final_spin': 0.72, 'snr': 18.3,
         'mass_1': 30.7, 'mass_2': 25.2},
        {'name': 'GW170823', 'gps': 1187529256.5,
         'final_mass': 65.4, 'final_spin': 0.72, 'snr': 11.5,
         'mass_1': 39.5, 'mass_2': 29.0},
        {'name': 'GW190412', 'gps': 1239082262.2,
         'final_mass': 37.0, 'final_spin': 0.67, 'snr': 19.1,
         'mass_1': 30.1, 'mass_2': 8.3},
        {'name': 'GW190519_153544', 'gps': 1242315362.4,
         'final_mass': 103.0, 'final_spin': 0.76, 'snr': 12.8,
         'mass_1': 66.0, 'mass_2': 40.0},
        {'name': 'GW190521', 'gps': 1242442967.4,
         'final_mass': 150.0, 'final_spin': 0.72, 'snr': 14.7,
         'mass_1': 85.0, 'mass_2': 66.0},
        {'name': 'GW190814', 'gps': 1249852257.0,
         'final_mass': 25.6, 'final_spin': 0.28, 'snr': 25.0,
         'mass_1': 23.2, 'mass_2': 2.6},
        {'name': 'GW190828_063405', 'gps': 1251009263.8,
         'final_mass': 55.0, 'final_spin': 0.68, 'snr': 16.3,
         'mass_1': 32.1, 'mass_2': 26.2},
        {'name': 'GW191109_010717', 'gps': 1257296855.2,
         'final_mass': 107.0, 'final_spin': 0.73, 'snr': 15.5,
         'mass_1': 65.0, 'mass_2': 47.0},
        {'name': 'GW200129_065458', 'gps': 1264316116.4,
         'final_mass': 59.0, 'final_spin': 0.68, 'snr': 26.8,
         'mass_1': 34.5, 'mass_2': 28.3},
        {'name': 'GW200225_060421', 'gps': 1266645879.4,
         'final_mass': 48.0, 'final_spin': 0.63, 'snr': 13.0,
         'mass_1': 28.3, 'mass_2': 21.7},
        {'name': 'GW200311_115853', 'gps': 1267963151.3,
         'final_mass': 51.0, 'final_spin': 0.67, 'snr': 17.8,
         'mass_1': 34.2, 'mass_2': 23.8},
    ]


# ══════════════════════════════════════════════════════════════
# 2. TEST A — QNM FREQUENCY RATIO ANALYSIS
# ══════════════════════════════════════════════════════════════

def compute_qnm_spectrum(final_mass_msun, final_spin, n_overtones=7):
    """
    Compute QNM frequencies for a Kerr black hole.

    Parameters
    ----------
    final_mass_msun : float
        Final mass in solar masses
    final_spin : float
        Dimensionless spin parameter (0 <= a < 1)
    n_overtones : int
        Number of overtones to compute (n = 0, 1, ..., n_overtones-1)

    Returns
    -------
    dict with 'frequencies_hz', 'damping_times_s', 'omega_complex'
    """
    try:
        import qnm
    except ImportError:
        return _qnm_fallback(final_mass_msun, final_spin, n_overtones)

    M_sec = final_mass_msun * M_SUN_SEC

    results = {
        'frequencies_hz': [],
        'damping_times_s': [],
        'quality_factors': [],
        'omega_complex': [],
    }

    # Dominant mode: (l, m) = (2, 2)
    for n in range(n_overtones):
        try:
            mode = qnm.modes_cache(s=-2, l=2, m=2, n=n)
            omega, _, _ = mode(a=final_spin)

            # Convert from dimensionless to physical units
            # omega is in units of 1/M (geometric units)
            f_hz = np.real(omega) / (2 * np.pi * M_sec)
            tau_s = -1.0 / (np.imag(omega) / M_sec)
            Q = np.pi * f_hz * tau_s

            results['frequencies_hz'].append(float(f_hz))
            results['damping_times_s'].append(float(tau_s))
            results['quality_factors'].append(float(Q))
            results['omega_complex'].append(complex(omega))

        except Exception as e:
            break

    results['n_modes'] = len(results['frequencies_hz'])
    return results


def _qnm_fallback(final_mass_msun, final_spin, n_overtones):
    """
    Approximate QNM frequencies using Berti et al. (2006) fitting formulas.
    These are accurate to ~1% for the (2,2) mode.
    """
    # Fitting coefficients for (l=2, m=2) fundamental and first overtone
    # From Berti, Cardoso, Will (2006), Table VIII
    a = final_spin
    M_sec = final_mass_msun * M_SUN_SEC

    # (2,2,0) fundamental mode
    f1_0 = 1.5251 - 1.1568 * (1 - a)**0.1292
    f2_0 = 0.7000
    q1_0 = 0.7000 + 1.4187 * (1 - a)**(-0.4990)

    f_220 = f1_0 / (2 * np.pi * M_sec)
    Q_220 = q1_0
    tau_220 = Q_220 / (np.pi * f_220)

    results = {
        'frequencies_hz': [float(f_220)],
        'damping_times_s': [float(tau_220)],
        'quality_factors': [float(Q_220)],
        'omega_complex': [complex(2*np.pi*f_220*M_sec, -1.0/(tau_220/M_sec))],
        'n_modes': 1,
    }

    # Approximate overtones by scaling (rough but sufficient for DSI search)
    for n in range(1, min(n_overtones, 4)):
        f_n = f_220 * (1 + 0.35 * n)  # overtone spacing ~35% per mode
        tau_n = tau_220 / (1 + 0.9 * n)
        Q_n = np.pi * f_n * tau_n
        results['frequencies_hz'].append(float(f_n))
        results['damping_times_s'].append(float(tau_n))
        results['quality_factors'].append(float(Q_n))
        results['n_modes'] = n + 1

    return results


def test_a_qnm_ratios(events):
    """
    TEST A: Search for φ-periodic structure in QNM frequency ratios.

    For each event, compute the (2,2) QNM overtone frequencies.
    Form frequency ratios f_n / f_0 for n = 1, 2, 3, ...
    Stack these ratios across all events.
    Search for clustering near powers of φ.
    """
    print("\n" + "=" * 70)
    print("  TEST A: QNM Frequency Ratio Analysis")
    print("=" * 70)

    all_ratios = []
    all_ln_ratios = []
    event_qnm_data = []

    for ev in events:
        mf = ev['final_mass']
        af = ev['final_spin']

        if mf is None or af is None:
            continue
        if af >= 0.998 or af < 0.0:
            continue

        qnm_data = compute_qnm_spectrum(mf, af, n_overtones=7)

        if qnm_data['n_modes'] < 2:
            continue

        freqs = np.array(qnm_data['frequencies_hz'])
        f0 = freqs[0]

        # Compute all pairwise ratios
        for i in range(1, len(freqs)):
            ratio = freqs[i] / f0
            all_ratios.append(ratio)
            all_ln_ratios.append(np.log(ratio))

        # Also compute consecutive ratios
        for i in range(1, len(freqs)):
            ratio_consec = freqs[i] / freqs[i-1]
            if ratio_consec > 1:
                all_ratios.append(ratio_consec)
                all_ln_ratios.append(np.log(ratio_consec))

        event_qnm_data.append({
            'name': ev['name'],
            'final_mass': mf,
            'final_spin': af,
            'frequencies_hz': qnm_data['frequencies_hz'],
            'quality_factors': qnm_data['quality_factors'],
            'n_modes': qnm_data['n_modes'],
        })

    print(f"\n  Events with QNM data: {len(event_qnm_data)}")
    print(f"  Total frequency ratios: {len(all_ratios)}")

    if len(all_ratios) < 10:
        print("  ⚠ Too few ratios for meaningful analysis.")
        return {'status': 'insufficient_data'}

    all_ratios = np.array(all_ratios)
    all_ln_ratios = np.array(all_ln_ratios)

    # Check if ratios cluster near φ powers
    phi_powers = [PHI**n for n in range(-3, 4)]  # φ^{-3} to φ^3
    ln_phi_multiples = [n * LN_PHI for n in range(-3, 4)]

    print(f"\n  Checking ratio clustering near φ powers:")
    print(f"  {'φ^n':>8} {'value':>10} {'nearest ratio':>15} {'deviation':>10}")
    print(f"  {'─' * 48}")

    for n in range(-3, 4):
        target = PHI ** n
        if target < 0.5 or target > 5.0:
            continue
        nearest = all_ratios[np.argmin(np.abs(all_ratios - target))]
        dev_pct = abs(nearest - target) / target * 100
        marker = " ◄" if dev_pct < 2.0 else ""
        print(f"  {'φ^'+str(n):>8} {target:>10.4f} {nearest:>15.4f} "
              f"{dev_pct:>9.1f}%{marker}")

    # Lomb-Scargle periodogram of ln(ratio) values
    # Treat each ratio as a "measurement" and search for periodicity
    # in the distribution of ln(ratios)
    if len(all_ln_ratios) > 20:
        ln_r_sorted = np.sort(all_ln_ratios)
        # Create a pseudo-signal: histogram in ln(ratio) space
        n_bins = min(100, len(all_ln_ratios) // 3)
        hist, bin_edges = np.histogram(all_ln_ratios, bins=n_bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Periodogram of the histogram
        freqs_ls = np.linspace(0.1, 20.0, 5000)
        angular_freqs = 2 * np.pi * freqs_ls

        hist_demean = hist - np.mean(hist)
        power = lombscargle(bin_centers, hist_demean, angular_freqs,
                            normalize=True)

        # Bootstrap significance
        n_boot = 5000
        rng = np.random.default_rng(42)
        null_max = np.zeros(n_boot)
        for i in range(n_boot):
            shuffled = rng.permutation(hist_demean)
            p_null = lombscargle(bin_centers, shuffled, angular_freqs,
                                 normalize=True)
            null_max[i] = p_null.max()

        sig_1 = float(np.percentile(null_max, 68.27))
        sig_2 = float(np.percentile(null_max, 95.45))
        sig_3 = float(np.percentile(null_max, 99.73))

        # Check at φ harmonics
        print(f"\n  Lomb-Scargle of ln(ratio) distribution:")
        print(f"  1σ = {sig_1:.4f}, 2σ = {sig_2:.4f}, 3σ = {sig_3:.4f}")
        print(f"\n  {'m':>3} {'k_target':>10} {'k_found':>10} "
              f"{'power':>8} {'significance':>12}")
        print(f"  {'─' * 50}")

        harmonic_results = []
        for m in range(1, 5):
            k_target = m * K_PHI
            mask = np.abs(freqs_ls - k_target) < 0.5
            if not np.any(mask):
                continue
            idx_local = np.argmax(power[mask])
            idx_global = np.where(mask)[0][idx_local]
            k_found = freqs_ls[idx_global]
            p_found = power[idx_global]

            if p_found > sig_3:
                sigma = '>3σ'
            elif p_found > sig_2:
                sigma = '>2σ'
            elif p_found > sig_1:
                sigma = '>1σ'
            else:
                sigma = '<1σ'

            harmonic_results.append({
                'm': m, 'k_target': float(k_target),
                'k_found': float(k_found), 'power': float(p_found),
                'sigma': sigma,
            })
            print(f"  {m:>3} {k_target:>10.3f} {k_found:>10.3f} "
                  f"{p_found:>8.4f} {sigma:>12}")
    else:
        freqs_ls = np.array([])
        power = np.array([])
        sig_1, sig_2, sig_3 = 0, 0, 0
        harmonic_results = []

    return {
        'n_events': len(event_qnm_data),
        'n_ratios': len(all_ratios),
        'all_ratios': all_ratios,
        'all_ln_ratios': all_ln_ratios,
        'event_qnm_data': event_qnm_data,
        'periodogram_freqs': freqs_ls,
        'periodogram_power': power,
        'sig_levels': {'1sigma': sig_1, '2sigma': sig_2, '3sigma': sig_3},
        'harmonic_results': harmonic_results,
    }


# ══════════════════════════════════════════════════════════════
# 3. TEST B — RINGDOWN POWER SPECTRUM ANALYSIS
# ══════════════════════════════════════════════════════════════

def fetch_ringdown_strain(event_name, gps_time, detector='H1',
                          pre_merger=0.5, post_merger=0.5):
    """
    Download strain data around a merger event.

    Returns the post-merger (ringdown) portion.
    """
    try:
        from gwpy.timeseries import TimeSeries
    except ImportError:
        print("    ⚠ gwpy not installed — skipping strain download")
        return None

    t_start = gps_time - pre_merger - 2  # extra padding for filtering
    t_end = gps_time + post_merger + 2

    cache_file = CACHE_DIR / f"{event_name}_{detector}_ringdown.npz"
    if cache_file.exists():
        data = np.load(str(cache_file))
        return {
            'times': data['times'],
            'strain': data['strain'],
            'sample_rate': float(data['sample_rate']),
            'detector': detector,
        }

    try:
        print(f"    Downloading {event_name} ({detector})...", end='', flush=True)
        strain = TimeSeries.fetch_open_data(detector, t_start, t_end)

        # Bandpass filter: 30 Hz to 1024 Hz
        strain = strain.bandpass(30, 1024)

        # Extract post-merger portion
        t_merger = gps_time
        ringdown_mask = (strain.times.value >= t_merger) & \
                        (strain.times.value <= t_merger + post_merger)

        times = strain.times.value[ringdown_mask] - t_merger
        h = strain.value[ringdown_mask]
        sr = float(strain.sample_rate.value)

        # Cache
        np.savez(str(cache_file), times=times, strain=h, sample_rate=sr)
        print(f" done ({len(times)} samples at {sr:.0f} Hz)")

        return {
            'times': times,
            'strain': h,
            'sample_rate': sr,
            'detector': detector,
        }

    except Exception as e:
        print(f" FAILED: {e}")
        return None


def analyze_ringdown_spectrum(ringdown_data, event_qnm, event_name):
    """
    Compute power spectrum of ringdown in ln(f) space.
    Search for φ-periodic modulations.
    """
    h = ringdown_data['strain']
    sr = ringdown_data['sample_rate']

    if len(h) < 64:
        return None

    # Compute power spectrum
    nperseg = min(len(h), 256)
    freqs_psd, psd = welch(h, fs=sr, nperseg=nperseg,
                           noverlap=nperseg // 2)

    # Restrict to physically meaningful range (30 Hz to Nyquist/2)
    f_min = 30.0
    f_max = sr / 4
    mask = (freqs_psd > f_min) & (freqs_psd < f_max) & (psd > 0)

    if np.sum(mask) < 20:
        return None

    freqs_cut = freqs_psd[mask]
    psd_cut = psd[mask]

    # Work in ln(f) space
    ln_f = np.log(freqs_cut)
    ln_psd = np.log10(psd_cut)  # log10 for dynamic range

    # Remove smooth trend (polynomial in ln(f))
    poly_order = 3
    coeffs = np.polyfit(ln_f, ln_psd, poly_order)
    trend = np.polyval(coeffs, ln_f)
    residuals = ln_psd - trend

    # Lomb-Scargle in ln(f) space
    n_ls = 3000
    freqs_ls = np.linspace(0.1, 15.0, n_ls)
    angular_freqs = 2 * np.pi * freqs_ls

    y = residuals - np.mean(residuals)
    power = lombscargle(ln_f, y, angular_freqs, normalize=True)

    return {
        'event_name': event_name,
        'ln_f': ln_f,
        'residuals': residuals,
        'ls_freqs': freqs_ls,
        'ls_power': power,
        'n_points': len(ln_f),
        'ln_f_range': float(ln_f.max() - ln_f.min()),
    }


def test_b_ringdown_stacking(events, n_events_max=20, detectors=('H1', 'L1')):
    """
    TEST B: Stack ringdown power spectra across events,
    search for common φ-periodic modulation.
    """
    print("\n" + "=" * 70)
    print("  TEST B: Ringdown Power Spectrum Stacking")
    print("=" * 70)

    # Select top events by SNR
    top_events = [ev for ev in events if ev.get('snr', 0) and ev['snr'] > 10]
    top_events = top_events[:n_events_max]

    print(f"\n  Analyzing top {len(top_events)} events (SNR > 10)...")

    all_spectra = []

    for ev in top_events:
        # Compute QNM for this event
        qnm_data = compute_qnm_spectrum(ev['final_mass'], ev['final_spin'])

        for det in detectors:
            ringdown = fetch_ringdown_strain(
                ev['name'], ev['gps'], detector=det,
                post_merger=0.5
            )

            if ringdown is None:
                continue

            result = analyze_ringdown_spectrum(ringdown, qnm_data, ev['name'])

            if result is not None:
                all_spectra.append(result)
                print(f"    ✓ {ev['name']} ({det}): {result['n_points']} "
                      f"freq points, Δln(f) = {result['ln_f_range']:.2f}")

    print(f"\n  Successfully analyzed: {len(all_spectra)} ringdown spectra")

    if len(all_spectra) < 3:
        print("  ⚠ Too few spectra for stacking analysis.")
        return {'status': 'insufficient_data', 'n_spectra': len(all_spectra)}

    # Stack periodograms
    # Interpolate all onto common frequency grid
    common_freqs = all_spectra[0]['ls_freqs']
    stacked_power = np.zeros_like(common_freqs)

    for spec in all_spectra:
        # All should share the same frequency grid, but interpolate to be safe
        stacked_power += spec['ls_power']

    stacked_power /= len(all_spectra)

    # Bootstrap significance for stacked periodogram
    n_boot = 5000
    rng = np.random.default_rng(42)
    null_max = np.zeros(n_boot)

    print(f"  Stacking bootstrap ({n_boot})...", end='', flush=True)
    for i in range(n_boot):
        shuffled_stack = np.zeros_like(common_freqs)
        for spec in all_spectra:
            y_shuf = rng.permutation(spec['residuals'] - np.mean(spec['residuals']))
            p_null = lombscargle(spec['ln_f'], y_shuf,
                                 2 * np.pi * common_freqs, normalize=True)
            shuffled_stack += p_null
        shuffled_stack /= len(all_spectra)
        null_max[i] = shuffled_stack.max()
        if (i + 1) % 1000 == 0:
            print(f" {i+1}", end='', flush=True)
    print(" done.")

    sig_1 = float(np.percentile(null_max, 68.27))
    sig_2 = float(np.percentile(null_max, 95.45))
    sig_3 = float(np.percentile(null_max, 99.73))

    # Check harmonics
    print(f"\n  Stacked periodogram significance levels:")
    print(f"  1σ = {sig_1:.4f}, 2σ = {sig_2:.4f}, 3σ = {sig_3:.4f}")

    harmonic_results = []
    print(f"\n  {'m':>3} {'k_target':>10} {'k_found':>10} "
          f"{'power':>8} {'significance':>12}")
    print(f"  {'─' * 50}")

    for m in range(1, 5):
        k_target = m * K_PHI
        mask = np.abs(common_freqs - k_target) < 0.5
        if not np.any(mask):
            continue
        idx_local = np.argmax(stacked_power[mask])
        idx_global = np.where(mask)[0][idx_local]
        k_found = common_freqs[idx_global]
        p_found = stacked_power[idx_global]

        if p_found > sig_3:
            sigma = '>3σ'
        elif p_found > sig_2:
            sigma = '>2σ'
        elif p_found > sig_1:
            sigma = '>1σ'
        else:
            sigma = '<1σ'

        harmonic_results.append({
            'm': m, 'k_target': float(k_target),
            'k_found': float(k_found), 'power': float(p_found),
            'sigma': sigma,
        })
        print(f"  {m:>3} {k_target:>10.3f} {k_found:>10.3f} "
              f"{p_found:>8.4f} {sigma:>12}")

    return {
        'n_spectra': len(all_spectra),
        'stacked_freqs': common_freqs,
        'stacked_power': stacked_power,
        'sig_levels': {'1sigma': sig_1, '2sigma': sig_2, '3sigma': sig_3},
        'harmonic_results': harmonic_results,
        'individual_spectra': all_spectra,
    }


# ══════════════════════════════════════════════════════════════
# 4. PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_results(test_a, test_b):
    """Generate summary plots for both tests."""

    n_rows = 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 10))

    # ─── Test A: Top-left — ratio histogram ───
    ax = axes[0, 0]
    if 'all_ratios' in test_a:
        ratios = test_a['all_ratios']
        ax.hist(ratios[ratios < 5], bins=50, density=True,
                color='steelblue', alpha=0.7, edgecolor='white')

        # Mark φ powers
        for n in range(-2, 4):
            val = PHI ** n
            if 0.3 < val < 5:
                ax.axvline(val, color='gold', ls='--', lw=2,
                           label=f'φ^{n}={val:.3f}' if n >= 0 else None)

        ax.set_xlabel('Frequency ratio f_n / f_0', fontsize=11)
        ax.set_ylabel('Probability density', fontsize=11)
        ax.set_title('Test A: QNM Frequency Ratios', fontsize=12)
        ax.legend(fontsize=8)

    # ─── Test A: Top-right — periodogram ───
    ax = axes[0, 1]
    if 'periodogram_power' in test_a and len(test_a['periodogram_power']) > 0:
        ax.plot(test_a['periodogram_freqs'], test_a['periodogram_power'],
                '-', color='gray', lw=0.6)

        for m in range(1, 5):
            ax.axvline(m * K_PHI, color='gold', lw=2 if m == 1 else 1,
                       alpha=0.7, label=f'm={m}: k={m*K_PHI:.2f}')

        sl = test_a['sig_levels']
        for level, color in [('1sigma', 'green'), ('2sigma', 'orange'),
                              ('3sigma', 'red')]:
            ax.axhline(sl[level], color=color, ls='--', lw=1,
                       label=f'{level}: {sl[level]:.3f}')

        ax.set_xlabel('Frequency k (cycles per unit ln(ratio))', fontsize=11)
        ax.set_ylabel('Lomb-Scargle Power', fontsize=11)
        ax.set_title('Test A: Periodogram of ln(ratio) distribution', fontsize=12)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlim(0, 15)

    # ─── Test B: Bottom-left — example ringdown spectrum ───
    ax = axes[1, 0]
    if isinstance(test_b, dict) and 'individual_spectra' in test_b:
        specs = test_b['individual_spectra']
        if len(specs) > 0:
            # Plot first few
            for i, spec in enumerate(specs[:5]):
                ax.scatter(spec['ln_f'], spec['residuals'], s=1,
                           alpha=0.5, label=spec['event_name'] if i < 3 else None)

            # Mark φ intervals
            for j in range(20):
                x = j * LN_PHI
                if 3 < x < 10:
                    ax.axvline(x, color='gold', ls='--', alpha=0.2)

            ax.set_xlabel('ln(f) [Hz]', fontsize=11)
            ax.set_ylabel('PSD residual (detrended)', fontsize=11)
            ax.set_title('Test B: Ringdown PSD Residuals', fontsize=12)
            ax.legend(fontsize=8)

    # ─── Test B: Bottom-right — stacked periodogram ───
    ax = axes[1, 1]
    if isinstance(test_b, dict) and 'stacked_power' in test_b:
        ax.plot(test_b['stacked_freqs'], test_b['stacked_power'],
                '-', color='gray', lw=0.6)

        for m in range(1, 5):
            ax.axvline(m * K_PHI, color='gold', lw=2 if m == 1 else 1,
                       alpha=0.7, label=f'm={m}: k={m*K_PHI:.2f}')

        sl = test_b['sig_levels']
        for level, color in [('1sigma', 'green'), ('2sigma', 'orange'),
                              ('3sigma', 'red')]:
            ax.axhline(sl[level], color=color, ls='--', lw=1,
                       label=f'{level}: {sl[level]:.3f}')

        ax.set_xlabel('Frequency k (cycles per unit ln f)', fontsize=11)
        ax.set_ylabel('Stacked Lomb-Scargle Power', fontsize=11)
        ax.set_title(f'Test B: Stacked Periodogram '
                     f'({test_b["n_spectra"]} ringdowns)', fontsize=12)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlim(0, 15)

    plt.tight_layout()
    fig_path = FIG_DIR / 'ligo_dsi_search.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to {fig_path}")
    # Avoid blocking CLI/batch runs in non-interactive workflows
    if os.environ.get('LIGO_DSI_SHOW', '0') == '1':
        plt.show()
    else:
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════

def main(skip_strain=False, n_events_max=20):
    """
    Run the complete LIGO DSI search.

    Parameters
    ----------
    skip_strain : bool
        If True, only run Test A (no strain download needed).
    n_events_max : int
        Maximum number of events for strain analysis.
    """
    print("=" * 70)
    print("  LIGO DISCRETE SCALE INVARIANCE (DSI) SEARCH")
    print("  f(φ) ≈ φ^{-n} · cos(2π ln(r) / ln(φ))")
    print("  Searching in gravitational wave ringdown spectra")
    print("=" * 70)

    # Fetch catalog
    events = fetch_catalog(min_snr=8.0)

    if len(events) == 0:
        print("  ⚠ No events found. Using fallback catalog.")
        events = _fallback_catalog()

    print(f"\n  Working with {len(events)} events")
    print(f"  Top 5 by SNR:")
    for ev in events[:5]:
        print(f"    {ev['name']:>25s}  SNR={ev.get('snr', '?'):>5}  "
              f"Mf={ev['final_mass']:.1f} M☉  af={ev['final_spin']:.3f}")

    # ═══ TEST A ═══
    test_a_results = test_a_qnm_ratios(events)

    # ═══ TEST B ═══
    if skip_strain:
        print("\n  [Test B skipped — use --strain to enable]")
        test_b_results = {'status': 'skipped'}
    else:
        test_b_results = test_b_ringdown_stacking(
            events, n_events_max=n_events_max
        )

    # ═══ PLOT ═══
    plot_results(test_a_results, test_b_results)

    # ═══ SUMMARY ═══
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n  Test A (QNM frequency ratios):")
    if 'harmonic_results' in test_a_results:
        for h in test_a_results['harmonic_results']:
            print(f"    m={h['m']}: k={h['k_found']:.3f} "
                  f"(target {h['k_target']:.3f}), "
                  f"power={h['power']:.4f}, {h['sigma']}")

    if isinstance(test_b_results, dict) and 'harmonic_results' in test_b_results:
        print(f"\n  Test B (stacked ringdown spectra, "
              f"{test_b_results['n_spectra']} events):")
        for h in test_b_results['harmonic_results']:
            print(f"    m={h['m']}: k={h['k_found']:.3f} "
                  f"(target {h['k_target']:.3f}), "
                  f"power={h['power']:.4f}, {h['sigma']}")

    # Save
    save_data = {
        'phi': float(PHI),
        'k_phi': float(K_PHI),
        'n_events': len(events),
        'test_a': {k: v for k, v in test_a_results.items()
                   if k not in ('all_ratios', 'all_ln_ratios',
                                'periodogram_freqs', 'periodogram_power')},
        'test_b': {k: v for k, v in test_b_results.items()
                   if k not in ('stacked_freqs', 'stacked_power',
                                'individual_spectra')}
                  if isinstance(test_b_results, dict) else test_b_results,
    }

    results_file = RESULTS_DIR / 'ligo_dsi_search_results.json'
    with open(str(results_file), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")

    return test_a_results, test_b_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LIGO DSI Search')
    parser.add_argument('--strain', action='store_true',
                        help='Download strain data for Test B (slower)')
    parser.add_argument('--n-events', type=int, default=20,
                        help='Max events for strain analysis')
    args = parser.parse_args()

    main(skip_strain=not args.strain, n_events_max=args.n_events)
