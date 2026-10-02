"""
sim/probes.py
Probe signals: click, log-chirp, MLS, multitone.

All probes are band-limited to fcut before returning, ensuring zero source
energy above the resolved frequency.  Band-limiting is via spectral
truncation with a cosine taper (Tukey window in frequency domain).
"""
import numpy as np
from scipy.signal import max_len_seq


# ── Band-limiting ────────────────────────────────────────────────

def bandlimit_probe(signal, dt, fcut, rolloff_frac=0.10):
    """Band-limit *signal* so that it contains exactly zero energy above fcut.

    Method: DFT → multiply by a gain curve that is 1 below
    ``fcut * (1 - rolloff_frac)``, tapers via a raised-cosine
    (Tukey / Hann half-window) from there to ``fcut``, and is
    exactly 0 above ``fcut`` → inverse DFT.

    Parameters
    ----------
    signal : 1-D array
    dt : float – time step (seconds)
    fcut : float – upper frequency limit (Hz)
    rolloff_frac : float in (0, 1) – fraction of fcut used for the
        cosine transition band.  Default 0.10 gives a taper from
        0.9 * fcut to fcut, preserving >95 % of passband energy while
        avoiding Gibbs ringing in the time domain.

    Returns
    -------
    1-D array, same length as *signal*, with zero DFT energy above fcut.
    """
    N = len(signal)
    S = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=dt)

    rolloff_hz = fcut * rolloff_frac
    f_lo = fcut - rolloff_hz          # taper starts here
    f_hi = fcut                       # hard zero above here

    gain = np.ones_like(freqs)
    taper = (freqs >= f_lo) & (freqs <= f_hi)
    gain[taper] = 0.5 * (1.0 + np.cos(np.pi * (freqs[taper] - f_lo) / rolloff_hz))
    gain[freqs > f_hi] = 0.0

    S *= gain
    return np.fft.irfft(S, n=N)


# ── Probe generation ─────────────────────────────────────────────

def make_probe(kind, n_samples, dt, fcut, rng=None):
    """Return a band-limited probe signal of length *n_samples*.

    Supported kinds: ``click``, ``logchirp``, ``mls``, ``multitone``.
    Every probe is passed through :func:`bandlimit_probe` before
    returning, guaranteeing zero spectral content above *fcut*.
    """
    t = np.arange(n_samples) * dt

    if kind == "click":
        sigma = 1.0 / (2.0 * np.pi * fcut)
        t0 = 4.0 * sigma
        pulse = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
        raw = pulse / (pulse.max() + 1e-12)

    elif kind == "logchirp":
        f0, f1 = 20.0, fcut
        T = t[-1]
        phase = 2 * np.pi * f0 * T / np.log(f1 / f0) * (
            np.exp(t / T * np.log(f1 / f0)) - 1
        )
        raw = np.sin(phase)
        taper = min(len(raw) // 10, 200)
        raw[:taper]  *= np.linspace(0, 1, taper)
        raw[-taper:] *= np.linspace(1, 0, taper)

    elif kind == "mls":
        nbits = min(max(4, int(np.ceil(np.log2(n_samples)))), 18)
        seq, _ = max_len_seq(nbits)
        seq = 2.0 * seq.astype(np.float64) - 1.0
        raw = np.zeros(n_samples)
        raw[: min(len(seq), n_samples)] = seq[:n_samples]

    elif kind == "multitone":
        if rng is None:
            rng = np.random.default_rng(42)
        freqs = rng.uniform(50, fcut, size=8)
        phases = rng.uniform(0, 2 * np.pi, size=8)
        raw = sum(np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, phases))
        raw = raw / 8.0

    else:
        raise ValueError(f"Unknown probe: {kind}")

    # ── Enforce fcut band-limit on every probe ──────────────────
    return bandlimit_probe(raw, dt, fcut)