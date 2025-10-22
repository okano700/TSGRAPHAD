from scipy.signal import periodogram
import numpy as np
from math import isfinite
from typing import List, Optional


def get_period(
    data: np.ndarray,
    n: int = 1,
    fs: float = 1.0,
    min_period: int = 2,
    max_period: Optional[int] = None,
    exclude_harmonics: bool = True,
) -> List[int]:
    """
    Estimate up to `n` dominant periods (in samples) from a 1D signal using the periodogram.

    Strategy
    --------
    1) Compute the (one-sided) periodogram.
    2) Ignore DC (f == 0).
    3) Convert peak frequencies to periods in *samples*: period = round(fs / f_peak).
    4) Walk peaks from highest power to lower and collect unique base periods,
       optionally excluding harmonics (multiples/divisors of already-selected periods).

    Parameters
    ----------
    data : np.ndarray
        1D time series.
    n : int, default=1
        Number of periods to return (best to worst by spectral power).
    fs : float, default=1.0
        Sampling frequency (samples per second). If your data is 1 sample per unit time, leave at 1.
    min_period : int, default=2
        Smallest admissible period (in samples). Periods < 2 are not meaningful.
    max_period : int or None, default=None
        Largest admissible period (in samples). If None, no upper bound is applied.
    exclude_harmonics : bool, default=True
        If True, skip candidates that are integer multiples/divisors of already-selected periods.

    Returns
    -------
    List[int]
        List of up to `n` integer periods (in samples), ordered by decreasing peak power.

    Notes
    -----
    - This is a simple peak-picking over the raw periodogram; for noisy signals,
      consider smoothing or using Welch’s method/peak-prominence.
    - Periods are integers (rounded), expressed in *samples*. To get periods in time units,
      divide by `fs`.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 100.0
    >>> t = np.arange(0, 5, 1/fs)
    >>> x = np.sin(2*np.pi*2*t) + 0.5*np.sin(2*np.pi*5*t)  # 2 Hz and 5 Hz
    >>> get_period(x, n=2, fs=fs)
    [50, 20]  # 100/2 and 100/5 samples per cycle
    """
    x = np.asarray(data).ravel()
    if x.size < 8 or n <= 0:
        return []

    # 1) Periodogram: let SciPy choose nperseg/nfft to avoid invalid combos.
    f, px = periodogram(x, fs=fs, detrend="linear", scaling="spectrum")

    # 2) Remove DC (f == 0)
    mask = f > 0
    f = f[mask]
    px = px[mask]
    if f.size == 0:
        return []

    # 3) Sort peaks by descending power once
    order = np.argsort(px)[::-1]

    periods: List[int] = []
    for idx in order:
        f_peak = float(f[idx])
        if not isfinite(f_peak) or f_peak <= 0:
            continue

        # Convert to period in samples: T_samples = fs / f_hz
        T = int(round(fs / f_peak))
        if T < min_period:
            continue
        if max_period is not None and T > max_period:
            continue

        # 4) Harmonic exclusion: skip if candidate divides or is divisible by any chosen
        if exclude_harmonics:
            harmonic = any((T % p == 0) or (p % T == 0) for p in periods)
            if harmonic:
                continue

        # Keep it
        periods.append(T)
        if len(periods) >= n:
            break

    return periods
