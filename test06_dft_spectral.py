"""
NIST SP 800-22 Test 6: Discrete Fourier Transform (Spectral) Test

Purpose:
    Detect periodic features in the bit sequence that would indicate non-randomness.
    Applies a DFT to the ±1-coded sequence and examines the magnitude of the
    resulting frequency components. 95% of peaks should fall below the
    95% confidence threshold h = sqrt(log(1/0.05) * n).

Reference: NIST SP 800-22 Rev 1a, Section 2.6
"""

import numpy as np
from typing import Union


def dft_spectral(bits: Union[list, np.ndarray]) -> dict:
    """
    Discrete Fourier Transform (Spectral) Test (NIST SP 800-22, Test 6).

    Parameters
    ----------
    bits : array-like of 0/1
        Binary sequence to test (length n should be even, >= 1000).

    Returns
    -------
    dict with keys:
        'p_value'  : float  – p-value (>= 0.01 => pass)
        'N1'       : int    – observed number of peaks below threshold h
        'N0'       : float  – expected number of peaks below threshold (0.95 * n/2)
        'd'        : float  – normalized difference statistic
        'h'        : float  – 95% peak threshold
        'passed'   : bool   – True if p_value >= 0.01
    """
    bits = np.asarray(bits, dtype=float)
    n = len(bits)

    if n < 1000:
        raise ValueError("Sequence length should be at least 1000 bits.")

    # Convert 0/1 to -1/+1
    x = 2.0 * bits - 1.0

    # Apply DFT
    f = np.fft.fft(x)

    # Take magnitudes of first n/2 components (excluding DC at index 0)
    # NIST considers indices 1 .. n/2 - 1 (i.e., the first n/2 peaks)
    modulus = np.abs(f[: n // 2])

    # 95% confidence threshold
    h = np.sqrt(np.log(1.0 / 0.05) * n)

    # N1: observed count of peaks below h
    N1 = int(np.sum(modulus < h))

    # N0: expected count (95% of n/2)
    N0 = 0.95 * n / 2.0

    # Normalized difference
    d = (N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4.0)

    # P-value using complementary error function
    p_value = float(np.exp(-(d ** 2) / 2.0))   # erfc(|d|/sqrt(2)) equivalent
    # More precise version:
    p_value = float(2.0 * (1.0 - _phi(abs(d))))

    return {
        "p_value": p_value,
        "N1":      N1,
        "N0":      N0,
        "d":       float(d),
        "h":       float(h),
        "passed":  bool(p_value >= 0.01),
    }


def _phi(x: float) -> float:
    """Standard normal CDF via erfc."""
    from math import erfc, sqrt
    return 0.5 * erfc(-x / sqrt(2))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Test 6: DFT (Spectral) Test ===\n")

    # Random sequence (should pass)
    bits_random = rng.integers(0, 2, 10_000)
    result = dft_spectral(bits_random)
    print("Random sequence (n=10000):")
    print(f"  h       = {result['h']:.4f}")
    print(f"  N0      = {result['N0']:.2f}")
    print(f"  N1      = {result['N1']}")
    print(f"  d       = {result['d']:.4f}")
    print(f"  p-value = {result['p_value']:.6f}")
    print(f"  Passed  = {result['passed']}\n")

    # Periodic sequence (all 01 repeating — should fail)
    bits_periodic = np.tile([0, 1], 5000)
    result2 = dft_spectral(bits_periodic)
    print("Periodic 010101… sequence (n=10000):")
    print(f"  N1      = {result2['N1']}")
    print(f"  d       = {result2['d']:.4f}")
    print(f"  p-value = {result2['p_value']:.6f}")
    print(f"  Passed  = {result2['passed']}\n")

    # NIST example (n=1000000, p-value ≈ 0.847)
    bits_large = rng.integers(0, 2, 1_000_000)
    result3 = dft_spectral(bits_large)
    print(f"Large random (n=1000000): p-value = {result3['p_value']:.6f}, passed = {result3['passed']}")
