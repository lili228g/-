"""
NIST SP 800-22 Test 13: Cumulative Sums (Cusum) Test

Purpose:
    Determine whether the cumulative sum of the partial sums of the ±1-coded
    sequence is too large or too small relative to what would be expected for
    a random sequence. Two modes are tested:
      - Forward (mode=0): partial sums S_1, S_2, …, S_n
      - Backward (mode=1): partial sums using the reversed sequence

    Large values suggest a surplus of 1s or 0s at the beginning (forward)
    or end (backward) of the sequence.

Reference: NIST SP 800-22 Rev 1a, Section 2.13
"""

import numpy as np
from math import erfc, sqrt, floor
from typing import Union


def cumulative_sums(
    bits: Union[list, np.ndarray],
    mode: int = 0,
) -> dict:
    """
    Cumulative Sums (Cusum) Test (NIST SP 800-22, Test 13).

    Parameters
    ----------
    bits : array-like of 0/1
        Binary sequence to test (n >= 100 recommended).
    mode : int
        0 = forward (default), 1 = backward (reversed sequence).

    Returns
    -------
    dict with keys:
        'p_value'  : float  – p-value (>= 0.01 => pass)
        'z'        : float  – maximum absolute partial sum (test statistic)
        'mode'     : str    – 'forward' or 'backward'
        'passed'   : bool   – True if p_value >= 0.01
    """
    bits = np.asarray(bits, dtype=int)
    n = len(bits)

    if n < 100:
        raise ValueError("Sequence length should be at least 100 bits.")

    # Convert to ±1
    x = 2 * bits - 1

    if mode == 1:
        x = x[::-1]

    # Compute cumulative sums S_k
    S = np.cumsum(x)

    # Test statistic: maximum absolute partial sum
    z = int(np.max(np.abs(S)))

    # P-value via the NIST formula (Eq. 11 from §3.13)
    p_value = _cusum_p_value(z, n)

    return {
        "p_value": float(p_value),
        "z":       int(z),
        "mode":    "forward" if mode == 0 else "backward",
        "passed":  bool(p_value >= 0.01),
    }


def _cusum_p_value(z: int, n: int) -> float:
    """
    Compute the p-value for the Cusum test using the series approximation
    from NIST SP 800-22 §3.13 (equation 11).

    P = 1 - sum_{k=floor(...)}^{floor(...)} [Phi((4k+1)z/sqrt(n)) - Phi((4k-1)z/sqrt(n))]
          + sum_{k=floor(...)}^{floor(...)} [Phi((4k+3)z/sqrt(n)) - Phi((4k+1)z/sqrt(n))]
    """
    from scipy.stats import norm

    sqrt_n = sqrt(n)

    def phi(x):
        return norm.cdf(x)

    # Lower/upper summation bounds (NIST formula)
    k1_lo = int(floor((-n / z + 1) / 4))
    k1_hi = int(floor((n / z - 1) / 4))
    k2_lo = int(floor((-n / z - 3) / 4))
    k2_hi = int(floor((n / z - 1) / 4))

    sum1 = sum(
        phi((4 * k + 1) * z / sqrt_n) - phi((4 * k - 1) * z / sqrt_n)
        for k in range(k1_lo, k1_hi + 1)
    )
    sum2 = sum(
        phi((4 * k + 3) * z / sqrt_n) - phi((4 * k + 1) * z / sqrt_n)
        for k in range(k2_lo, k2_hi + 1)
    )

    p_value = 1.0 - sum1 + sum2
    return max(0.0, min(1.0, p_value))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(13)
    n = 100_000

    print("=== Test 13: Cumulative Sums (Cusum) Test ===\n")

    bits_random = rng.integers(0, 2, n)

    # Forward mode
    result_fwd = cumulative_sums(bits_random, mode=0)
    print(f"Random sequence (n={n}), Forward:")
    print(f"  z       = {result_fwd['z']}")
    print(f"  p-value = {result_fwd['p_value']:.6f}")
    print(f"  Passed  = {result_fwd['passed']}\n")

    # Backward mode
    result_bwd = cumulative_sums(bits_random, mode=1)
    print(f"Random sequence (n={n}), Backward:")
    print(f"  z       = {result_bwd['z']}")
    print(f"  p-value = {result_bwd['p_value']:.6f}")
    print(f"  Passed  = {result_bwd['passed']}\n")

    # Biased sequence: first half all-ones, second half all-zeros
    bits_biased = np.concatenate([np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int)])
    result_biased = cumulative_sums(bits_biased, mode=0)
    print(f"Biased sequence (first half 1s, second half 0s):")
    print(f"  z       = {result_biased['z']}")
    print(f"  p-value = {result_biased['p_value']:.6f}")
    print(f"  Passed  = {result_biased['passed']}")
