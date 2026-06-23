"""
NIST SP 800-22 Test 10: Linear Complexity Test

Purpose:
    A sequence is divided into blocks of length M. The linear complexity
    L(s_M) — the length of the shortest LFSR generating each block — is
    computed via the Berlekamp-Massey algorithm. For a truly random sequence
    the linear complexity should be close to M/2. The test statistic T_M
    aggregates the deviation of each block's complexity from the expected value
    and compares the resulting frequency histogram to theoretical probabilities.

Reference: NIST SP 800-22 Rev 1a, Section 2.10
"""

import numpy as np
from scipy.special import gammainc
from typing import Union


# Theoretical probabilities for 7 classes (K=6 intervals) from NIST spec §2.10
_PI = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]


def linear_complexity(
    bits: Union[list, np.ndarray],
    M: int = 500,
) -> dict:
    """
    Linear Complexity Test (NIST SP 800-22, Test 10).

    Parameters
    ----------
    bits : array-like of 0/1
        Binary sequence to test. n = len(bits) should be >= M * 200.
    M : int
        Block length; NIST recommends 500 <= M <= 5000 (default 500).

    Returns
    -------
    dict with keys:
        'p_value'  : float       – p-value (>= 0.01 => pass)
        'chi2'     : float       – chi-square statistic
        'nu'       : list[int]   – observed frequency per class (7 classes)
        'pi'       : list[float] – theoretical probabilities per class
        'N'        : int         – number of blocks processed
        'passed'   : bool        – True if p_value >= 0.01
    """
    bits = np.asarray(bits, dtype=int)
    n = len(bits)
    N = n // M

    if N < 200:
        raise ValueError(
            f"Too few blocks: n={n}, M={M} gives N={N}. Need at least 200 blocks."
        )

    # xi_n = n/2 + (4 + r_n) / 18  where r_n = n mod 2
    # Mean of linear complexity for a block of length M
    # (NIST eq. 7)
    mu = M / 2.0 + (4 + (M % 2)) / 18.0 - (M % 2) / 6.0  # simplified

    # K = 6 classes, boundaries at semi-integers
    # Classes: T <= -2.5, -2.5<T<=-1.5, -1.5<T<=-0.5, -0.5<T<=0.5,
    #           0.5<T<=1.5, 1.5<T<=2.5, T>2.5
    K = 6
    nu = [0] * (K + 1)

    for j in range(N):
        block = bits[j * M : (j + 1) * M]
        lc = _berlekamp_massey(block)

        # Test statistic T_M  (NIST eq. 6)
        T = ((-1) ** M) * (lc - mu) + 2.0 / 9.0

        if T <= -2.5:
            nu[0] += 1
        elif T <= -1.5:
            nu[1] += 1
        elif T <= -0.5:
            nu[2] += 1
        elif T <= 0.5:
            nu[3] += 1
        elif T <= 1.5:
            nu[4] += 1
        elif T <= 2.5:
            nu[5] += 1
        else:
            nu[6] += 1

    chi2 = sum(
        (nu[i] - N * _PI[i]) ** 2 / (N * _PI[i])
        for i in range(K + 1)
        if N * _PI[i] > 0
    )

    # P-value: chi-square with K = 6 degrees of freedom
    p_value = float(1.0 - gammainc(K / 2.0, chi2 / 2.0))

    return {
        "p_value": p_value,
        "chi2":    float(chi2),
        "nu":      nu,
        "pi":      list(_PI),
        "N":       N,
        "passed":  bool(p_value >= 0.01),
    }


def _berlekamp_massey(s: np.ndarray) -> int:
    """
    Berlekamp-Massey algorithm over GF(2).
    Returns the linear complexity (length of the shortest LFSR) for sequence s.
    """
    s = list(s)
    n = len(s)
    # c: connection polynomial, b: previous c, L: current LC, m: shift register
    c = [1] + [0] * n
    b = [1] + [0] * n
    L = 0
    m = 1
    for i in range(n):
        # Compute discrepancy
        d = s[i]
        for j in range(1, L + 1):
            d ^= c[j] & s[i - j]
        d &= 1
        if d == 0:
            m += 1
        elif 2 * L <= i:
            t = c[:]
            for j in range(m, n + 1):
                c[j] ^= b[j - m]
            L = i + 1 - L
            b = t
            m = 1
        else:
            for j in range(m, n + 1):
                c[j] ^= b[j - m]
            m += 1
    return L


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(10)

    print("=== Test 10: Linear Complexity Test ===\n")

    # Random sequence (should pass)
    bits_random = rng.integers(0, 2, 1_000_000)
    result = linear_complexity(bits_random, M=500)
    print("Random sequence (n=1,000,000, M=500):")
    print(f"  N blocks  = {result['N']}")
    print(f"  nu counts = {result['nu']}")
    print(f"  chi²      = {result['chi2']:.4f}")
    print(f"  p-value   = {result['p_value']:.6f}")
    print(f"  Passed    = {result['passed']}\n")

    # LFSR-generated sequence (low complexity, should fail)
    # Simple LFSR of period 15 (x^4 + x + 1)
    lfsr = np.zeros(1_000_000, dtype=int)
    state = [1, 0, 0, 1]
    for i in range(1_000_000):
        lfsr[i] = state[0]
        new_bit = (state[0] ^ state[3]) & 1
        state = [new_bit] + state[:3]
    result2 = linear_complexity(lfsr, M=500)
    print("LFSR sequence (period 15, low LC):")
    print(f"  nu counts = {result2['nu']}")
    print(f"  p-value   = {result2['p_value']:.6f}")
    print(f"  Passed    = {result2['passed']}")
