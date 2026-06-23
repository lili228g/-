"""
NIST SP 800-22 Tests 11 & 12: Serial Test and Approximate Entropy Test

Test 11 – Serial Test
    Determines whether the frequency of all possible overlapping m-bit patterns
    across the sequence is approximately equal as would be expected for a truly
    random sequence. Returns two p-values (for ∇ψ²_m and ∇²ψ²_m).

Test 12 – Approximate Entropy Test
    Compares the frequency of overlapping blocks of two consecutive lengths
    (m and m+1) against the expected result for a random sequence. Closely
    related to the serial test, but uses the approximate entropy measure.

Reference: NIST SP 800-22 Rev 1a, Sections 2.11 and 2.12
"""

import numpy as np
from scipy.special import gammainc
from math import log
from typing import Union


# =============================================================================
# Shared helper: psi²_m (overlapping pattern frequency statistic)
# =============================================================================

def _psi_sq(bits: np.ndarray, m: int) -> float:
    """
    Compute ψ²_m — the chi-square-like statistic for overlapping m-bit patterns.
    The sequence is extended by wrapping the first m-1 bits at the end.
    """
    n = len(bits)
    # Augment sequence for circular treatment
    augmented = np.concatenate([bits, bits[:m - 1]])

    # Count all 2^m overlapping patterns
    counts = np.zeros(2 ** m, dtype=int)
    for i in range(n):
        val = 0
        for k in range(m):
            val = (val << 1) | int(augmented[i + k])
        counts[val] += 1

    psi_sq = (2 ** m / n) * np.sum(counts ** 2) - n
    return float(psi_sq)


# =============================================================================
# Test 11: Serial Test
# =============================================================================

def serial_test(
    bits: Union[list, np.ndarray],
    m: int = 3,
) -> dict:
    """
    Serial Test (NIST SP 800-22, Test 11).

    Parameters
    ----------
    bits : array-like of 0/1
        Binary sequence to test (n >= 10 * 2^m recommended).
    m : int
        Block length (default 3). Must satisfy n >> 2^m.

    Returns
    -------
    dict with keys:
        'p_value1'  : float  – p-value for ∇ψ²_m (>= 0.01 => pass)
        'p_value2'  : float  – p-value for ∇²ψ²_m (>= 0.01 => pass)
        'del1'      : float  – ∇ψ²_m statistic
        'del2'      : float  – ∇²ψ²_m statistic
        'psi_m'     : float  – ψ²_m
        'psi_m1'    : float  – ψ²_{m-1}
        'psi_m2'    : float  – ψ²_{m-2}
        'passed'    : bool   – True if both p-values >= 0.01
    """
    bits = np.asarray(bits, dtype=int)

    psi_m  = _psi_sq(bits, m)
    psi_m1 = _psi_sq(bits, m - 1)
    psi_m2 = _psi_sq(bits, m - 2) if m >= 2 else 0.0

    del1 = psi_m - psi_m1          # ∇ψ²_m
    del2 = psi_m - 2 * psi_m1 + psi_m2  # ∇²ψ²_m

    # Degrees of freedom
    df1 = 2 ** (m - 1)
    df2 = 2 ** (m - 2)

    p_value1 = float(1.0 - gammainc(df1 / 2.0, del1 / 2.0))
    p_value2 = float(1.0 - gammainc(df2 / 2.0, del2 / 2.0)) if m >= 2 else 1.0

    return {
        "p_value1": p_value1,
        "p_value2": p_value2,
        "del1":     float(del1),
        "del2":     float(del2),
        "psi_m":    float(psi_m),
        "psi_m1":   float(psi_m1),
        "psi_m2":   float(psi_m2),
        "passed":   bool(p_value1 >= 0.01 and p_value2 >= 0.01),
    }


# =============================================================================
# Test 12: Approximate Entropy Test
# =============================================================================

def approximate_entropy(
    bits: Union[list, np.ndarray],
    m: int = 10,
) -> dict:
    """
    Approximate Entropy Test (NIST SP 800-22, Test 12).

    Parameters
    ----------
    bits : array-like of 0/1
        Binary sequence to test (n >= 100 * 2^m recommended).
    m : int
        Block length (default 10). Must satisfy m < floor(log2(n)) - 5.

    Returns
    -------
    dict with keys:
        'p_value'   : float  – p-value (>= 0.01 => pass)
        'chi2'      : float  – chi-square statistic
        'ApEn'      : float  – approximate entropy value
        'phi_m'     : float  – φ(m) component
        'phi_m1'    : float  – φ(m+1) component
        'passed'    : bool   – True if p_value >= 0.01
    """
    bits = np.asarray(bits, dtype=int)
    n = len(bits)

    phi_m  = _phi_m(bits, m)
    phi_m1 = _phi_m(bits, m + 1)

    ApEn = phi_m - phi_m1

    chi2 = 2.0 * n * (log(2) - ApEn)
    df   = 2 ** m

    p_value = float(1.0 - gammainc(df / 2.0, chi2 / 2.0))

    return {
        "p_value": p_value,
        "chi2":    float(chi2),
        "ApEn":    float(ApEn),
        "phi_m":   float(phi_m),
        "phi_m1":  float(phi_m1),
        "passed":  bool(p_value >= 0.01),
    }


def _phi_m(bits: np.ndarray, m: int) -> float:
    """
    Compute φ(m) = sum_{all m-patterns} (count/n) * log(count/n).
    The sequence is augmented circularly.
    """
    n = len(bits)
    augmented = np.concatenate([bits, bits[:m]])  # circular wrap

    counts = np.zeros(2 ** m, dtype=int)
    for i in range(n):
        val = 0
        for k in range(m):
            val = (val << 1) | int(augmented[i + k])
        counts[val] += 1

    phi = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            phi += p * log(p)
    return float(phi)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n = 1_000_000
    bits = rng.integers(0, 2, n)

    print("=== Test 11: Serial Test ===\n")
    result11 = serial_test(bits, m=3)
    print(f"Random sequence (n={n}, m=3):")
    print(f"  ψ²_m  = {result11['psi_m']:.4f}")
    print(f"  ∇ψ²_m = {result11['del1']:.4f}")
    print(f"  ∇²ψ²m = {result11['del2']:.4f}")
    print(f"  p1    = {result11['p_value1']:.6f}")
    print(f"  p2    = {result11['p_value2']:.6f}")
    print(f"  Passed = {result11['passed']}\n")

    # Biased: 70% ones
    bits_biased = (rng.random(n) < 0.7).astype(int)
    result11b = serial_test(bits_biased, m=3)
    print(f"Biased (70% ones): p1={result11b['p_value1']:.6f}, p2={result11b['p_value2']:.6f}, passed={result11b['passed']}\n")

    print("=== Test 12: Approximate Entropy Test ===\n")
    result12 = approximate_entropy(bits, m=10)
    print(f"Random sequence (n={n}, m=10):")
    print(f"  ApEn    = {result12['ApEn']:.6f}")
    print(f"  chi²    = {result12['chi2']:.4f}")
    print(f"  p-value = {result12['p_value']:.6f}")
    print(f"  Passed  = {result12['passed']}\n")

    result12b = approximate_entropy(bits_biased, m=10)
    print(f"Biased (70% ones): p-value={result12b['p_value']:.6f}, passed={result12b['passed']}")
