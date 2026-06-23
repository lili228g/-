"""
NIST SP 800-22 Tests 7 & 8: Template Matching Tests

Test 7 – Non-overlapping Template Matching Test
    Detects sequences with too many or too few occurrences of a specified
    aperiodic m-bit pattern. The sequence is divided into N=8 blocks;
    within each block the count of non-overlapping template occurrences is
    compared to the expected count via chi-square.

Test 8 – Overlapping Template Matching Test
    Similar but counts *overlapping* occurrences of a template of all-ones
    (length m). The Poisson/chi-square distribution of occurrence counts is
    compared to theoretical values derived from the limiting distribution.

Reference: NIST SP 800-22 Rev 1a, Sections 2.7 and 2.8
"""

import numpy as np
from scipy.special import gammainc, comb
from math import exp, factorial
from typing import Union, Optional


# =============================================================================
# Test 7: Non-overlapping Template Matching
# =============================================================================

def non_overlapping_template(
    bits: Union[list, np.ndarray],
    template: Optional[Union[list, np.ndarray]] = None,
    m: int = 9,
) -> dict:
    """
    Non-overlapping Template Matching Test (NIST SP 800-22, Test 7).

    Parameters
    ----------
    bits : array-like of 0/1
        Binary sequence to test (n >= 10**6 recommended).
    template : array-like of 0/1, optional
        The aperiodic m-bit template. Defaults to 000000001 (m=9).
    m : int
        Template length if ``template`` is not provided (default 9).

    Returns
    -------
    dict with keys:
        'p_value'  : float  – p-value (>= 0.01 => pass)
        'chi2'     : float  – chi-square statistic
        'W'        : list   – occurrence counts per block
        'mu'       : float  – expected occurrences per block
        'sigma2'   : float  – variance per block
        'N'        : int    – number of blocks
        'M'        : int    – block length
        'passed'   : bool   – True if p_value >= 0.01
    """
    bits = np.asarray(bits, dtype=int)
    n = len(bits)

    if template is None:
        template = np.zeros(m, dtype=int)
        template[-1] = 1  # default: 000…001
    else:
        template = np.asarray(template, dtype=int)
        m = len(template)

    N = 8                   # NIST fixed value
    M = n // N              # block length

    mu = (M - m + 1) / (2 ** m)
    sigma2 = M * (1.0 / (2 ** m) - (2 * m - 1) / (2 ** (2 * m)))

    W = []
    for j in range(N):
        block = bits[j * M : (j + 1) * M]
        count = 0
        i = 0
        while i <= M - m:
            if np.array_equal(block[i : i + m], template):
                count += 1
                i += m   # non-overlapping: skip past match
            else:
                i += 1
        W.append(count)

    chi2 = sum((w - mu) ** 2 / sigma2 for w in W)
    # Chi-square with N = 8 degrees of freedom
    p_value = float(1.0 - gammainc(N / 2.0, chi2 / 2.0))

    return {
        "p_value": p_value,
        "chi2":    float(chi2),
        "W":       W,
        "mu":      float(mu),
        "sigma2":  float(sigma2),
        "N":       N,
        "M":       M,
        "passed":  bool(p_value >= 0.01),
    }


# =============================================================================
# Test 8: Overlapping Template Matching
# =============================================================================

def overlapping_template(
    bits: Union[list, np.ndarray],
    m: int = 9,
) -> dict:
    """
    Overlapping Template Matching Test (NIST SP 800-22, Test 8).

    Uses a template of all-ones of length m. Counts *overlapping* occurrences
    within N blocks of length M = 1032. The empirical frequency histogram
    over 6 classes (0..4, >=5) is compared to theoretical probabilities
    from NIST SP 800-22 Table 1 (pre-computed for m=9, M=1032, K=5).

    Parameters
    ----------
    bits : array-like of 0/1
        Binary sequence to test.
    m : int
        Template length (default 9, as in the NIST spec).

    Returns
    -------
    dict with keys:
        'p_value'  : float  – p-value (>= 0.01 => pass)
        'chi2'     : float  – chi-square statistic
        'nu'       : list   – observed frequencies for 6 classes
        'pi'       : list   – theoretical probabilities for 6 classes
        'N'        : int    – number of blocks used
        'passed'   : bool   – True if p_value >= 0.01
    """
    bits = np.asarray(bits, dtype=int)
    n = len(bits)

    template = np.ones(m, dtype=int)  # all-ones template
    M = 1032   # NIST fixed block length
    K = 5      # 6 classes: 0, 1, 2, 3, 4, >=5
    N = n // M

    if N == 0:
        raise ValueError(
            f"Sequence too short; need at least {M} bits for one block."
        )

    # NIST pre-computed theoretical probabilities for m=9, M=1032 (Table 1 §2.8.8)
    # Derived from the exact formula in §3.8 using lambda=2, eta=1
    pi = _compute_pi_overlapping(m, M, K)

    # Count overlapping occurrences in each block
    nu = [0] * (K + 1)   # indices 0..K: K+1 = 6 slots
    for j in range(N):
        block = bits[j * M : (j + 1) * M]
        count = 0
        for i in range(M - m + 1):
            if np.all(block[i : i + m] == template):
                count += 1
        idx = min(count, K)
        nu[idx] += 1

    chi2 = sum(
        (nu[i] - N * pi[i]) ** 2 / (N * pi[i])
        for i in range(K + 1)
        if N * pi[i] > 0
    )

    # Chi-square with K = 5 degrees of freedom
    p_value = float(1.0 - gammainc(K / 2.0, chi2 / 2.0))

    return {
        "p_value": p_value,
        "chi2":    float(chi2),
        "nu":      list(nu),
        "pi":      pi,
        "N":       N,
        "passed":  bool(p_value >= 0.01),
    }


def _compute_pi_overlapping(m: int, M: int, K: int) -> list:
    """
    Compute theoretical class probabilities for the overlapping template test.
    For the canonical case m=9, M=1032, K=5 the NIST-published values are
    returned directly (from NIST SP 800-22 and the sts-2.1.2 reference code).
    For other parameters a Poisson approximation is used.

    References: NIST SP 800-22 §3.8, sts-2.1.2 overlappingTemplateMatchings.c
    """
    # Canonical NIST values for the default m=9, M=1032, K=5 configuration
    if m == 9 and M == 1032 and K == 5:
        return [0.324652, 0.182617, 0.142670, 0.106645, 0.077364, 0.166052]

    # General approximation: negative-binomial / Poisson with eta = lambda/2
    lam = (M - m + 1) / (2 ** m)
    eta = lam / 2.0
    from math import exp as _exp
    # pi[v] ≈ Pr(Poisson(eta) = v) for v < K, rest goes into last bucket
    pi = []
    running = 0.0
    for v in range(K):
        from math import factorial
        p = _exp(-eta) * eta ** v / factorial(v)
        pi.append(p)
        running += p
    pi.append(max(0.0, 1.0 - running))
    return pi


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 1_000_000
    bits = rng.integers(0, 2, n)

    print("=== Test 7: Non-overlapping Template Matching ===\n")
    result7 = non_overlapping_template(bits, m=9)
    print(f"  Template: 000000001 (m=9)")
    print(f"  N blocks  = {result7['N']}, M = {result7['M']}")
    print(f"  mu        = {result7['mu']:.4f}, sigma2 = {result7['sigma2']:.4f}")
    print(f"  W counts  = {result7['W']}")
    print(f"  chi²      = {result7['chi2']:.4f}")
    print(f"  p-value   = {result7['p_value']:.6f}")
    print(f"  Passed    = {result7['passed']}\n")

    # Custom template
    result7b = non_overlapping_template(bits, template=[0, 0, 1, 1, 1, 0, 1, 0, 0])
    print(f"  Template: 001110100, p-value = {result7b['p_value']:.6f}, passed = {result7b['passed']}\n")

    print("=== Test 8: Overlapping Template Matching ===\n")
    result8 = overlapping_template(bits, m=9)
    print(f"  Template: 111111111 (m=9)")
    print(f"  N blocks  = {result8['N']}")
    print(f"  nu (observed freq per class): {result8['nu']}")
    print(f"  pi (theoretical probabilities): {[f'{p:.4f}' for p in result8['pi']]}")
    print(f"  chi²      = {result8['chi2']:.4f}")
    print(f"  p-value   = {result8['p_value']:.6f}")
    print(f"  Passed    = {result8['passed']}")
