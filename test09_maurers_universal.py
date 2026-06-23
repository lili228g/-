"""
NIST SP 800-22 Test 9: Maurer's "Universal Statistical" Test

Purpose:
    Detect whether or not the sequence can be significantly compressed
    without loss of information. A significantly compressible sequence is
    considered non-random. The test computes the sum of log2 distances
    between matching L-bit patterns in the sequence.

Reference: NIST SP 800-22 Rev 1a, Section 2.9
"""

import numpy as np
from math import log, sqrt, erfc
from typing import Union

# Expected values of fn and variance c for each L (from NIST table §2.9)
# Keys: L (block length). Values: (expected_value, variance_coeff)
_NIST_TABLE = {
     1: (0.7326495,  0.690),
     2: (1.5374383,  1.338),
     3: (2.4016068,  1.901),
     4: (3.3112247,  2.358),
     5: (4.2534266,  2.705),
     6: (5.2177052,  2.954),
     7: (6.1962507,  3.125),
     8: (7.1836656,  3.238),
     9: (8.1764248,  3.311),
    10: (9.1723243,  3.356),
    11: (10.170032,  3.384),
    12: (11.168765,  3.401),
    13: (12.168070,  3.410),
    14: (13.167693,  3.416),
    15: (14.167488,  3.419),
    16: (15.167379,  3.421),
}

# Recommended (Q, L) pairs from NIST spec (Table 1, §2.9.7)
_RECOMMENDED = {
    387840:   (1280,  6),
    904960:   (2560,  7),
    2068480:  (5120,  8),
    4654080:  (10240, 9),
    10342400: (20480, 10),
    22753280: (40960, 11),
    49643520: (81920, 12),
    107560960:(163840,13),
    231669760:(327680,14),
    496435200:(655360,15),
    1059061760:(1310720,16),
}


def maurers_universal(
    bits: Union[list, np.ndarray],
    L: int = None,
    Q: int = None,
) -> dict:
    """
    Maurer's Universal Statistical Test (NIST SP 800-22, Test 9).

    Parameters
    ----------
    bits : array-like of 0/1
        Binary sequence (n >= 387840 recommended for L=6).
    L : int, optional
        Block length. Auto-selected from n if not given.
    Q : int, optional
        Initialisation segment length. Auto-selected if not given.

    Returns
    -------
    dict with keys:
        'p_value'  : float  – p-value (>= 0.01 => pass)
        'fn'       : float  – observed test statistic (sum of log2 distances / K)
        'expected' : float  – expected fn under H0
        'sigma'    : float  – standard deviation of fn
        'L'        : int    – block length used
        'Q'        : int    – initialisation blocks used
        'K'        : int    – test blocks used
        'passed'   : bool   – True if p_value >= 0.01
    """
    bits = np.asarray(bits, dtype=int)
    n = len(bits)

    # Auto-select L and Q
    if L is None or Q is None:
        L, Q = _select_params(n)

    if L not in _NIST_TABLE:
        raise ValueError(f"L must be one of {sorted(_NIST_TABLE.keys())}")

    expected_fn, c = _NIST_TABLE[L]
    K = n // L - Q  # number of test blocks

    if K <= 0:
        raise ValueError(
            f"Sequence too short for L={L}, Q={Q}. Need at least {(Q + 1) * L} bits."
        )

    # Step 1: Initialisation — record last occurrence index of each L-bit pattern
    table = {}
    for i in range(Q):
        block = _block_to_int(bits[i * L : (i + 1) * L])
        table[block] = i + 1  # 1-based index

    # Step 2: Test — accumulate log2 distances
    fn_sum = 0.0
    for i in range(Q, Q + K):
        block = _block_to_int(bits[i * L : (i + 1) * L])
        if block in table:
            distance = i + 1 - table[block]
        else:
            distance = i + 1  # first occurrence: distance from start
        fn_sum += log(distance, 2)
        table[block] = i + 1

    fn = fn_sum / K

    # Standard deviation
    sigma = sqrt(c / K)

    # P-value
    p_value = float(erfc(abs(fn - expected_fn) / (sqrt(2) * sigma)))

    return {
        "p_value": p_value,
        "fn":      float(fn),
        "expected": float(expected_fn),
        "sigma":   float(sigma),
        "L":       L,
        "Q":       Q,
        "K":       K,
        "passed":  bool(p_value >= 0.01),
    }


def _block_to_int(block: np.ndarray) -> int:
    """Convert a binary array to an integer."""
    result = 0
    for bit in block:
        result = (result << 1) | int(bit)
    return result


def _select_params(n: int):
    """Select L and Q from NIST recommended table based on sequence length."""
    for threshold in sorted(_RECOMMENDED.keys()):
        if n >= threshold:
            Q, L = _RECOMMENDED[threshold]
    # Default to L=6 if nothing matched
    try:
        return L, Q
    except NameError:
        return 6, 1280


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(9)

    print("=== Test 9: Maurer's Universal Statistical Test ===\n")

    # Random sequence (should pass)
    bits_random = rng.integers(0, 2, 1_000_000)
    result = maurers_universal(bits_random, L=7, Q=1280)
    print("Random sequence (n=1,000,000, L=7):")
    print(f"  K         = {result['K']}")
    print(f"  fn        = {result['fn']:.6f}")
    print(f"  expected  = {result['expected']:.6f}")
    print(f"  sigma     = {result['sigma']:.6f}")
    print(f"  p-value   = {result['p_value']:.6f}")
    print(f"  Passed    = {result['passed']}\n")

    # Compressible sequence (e.g., constant 0 — should fail)
    bits_flat = np.zeros(1_000_000, dtype=int)
    result2 = maurers_universal(bits_flat, L=7, Q=1280)
    print("All-zeros sequence:")
    print(f"  fn        = {result2['fn']:.6f}")
    print(f"  p-value   = {result2['p_value']:.6f}")
    print(f"  Passed    = {result2['passed']}")
