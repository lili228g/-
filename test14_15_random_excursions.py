"""
NIST SP 800-22 Tests 14 & 15: Random Excursions Tests

Test 14 – Random Excursions Test
    Determines whether the number of visits to a particular state (integer value)
    in the cumulative sum random walk differs from what would be expected for a
    truly random sequence. Tested for states x ∈ {-4,-3,-2,-1,+1,+2,+3,+4}.
    The walk is decomposed into excursions (subsequences between returns to zero).

Test 15 – Random Excursions Variant Test
    Detects deviations from the expected number of times a given state x is
    visited within a complete random walk. Simpler but complementary to Test 14.
    Tested for states x ∈ {-9,…,-1,+1,…,+9}.

Reference: NIST SP 800-22 Rev 1a, Sections 2.14 and 2.15
"""

import numpy as np
from math import erfc, sqrt, exp, factorial
from scipy.special import gammainc
from typing import Union


# =============================================================================
# Test 14: Random Excursions Test
# =============================================================================

# Theoretical probabilities P(v | x) for each state x and visit count v
# v = 0, 1, 2, 3, 4, 5 (>=5 merged into last class)
# P(v | x) for positive/negative x are symmetric:
# P(v=0|x) = 1 - 1/(2|x|)
# P(v>=1|x) = 1/(4x²) * (1 - 1/(2|x|))^(v-1),  v >= 1
def _excursion_prob(x: int, v: int) -> float:
    """P(visit count = v in one excursion | state x), v in 0..K."""
    ax = abs(x)
    if v == 0:
        return 1.0 - 1.0 / (2.0 * ax)
    else:
        return (1.0 / (4.0 * ax * ax)) * (1.0 - 1.0 / (2.0 * ax)) ** (v - 1)


def random_excursions(bits: Union[list, np.ndarray]) -> dict:
    """
    Random Excursions Test (NIST SP 800-22, Test 14).

    Parameters
    ----------
    bits : array-like of 0/1
        Binary sequence (n >= 1,000,000 recommended).

    Returns
    -------
    dict with keys:
        'results'  : dict mapping state x -> {'p_value', 'chi2', 'nu', 'passed'}
        'J'        : int   – number of complete excursions found
        'passed'   : bool  – True if ALL state p-values >= 0.01
    """
    bits = np.asarray(bits, dtype=int)
    n = len(bits)

    # Convert to ±1 and compute cumulative sum (random walk)
    x_walk = 2 * bits - 1
    S = np.concatenate([[0], np.cumsum(x_walk), [0]])

    # Find zero-crossing indices to identify excursions
    zero_positions = np.where(S == 0)[0]
    J = len(zero_positions) - 1  # number of complete excursions

    if J < 500:
        raise ValueError(
            f"Too few excursions (J={J}); need >= 500. Use a longer sequence (n >= 1,000,000)."
        )

    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    K = 5  # visit classes: 0, 1, 2, 3, 4, >=5

    # Count visits per state per excursion
    results = {}
    for state in states:
        nu = np.zeros(K + 1, dtype=int)
        for j in range(J):
            seg = S[zero_positions[j] : zero_positions[j + 1] + 1]
            visits = int(np.sum(seg == state))
            idx = min(visits, K)
            nu[idx] += 1

        # Build theoretical probabilities for this state
        pi = [_excursion_prob(state, v) for v in range(K)]
        pi.append(1.0 - sum(pi))  # >= K class

        # Chi-square with K degrees of freedom
        chi2 = sum(
            (nu[v] - J * pi[v]) ** 2 / (J * pi[v])
            for v in range(K + 1)
            if J * pi[v] > 0
        )

        p_value = float(1.0 - gammainc(K / 2.0, chi2 / 2.0))

        results[state] = {
            "p_value": p_value,
            "chi2":    float(chi2),
            "nu":      list(nu),
            "passed":  bool(p_value >= 0.01),
        }

    all_passed = all(r["passed"] for r in results.values())

    return {
        "results": results,
        "J":       J,
        "passed":  all_passed,
    }


# =============================================================================
# Test 15: Random Excursions Variant Test
# =============================================================================

def random_excursions_variant(bits: Union[list, np.ndarray]) -> dict:
    """
    Random Excursions Variant Test (NIST SP 800-22, Test 15).

    Parameters
    ----------
    bits : array-like of 0/1
        Binary sequence (n >= 1,000,000 recommended).

    Returns
    -------
    dict with keys:
        'results'  : dict mapping state x -> {'p_value', 'xi', 'J', 'passed'}
        'J'        : int   – number of complete cycles (visits to zero)
        'passed'   : bool  – True if ALL state p-values >= 0.01
    """
    bits = np.asarray(bits, dtype=int)
    n = len(bits)

    # Convert to ±1 and compute cumulative sum
    x_walk = 2 * bits - 1
    S = np.concatenate([[0], np.cumsum(x_walk)])

    # J = number of times the walk returns to zero (cycles)
    J = int(np.sum(S[1:] == 0))

    if J < 500:
        raise ValueError(
            f"Too few cycles (J={J}); need >= 500. Use a longer sequence."
        )

    states = list(range(-9, 0)) + list(range(1, 10))

    results = {}
    for state in states:
        xi = int(np.sum(S == state))  # total visits to state in entire walk
        # P-value via erfc
        p_value = float(erfc(abs(xi - J) / sqrt(2.0 * J * (4.0 * abs(state) - 2))))

        results[state] = {
            "p_value": p_value,
            "xi":      xi,
            "J":       J,
            "passed":  bool(p_value >= 0.01),
        }

    all_passed = all(r["passed"] for r in results.values())

    return {
        "results": results,
        "J":       J,
        "passed":  all_passed,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(14)
    n = 1_000_000

    print("=== Test 14: Random Excursions Test ===\n")
    bits = rng.integers(0, 2, n)

    try:
        result14 = random_excursions(bits)
        print(f"Random sequence (n={n}): J = {result14['J']} excursions")
        print(f"{'State':>6} | {'p-value':>10} | {'chi²':>8} | {'Passed'}")
        print("-" * 45)
        for state, r in sorted(result14["results"].items()):
            print(f"{state:>6} | {r['p_value']:>10.6f} | {r['chi2']:>8.4f} | {r['passed']}")
        print(f"\nOverall passed: {result14['passed']}\n")
    except ValueError as e:
        print(f"Test 14 skipped: {e}\n")

    print("=== Test 15: Random Excursions Variant Test ===\n")

    try:
        result15 = random_excursions_variant(bits)
        print(f"Random sequence (n={n}): J = {result15['J']} cycles")
        print(f"{'State':>6} | {'xi':>8} | {'p-value':>10} | {'Passed'}")
        print("-" * 42)
        for state, r in sorted(result15["results"].items()):
            print(f"{state:>6} | {r['xi']:>8} | {r['p_value']:>10.6f} | {r['passed']}")
        print(f"\nOverall passed: {result15['passed']}")
    except ValueError as e:
        print(f"Test 15 skipped: {e}")
