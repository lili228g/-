"""
NIST SP 800-22 Tests 1, 2 & 5: Frequency and Runs Tests

Test 1 – Frequency (Monobit) Test
    Determines whether the number of ones and zeros in a sequence is
    approximately equal, as would be expected for a random sequence.
    Tests basic randomness at the bit level.

Test 2 – Frequency Test within a Block
    Divides the sequence into blocks and tests whether the frequency of ones
    in each block is approximately 50%, as expected for random data.

Test 5 – Runs Test
    Determines whether the frequency of runs of ones and zeros of various
    lengths is as expected for a random sequence.

Reference: NIST SP 800-22 Rev 1a, Sections 2.1, 2.2, and 2.3
"""

import argparse
import math
import re
from dataclasses import dataclass


ALPHA = 0.01
EPSILON = 1e-14
MAX_ITERATIONS = 1000


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    passed: bool
    details: str


def regularized_gamma_q(a: float, x: float) -> float:
    if a <= 0:
        raise ValueError("Parameter 'a' must be positive.")
    if x < 0:
        raise ValueError("Parameter 'x' must be non-negative.")
    if x == 0:
        return 1.0

    if x < a + 1.0:
        return 1.0 - _regularized_gamma_p_series(a, x)
    return _regularized_gamma_q_fraction(a, x)


def _regularized_gamma_p_series(a: float, x: float) -> float:
    gln = math.lgamma(a)
    term = 1.0 / a
    total = term

    for n in range(1, MAX_ITERATIONS + 1):
        term *= x / (a + n)
        total += term
        if abs(term) < abs(total) * EPSILON:
            break

    return total * math.exp(-x + a * math.log(x) - gln)


def _regularized_gamma_q_fraction(a: float, x: float) -> float:
    gln = math.lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / max(b, tiny)
    h = d

    for i in range(1, MAX_ITERATIONS + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPSILON:
            break

    return math.exp(-x + a * math.log(x) - gln) * h


def bits_from_binary(raw_value: str) -> str:
    bits = re.sub(r"[\s_]+", "", raw_value)
    if not bits:
        raise ValueError("Binary input is empty.")
    if re.search(r"[^01]", bits):
        raise ValueError("Binary input must contain only 0 and 1.")
    return bits


def bits_from_text(raw_value: str) -> str:
    if not raw_value:
        raise ValueError("Text input is empty.")
    return "".join(f"{byte:08b}" for byte in raw_value.encode("utf-8"))


def bits_from_hex(raw_value: str) -> str:
    hex_value = re.sub(r"[\s_]+", "", raw_value).lower()
    if hex_value.startswith("0x"):
        hex_value = hex_value[2:]
    if not hex_value:
        raise ValueError("Hex input is empty.")
    if re.search(r"[^0-9a-f]", hex_value):
        raise ValueError("Hex input must contain only hexadecimal digits.")
    return "".join(f"{int(symbol, 16):04b}" for symbol in hex_value)


def frequency_monobit_test(bits: str, alpha: float) -> TestResult:
    n = len(bits)
    total = sum(1 if bit == "1" else -1 for bit in bits)
    s_obs = abs(total) / math.sqrt(n)
    p_value = math.erfc(s_obs / math.sqrt(2.0))

    return TestResult(
        name="Frequency (Monobit) Test",
        statistic=s_obs,
        p_value=p_value,
        passed=p_value >= alpha,
        details=f"n={n}, S_n={total}",
    )


def frequency_block_test(bits: str, block_size: int, alpha: float) -> TestResult:
    n = len(bits)
    if block_size <= 0:
        raise ValueError("Block size must be positive.")

    block_count = n // block_size
    if block_count == 0:
        raise ValueError("Block size is larger than the bit sequence.")

    chi_square = 0.0
    for block_index in range(block_count):
        start = block_index * block_size
        block = bits[start : start + block_size]
        ones_ratio = block.count("1") / block_size
        chi_square += (ones_ratio - 0.5) ** 2

    chi_square *= 4.0 * block_size
    p_value = regularized_gamma_q(block_count / 2.0, chi_square / 2.0)

    return TestResult(
        name="Frequency Test within a Block",
        statistic=chi_square,
        p_value=p_value,
        passed=p_value >= alpha,
        details=f"block_size={block_size}, blocks={block_count}, ignored_bits={n % block_size}",
    )


def runs_test(bits: str, alpha: float) -> TestResult:
    n = len(bits)
    ones_ratio = bits.count("1") / n
    prerequisite_limit = 2.0 / math.sqrt(n)

    if abs(ones_ratio - 0.5) >= prerequisite_limit:
        return TestResult(
            name="Runs Test",
            statistic=0.0,
            p_value=0.0,
            passed=False,
            details=(
                f"pi={ones_ratio:.6f}, prerequisite failed: "
                f"|pi - 0.5| must be < {prerequisite_limit:.6f}"
            ),
        )

    runs = 1 + sum(1 for index in range(1, n) if bits[index] != bits[index - 1])
    expected_runs = 2.0 * n * ones_ratio * (1.0 - ones_ratio)
    denominator = 2.0 * math.sqrt(2.0 * n) * ones_ratio * (1.0 - ones_ratio)
    p_value = math.erfc(abs(runs - expected_runs) / denominator)

    return TestResult(
        name="Runs Test",
        statistic=float(runs),
        p_value=p_value,
        passed=p_value >= alpha,
        details=f"pi={ones_ratio:.6f}, V_n={runs}, expected_runs={expected_runs:.3f}",
    )


def suggest_block_size(bit_count: int) -> int:
    return max(1, min(128, bit_count // 10))


def analyze_bits(bits: str, block_size: int | None, alpha: float) -> list[TestResult]:
    if not bits:
        raise ValueError("Bit sequence is empty.")

    selected_block_size = block_size if block_size is not None else suggest_block_size(len(bits))
    return [
        frequency_monobit_test(bits, alpha),
        frequency_block_test(bits, selected_block_size, alpha),
        runs_test(bits, alpha),
    ]


def print_report(bits: str, results: list[TestResult], alpha: float) -> None:
    ones = bits.count("1")
    zeros = len(bits) - ones

    print("\n=== Randomness Test Report ===")
    print(f"Bits: {len(bits)} | zeros: {zeros} | ones: {ones} | alpha: {alpha}")
    print("-" * 72)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.name}")
        print(f"  status:    {status}")
        print(f"  statistic: {result.statistic:.6f}")
        print(f"  p-value:   {result.p_value:.6f}")
        print(f"  details:   {result.details}")
        print("-" * 72)

    passed_count = sum(result.passed for result in results)
    print(f"Summary: {passed_count}/{len(results)} tests passed.")


def read_interactive_input() -> tuple[str, int | None, float]:
    print("Cryptographic Randomness Tests")
    print("1 - binary bits, example: 0101011100")
    print("2 - text, converted to UTF-8 bits")
    print("3 - hex, example: 3f a9 00")

    mode = input("Choose input type [1/2/3]: ").strip()
    raw_value = input("Enter value: ")

    if mode == "1":
        bits = bits_from_binary(raw_value)
    elif mode == "2":
        bits = bits_from_text(raw_value)
    elif mode == "3":
        bits = bits_from_hex(raw_value)
    else:
        raise ValueError("Unknown input type.")

    default_block_size = suggest_block_size(len(bits))
    block_size_value = input(f"Block size [{default_block_size}]: ").strip()
    block_size = int(block_size_value) if block_size_value else default_block_size

    alpha_value = input(f"Significance level alpha [{ALPHA}]: ").strip()
    alpha = float(alpha_value) if alpha_value else ALPHA

    return bits, block_size, alpha


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Console app for Frequency Monobit, Block Frequency, and Runs tests."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bits", help="Binary sequence with 0 and 1.")
    group.add_argument("--text", help="Text converted to UTF-8 bits.")
    group.add_argument("--hex", help="Hexadecimal data converted to bits.")
    parser.add_argument("--block-size", type=int, help="Block size for the block frequency test.")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Significance level.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.bits is not None:
            bits = bits_from_binary(args.bits)
            block_size = args.block_size
            alpha = args.alpha
        elif args.text is not None:
            bits = bits_from_text(args.text)
            block_size = args.block_size
            alpha = args.alpha
        elif args.hex is not None:
            bits = bits_from_hex(args.hex)
            block_size = args.block_size
            alpha = args.alpha
        else:
            bits, block_size, alpha = read_interactive_input()

        results = analyze_bits(bits, block_size, alpha)
        print_report(bits, results, alpha)
    except ValueError as error:
        print(f"Error: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
