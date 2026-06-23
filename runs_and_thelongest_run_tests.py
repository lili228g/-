"""
NIST SP 800-22 Rev 1a
Тест 3: Runs Test (Тест на серии)
Тест 4: Test for the Longest Run of Ones in a Block (Тест на длиннейшую серию единиц в блоке)

Использованные библиотеки: argparse, math, re, dataclasses, numpy
"""

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import erfc, gammaincc  # igamc = gammaincc


# ─────────────────────────────────────────────────────────────
#  Структуры данных
# ─────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """Результат одного статистического теста NIST."""
    test_name: str
    p_value: float
    passed: bool
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS  ✓" if self.passed else "FAIL  ✗"
        lines = [
            f"\n{'─' * 60}",
            f"  {self.test_name}",
            f"{'─' * 60}",
            f"  P-value : {self.p_value:.6f}",
            f"  Статус  : {status}  (порог α = 0.01)",
        ]
        for k, v in self.details.items():
            lines.append(f"  {k:<20}: {v}")
        lines.append(f"{'─' * 60}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  Утилиты
# ─────────────────────────────────────────────────────────────

def parse_bitstring(raw: str) -> np.ndarray:
    """
    Принимает строку вида '0110...' или '0x1A2B...' (hex).
    Возвращает numpy-массив из 0 и 1 (dtype=int8).
    """
    raw = raw.strip()
    # Шестнадцатеричный формат
    if re.fullmatch(r'(0x|0X)?[0-9A-Fa-f]+', raw):
        hex_str = re.sub(r'^0[xX]', '', raw)
        bits = bin(int(hex_str, 16))[2:]          # '0b...' → убираем '0b'
        bits = bits.zfill(len(hex_str) * 4)       # дополняем до полных байт
    elif re.fullmatch(r'[01]+', raw):
        bits = raw
    else:
        raise ValueError(
            "Неверный формат входных данных. Ожидается строка из 0/1 "
            "или шестнадцатеричное число (опционально с префиксом 0x)."
        )
    return np.array(list(bits), dtype=np.int8)


def load_bits_from_file(path: str) -> np.ndarray:
    """Читает битовую строку из файла (первая непустая строка)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                return parse_bitstring(line)
    raise ValueError(f"Файл '{path}' пуст или не содержит битовой строки.")


# ─────────────────────────────────────────────────────────────
#  Тест 3: Runs Test  (NIST SP 800-22 §2.3)
# ─────────────────────────────────────────────────────────────

def runs_test(bits: np.ndarray, alpha: float = 0.01) -> TestResult:
    """
    Тест на серии (Runs Test) — NIST SP 800-22 Rev 1a, Section 2.3.

    Алгоритм:
        1. Вычислить долю единиц π = Σ εj / n.
        2. Предварительная проверка: если |π − 1/2| ≥ τ = 2/√n,
           тест неприменим (провал Frequency Test), P-value = 0.
        3. Vn(obs) = 1 + Σ r(k), где r(k) = 1, если εk ≠ εk+1.
        4. P-value = erfc( |Vn(obs) − 2nπ(1−π)| / (2√(2n)·π·(1−π)) ).
    """
    n = len(bits)
    if n < 100:
        raise ValueError(f"Длина последовательности {n} < 100 (рекомендуется n ≥ 100).")

    # Шаг 1: доля единиц
    pi_hat = float(np.sum(bits)) / n

    # Шаг 2: предварительная проверка
    tau = 2.0 / math.sqrt(n)
    freq_check_passed = abs(pi_hat - 0.5) < tau

    if not freq_check_passed:
        return TestResult(
            test_name="Тест 3: Runs Test",
            p_value=0.0,
            passed=False,
            details={
                "n": n,
                "π (доля единиц)": f"{pi_hat:.6f}",
                "τ": f"{tau:.6f}",
                "Предпроверка": "НЕ пройдена — тест неприменим",
            },
        )

    # Шаг 3: число серий Vn(obs)
    # r(k) = 1 если εk ≠ εk+1, иначе 0
    transitions = np.sum(bits[:-1] != bits[1:])
    vn_obs = int(transitions) + 1

    # Шаг 4: P-value через erfc
    numerator = abs(vn_obs - 2.0 * n * pi_hat * (1.0 - pi_hat))
    denominator = 2.0 * math.sqrt(2.0 * n) * pi_hat * (1.0 - pi_hat)
    p_value = float(erfc(numerator / denominator))

    return TestResult(
        test_name="Тест 3: Runs Test",
        p_value=p_value,
        passed=p_value >= alpha,
        details={
            "n": n,
            "π (доля единиц)": f"{pi_hat:.6f}",
            "τ": f"{tau:.6f}",
            "Предпроверка": "пройдена",
            "Vn(obs) — число серий": vn_obs,
            "E[Vn] = 2nπ(1−π)": f"{2*n*pi_hat*(1-pi_hat):.4f}",
        },
    )


# ─────────────────────────────────────────────────────────────
#  Тест 4: Longest Run of Ones in a Block  (NIST SP 800-22 §2.4)
# ─────────────────────────────────────────────────────────────

# Предустановленные параметры и теоретические вероятности (таблица §3.4)
_LONGEST_RUN_PARAMS = {
    # M: (K, N, min_run_lower_bound, [πi], [границы классов (правые)])
    8: {
        "K": 3, "N": 16,
        "pi": [0.2148, 0.3672, 0.2305, 0.1875],
        "bounds": [1, 2, 3, 4],   # ν≤1, ν=2, ν=3, ν≥4
    },
    128: {
        "K": 5, "N": 49,
        "pi": [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124],
        "bounds": [4, 5, 6, 7, 8, 9],  # ν≤4, ν=5,...,ν≥9
    },
    # M=10000 (в документе обозначено как 10^4)
    10000: {
        "K": 6, "N": 75,
        "pi": [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727],
        "bounds": [10, 11, 12, 13, 14, 15, 16],  # ν≤10,...,ν≥16
    },
}

# Псевдонимы: 128 используется для n ≥ 6272, 10000 — для n ≥ 750 000
def _select_M(n: int) -> int:
    if n >= 750_000:
        return 10_000
    elif n >= 6_272:
        return 128
    elif n >= 128:
        return 8
    else:
        raise ValueError(
            f"Длина последовательности {n} слишком мала. "
            "Минимум: n ≥ 128 для M=8."
        )


def _longest_run_of_ones(block: np.ndarray) -> int:
    """Длина наибольшей серии единиц в блоке (через numpy)."""
    if np.sum(block) == 0:
        return 0
    # Добавляем нули-разделители по краям, находим группы
    padded = np.concatenate(([0], block, [0]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return int(np.max(ends - starts))


def _classify(run_len: int, bounds: list[int]) -> int:
    """
    Определяет класс ν_i по длине серии и таблице границ.
    bounds[i] — правая граница i-го класса;
    последний класс — всё, что ≥ bounds[-2]+1.
    """
    for i, b in enumerate(bounds[:-1]):
        if run_len <= b:
            return i
    return len(bounds) - 1


def longest_run_test(bits: np.ndarray, alpha: float = 0.01) -> TestResult:
    """
    Тест на длиннейшую серию единиц в блоке — NIST SP 800-22, Section 2.4.

    Алгоритм:
        1. Определить M по длине n; взять соответствующие K, N, πi.
        2. Разбить последовательность на N блоков по M бит.
        3. Для каждого блока найти длину наибольшей серии единиц.
        4. Подсчитать частоты νi по K+1 классам.
        5. χ²(obs) = Σ (νi − N·πi)² / (N·πi).
        6. P-value = igamc(K/2, χ²(obs)/2) = gammaincc(K/2, χ²(obs)/2).
    """
    n = len(bits)
    M = _select_M(n)
    params = _LONGEST_RUN_PARAMS[M]
    K: int = params["K"]
    N: int = params["N"]
    pi_vals: list[float] = params["pi"]
    bounds: list[int] = params["bounds"]

    # Шаг 2: блоки (берём только первые N·M бит)
    blocks = bits[: N * M].reshape(N, M)

    # Шаг 3–4: длины наибольших серий единиц → классы → частоты
    freq = np.zeros(K + 1, dtype=np.int64)
    max_runs = np.array([_longest_run_of_ones(blocks[j]) for j in range(N)])
    for run_len in max_runs:
        freq[_classify(run_len, bounds)] += 1

    # Шаг 5: χ²(obs)
    pi_arr = np.array(pi_vals)
    expected = N * pi_arr
    chi2_obs = float(np.sum((freq - expected) ** 2 / expected))

    # Шаг 6: P-value = igamc(K/2, χ²/2) ≡ gammaincc(K/2, χ²/2)
    p_value = float(gammaincc(K / 2.0, chi2_obs / 2.0))

    # Формируем строки классов для вывода
    class_labels = []
    for i, b in enumerate(bounds):
        if i == 0:
            class_labels.append(f"ν ≤ {b}")
        elif i == len(bounds) - 1:
            class_labels.append(f"ν ≥ {b}")
        else:
            class_labels.append(f"ν = {b}")

    freq_detail = {f"Класс {class_labels[i]}": f"ν_i={int(freq[i])}, E={expected[i]:.4f}"
                   for i in range(K + 1)}

    details = {
        "n": n,
        "M (размер блока)": M,
        "K (число классов − 1)": K,
        "N (число блоков)": N,
        "χ²(obs)": f"{chi2_obs:.6f}",
        **freq_detail,
    }

    return TestResult(
        test_name="Тест 4: Longest Run of Ones in a Block",
        p_value=p_value,
        passed=p_value >= alpha,
        details=details,
    )


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nist_tests_3_4",
        description=(
            "NIST SP 800-22 Rev 1a — Тесты 3 и 4:\n"
            "  3. Runs Test (Тест на серии)\n"
            "  4. Longest Run of Ones in a Block\n\n"
            "Входные данные: строка из 0/1 или шестнадцатеричное число."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "-b", "--bits",
        metavar="BITSTRING",
        help="Битовая строка (только символы 0 и 1) или hex-число (с 0x или без).",
    )
    src.add_argument(
        "-f", "--file",
        metavar="FILE",
        help="Путь к файлу, содержащему битовую строку в первой строке.",
    )
    src.add_argument(
        "--random",
        metavar="N",
        type=int,
        help="Сгенерировать N случайных бит (криптографически стойкий ГПСЧ, os.urandom).",
    )

    parser.add_argument(
        "--test",
        choices=["3", "4", "all"],
        default="all",
        help="Запустить только тест 3, только тест 4 или оба (по умолчанию: all).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Уровень значимости α (по умолчанию: 0.01).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Получаем биты ──────────────────────────────────────────
    if args.bits is not None:
        bits = parse_bitstring(args.bits)
    elif args.file is not None:
        bits = load_bits_from_file(args.file)
    else:  # --random N
        import os
        n_bytes = math.ceil(args.random / 8)
        raw_bytes = os.urandom(n_bytes)
        bitstr = bin(int.from_bytes(raw_bytes, "big"))[2:].zfill(n_bytes * 8)
        bits = np.array(list(bitstr[: args.random]), dtype=np.int8)

    print(f"\n{'═' * 60}")
    print(f"  NIST SP 800-22 — Тесты 3 и 4")
    print(f"  Длина входной последовательности: {len(bits)} бит")
    print(f"  Уровень значимости α = {args.alpha}")
    print(f"{'═' * 60}")

    results: list[TestResult] = []

    if args.test in ("3", "all"):
        try:
            res3 = runs_test(bits, alpha=args.alpha)
        except ValueError as exc:
            print(f"\n[Тест 3] Ошибка: {exc}")
        else:
            results.append(res3)
            print(res3)

    if args.test in ("4", "all"):
        try:
            res4 = longest_run_test(bits, alpha=args.alpha)
        except ValueError as exc:
            print(f"\n[Тест 4] Ошибка: {exc}")
        else:
            results.append(res4)
            print(res4)

    # Итоговая сводка
    if results:
        passed = sum(1 for r in results if r.passed)
        print(f"\n  Итог: {passed}/{len(results)} тест(ов) пройдено.\n")


if __name__ == "__main__":
    main()