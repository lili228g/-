import math
import random
import sys

try:
    import numpy as np
    from numpy.fft import fft
except ImportError:
    print("Для этого теста требуется библиотека numpy. Установите: pip install numpy")
    sys.exit(1)


def spectral_test(bitstring):
    """
    Выполняет спектральный тест (NIST SP 800-22, раздел 2.6).
    Возвращает (p_value, словарь со статистиками).
    """
    n = len(bitstring)

    # Шаг 1: преобразуем биты 0/1 в -1/+1
    x = [1 if b == '1' else -1 for b in bitstring]

    # Шаг 2: дискретное преобразование Фурье
    S = fft(x)

    # Шаг 3: берём модули первых n//2 частот (спектр симметричен)
    half = n // 2
    magnitudes = [abs(S[i]) for i in range(half)]

    # Шаг 4: порог T для 95% пиков
    T = math.sqrt(n * math.log(1 / 0.05))   # ln(20)

    # Шаг 5: ожидаемое число пиков ниже T
    N0 = 0.95 * n / 2.0

    # Шаг 6: наблюдаемое число пиков ниже T
    N1 = sum(1 for m in magnitudes if m < T)

    # Шаг 7: статистика d
    denominator = math.sqrt(n * 0.95 * 0.05 / 4.0)
    d = (N1 - N0) / denominator

    # Шаг 8: P-значение (erfc от |d|/sqrt(2))
    p_value = math.erfc(abs(d) / math.sqrt(2))

    stats = {
        'N1': N1,
        'N0': N0,
        'd': d,
        'T': T,
        'half': half
    }
    return p_value, stats


def main():
    print("=" * 60)
    print("СПЕКТРАЛЬНЫЙ ТЕСТ (DFT) – NIST")
    print("=" * 60)
    print("Проверяет наличие периодических закономерностей в битовой строке.\n")

    mode = input("Введите '1' для ввода своей строки, '2' для генерации случайной: ")
    if mode == '1':
        bits = input("Введите битовую строку (только 0 и 1): ").strip()
        if not all(c in '01' for c in bits):
            print("Ошибка: строка должна состоять только из 0 и 1.")
            return
    elif mode == '2':
        length = input("Введите длину (рекомендуется ≥ 1000): ")
        try:
            length = int(length)
        except ValueError:
            print("Ошибка: введите целое число.")
            return
        bits = ''.join(random.choice('01') for _ in range(length))
        print(f"Сгенерировано {length} бит.")
    else:
        print("Неверный режим.")
        return

    n = len(bits)
    if n < 2:
        print("Ошибка: слишком мало бит (нужно хотя бы 2).")
        return
    if n < 1000:
        print("Предупреждение: для надёжности рекомендуется n ≥ 1000.")

    try:
        p_val, stats = spectral_test(bits)
    except Exception as e:
        print("Ошибка выполнения теста:", e)
        return

    print("\n----- РЕЗУЛЬТАТЫ ТЕСТА -----")
    print(f"Длина последовательности (n)   : {n}")
    print(f"Порог T                        : {stats['T']:.4f}")
    print(f"Наблюдаемое число пиков < T (N1): {stats['N1']}")
    print(f"Ожидаемое число пиков < T (N0)  : {stats['N0']:.2f}")
    print(f"Статистика d                   : {stats['d']:.6f}")
    print(f"P-значение                     : {p_val:.6f}")

    if p_val < 0.01:
        print("\nВЫВОД: последовательность НЕСЛУЧАЙНА (отвергаем на уровне 1%)")
    else:
        print("\nВЫВОД: последовательность выглядит СЛУЧАЙНОЙ (не отвергаем на уровне 1%)")


if __name__ == "__main__":
    main()