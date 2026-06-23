import math
import random

def compute_rank(rows, M, Q):
    rank = 0
    for col in range(Q):
        pivot = None
        for r in range(rank, M):
            if (rows[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(M):
            if r != rank and ((rows[r] >> col) & 1):
                rows[r] ^= rows[rank]
        rank += 1
        if rank == M:
            break
    return rank


def binary_matrix_rank_test(bitstring, M=32, Q=32):
    n = len(bitstring)
    block_size = M * Q
    N = n // block_size
    if N == 0:
        raise ValueError(f"Нужно хотя бы {block_size} бит, получено {n}")

    full_rank = 0          # FM
    rank_minus_1 = 0       # FM-1
    # остальные будут N - full_rank - rank_minus_1

    for block_idx in range(N):
        start = block_idx * block_size
        rows = []
        for r in range(M):
            row_start = start + r * Q
            row_bits = bitstring[row_start:row_start + Q]
            val = int(row_bits, 2) if row_bits else 0
            rows.append(val)
        rnk = compute_rank(rows, M, Q)
        if rnk == M:
            full_rank += 1
        elif rnk == M - 1:
            rank_minus_1 += 1

    rest = N - full_rank - rank_minus_1

    # Ожидаемые вероятности для M=Q=32
    p_full = 0.2888
    p_almost = 0.5776
    p_rest = 0.1336

    exp_full = p_full * N
    exp_almost = p_almost * N
    exp_rest = p_rest * N

    chi2 = ((full_rank - exp_full) ** 2 / exp_full +
            (rank_minus_1 - exp_almost) ** 2 / exp_almost +
            (rest - exp_rest) ** 2 / exp_rest)

    p_value = math.exp(-chi2 / 2.0)  
    stats = {
        'N': N,
        'FM': full_rank,
        'FM_minus_1': rank_minus_1,
        'rest': rest,
        'chi2': chi2,
        'exp_full': exp_full,
        'exp_almost': exp_almost,
        'exp_rest': exp_rest
    }
    return p_value, stats


def main():
    print("=" * 60)
    print("ТЕСТ РАНГА БИНАРНЫХ МАТРИЦ (NIST)")
    print("=" * 60)
    print("Этот тест проверяет линейную зависимость между битами.\n")

    mode = input("Введите '1' для ввода своей битовой строки, '2' для генерации случайной: ")
    if mode == '1':
        bits = input("Введите битовую строку (только 0 и 1): ").strip()
        if not all(c in '01' for c in bits):
            print("Ошибка: строка должна состоять только из 0 и 1.")
            return
    elif mode == '2':
        length = input("Введите желаемую длину битовой строки (рекомендуется >= 38912): ")
        try:
            length = int(length)
        except ValueError:
            print("Ошибка: нужно целое число.")
            return
        bits = ''.join(random.choice('01') for _ in range(length))
        print(f"Сгенерировано {length} случайных бит (первые 100: {bits[:100]}...)")
    else:
        print("Неверный режим.")
        return

    M = 32
    Q = 321

    try:
        p_val, stats = binary_matrix_rank_test(bits, M, Q)
    except ValueError as e:
        print("Ошибка:", e)
        return

    print("\n----- РЕЗУЛЬТАТЫ -----")
    print(f"Количество матриц (N)        : {stats['N']}")
    print(f"Полный ранг (FM)             : {stats['FM']}")
    print(f"Ранг = M-1 (FM-1)            : {stats['FM_minus_1']}")
    print(f"Остальные (ранг <= M-2)      : {stats['rest']}")
    print(f"Ожидаемое полного ранга      : {stats['exp_full']:.2f}")
    print(f"Ожидаемое ранга M-1          : {stats['exp_almost']:.2f}")
    print(f"Ожидаемое остальных          : {stats['exp_rest']:.2f}")
    print(f"Хи-квадрат                   : {stats['chi2']:.6f}")
    print(f"P-значение                   : {p_val:.6f}")

    if p_val < 0.01:
        print("\nВЫВОД: последовательность НЕСЛУЧАЙНА (отвергаем гипотезу случайности на уровне 1%)")
    else:
        print("\nВЫВОД: последовательность выглядит СЛУЧАЙНОЙ (не отвергаем гипотезу случайности на уровне 1%)")


if __name__ == "__main__":
    main()