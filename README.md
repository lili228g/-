# NIST SP 800-22 Tests 3 and 4

Простой Python-скрипт для запуска NIST теста на серии и теста на длинейшую серию единиц в блоке.

## Установка зависимостей

```powershell
python -m pip install -r requirements.txt
```

## Запуск

```powershell
python runs_and_thelongest_run_tests.py --random 128
```

или

```powershell
python runs_and_thelongest_run_tests.py --bits 00110101... --test all
```

## Доступные параметры

- `--bits` — строка бит `0/1` или hex-число.
- `--file` — файл с битовой строкой.
- `--random N` — сгенерировать `N` случайных бит.
- `--test 3|4|all` — выбрать тест.
- `--alpha` — уровень значимости (по умолчанию `0.01`).
