"""
CallCenterOpt - Оптимизатор штата колл-центра
Румянцев В.И. | Курс "Теория игр" | Итоговый проект

Использует теорию массового обслуживания M/M/c и критерии принятия решений
для рекомендации оптимального количества операторов при неопределенной нагрузке.
"""

import sys
import argparse
from pathlib import Path

import yaml
import numpy as np
from pydantic import ValidationError

from visualize import plot_sla_landscape, plot_cost_matrix
from core.config_schema import CallCenterConfig
from core.queue_math import calculate_mm_c_metrics
from core.game_solver import (
    build_cost_matrix,
    apply_wald,
    apply_hurwicz,
    apply_bayes,
    apply_savage,
)


def load_config(path: str) -> CallCenterConfig:
    """Загрузка YAML конфигурации и валидация через Pydantic."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    # Если файл пустой или не YAML объект
    if not isinstance(raw_data, dict):
        raise ValueError("Конфигурационный файл должен содержать YAML-объект (словарь).")

    try:
        # Валидируем данные и возвращаем объект схемы
        return CallCenterConfig(**raw_data)
    except ValidationError as e:
        print(f"\n[Ошибка Валидации Конфигурации] Обнаружены ошибки в {path}:", file=sys.stderr)
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            print(f"  - {loc}: {error['msg']}", file=sys.stderr)
        sys.exit(1)


def print_cost_table(strategies, lambdas, matrix, best_idx=None, title="Матрица затрат"):
    """Вывод матрицы затрат в виде таблицы."""
    header = f"{'c':>4} |" + "".join(f" lam={int(l):>4} |" for l in lambdas)
    print(f"\n{title}:")
    print(header)
    print("—" * len(header))
    for i, c in enumerate(strategies):
        row = f"{c:>4} |" + "".join(
            f" {matrix[i, j]:>8.1f} |" for j in range(len(lambdas))
        )
        marker = " <--" if best_idx is not None and i == best_idx else ""
        print(row + marker)


def print_sla_details(lambda_val, mu, c, sla_threshold):
    """Проверка SLA с подробным выводом."""
    m = calculate_mm_c_metrics(lambda_val, mu, c)
    status = "OK" if m.stable and m.p_wait <= sla_threshold else "НАРУШЕНО"
    print(f"\n  Проверка SLA при lambda = {lambda_val:.1f}, c = {c}:")
    print(f"  Загрузка rho            = {m.rho:.3f}")
    print(f"  Вер. ожидания P_wait    = {m.p_wait:.3f}  (порог: {sla_threshold})")
    print(f"  Длина очереди L_q       = {m.l_q:.3f}")
    print(f"  Время ожидания W        = {m.w * 60:.1f} мин")
    print(f"  Статус SLA              = {status}")


def print_sla_landscape(lambdas, strategies, mu, sla_threshold):
    """Вывод таблицы соблюдения SLA для всех стратегий и сценариев."""
    print(f"\nПолная картина соблюдения SLA (P_wait <= {sla_threshold:.2f}):")
    header = f"  {'c':>3} |" + "".join(f" lam={int(l):>3} |" for l in lambdas)
    print(header)
    print("  " + "—" * (len(header) - 2))
    for c in strategies:
        row = f"  {c:>3} |"
        for lam in lambdas:
            m = calculate_mm_c_metrics(lam, mu, c)
            ok = "OK   " if (m.stable and m.p_wait <= sla_threshold) else "FAIL "
            row += f"  {ok}  |"
        print(row)


def print_regret_matrix(matrix, strategies, lambdas):
    """Вывод матрицы сожалений (для критерия Сэвиджа)."""
    min_per_scenario = matrix.min(axis=0, keepdims=True)
    regret = matrix - min_per_scenario
    print_cost_table(strategies, lambdas, regret, title="Матрица сожалений")


def run(cfg: CallCenterConfig, criterion: str = None, show_all: bool = False, plot: bool = False) -> None:
    """Основной анализ."""
    lambdas = cfg.lambda_scenarios
    strategies = cfg.strategies
    mu = cfg.mu
    salary = cfg.salary_per_operator
    penalty = cfg.penalty_factor
    penalty_lq = cfg.penalty_lq
    alpha = cfg.alpha
    probs = cfg.probabilities
    sla = cfg.sla_p_wait
    criterion = criterion or cfg.criterion

    print("\n" + "=" * 45)
    print("CallCenterOpt - Оптимизатор штата колл-центра")
    print("Румянцев В.И. | Итоговый проект по теории игр")
    print("=" * 45)

    # Построение матрицы затрат с учётом penalty_lq
    matrix = build_cost_matrix(lambdas, strategies, mu, salary, penalty, penalty_lq)

    # Если нужно показать все критерии
    if show_all:
        criteria_to_run = ["wald", "hurwicz", "bayes", "savage"]
    else:
        criteria_to_run = [criterion]

    # Словарь для хранения результатов (для сравнения)
    results = {}

    for crit in criteria_to_run:
        print(f"\n{'='*45}")
        if crit == "wald":
            idx, val = apply_wald(matrix)
            label = "Вальд (минимакс / пессимистический)"
        elif crit == "hurwicz":
            idx, val = apply_hurwicz(matrix, alpha)
            label = f"Гурвиц (alpha={alpha})"
        elif crit == "bayes":
            idx, val = apply_bayes(matrix, probs)
            label = f"Байес (E[затраты], вероятности={probs})"
        elif crit == "savage":
            idx, val = apply_savage(matrix)
            label = "Сэвидж (минимакс сожаления)"
        else:
            print(f"Неизвестный критерий: {crit}")
            continue

        best_c = strategies[idx]
        print(f"\nКритерий: {label}")
        print_cost_table(strategies, lambdas, matrix, best_idx=idx)

        # Для Сэвиджа дополнительно покажем матрицу сожалений
        if crit == "savage":
            print_regret_matrix(matrix, strategies, lambdas)

        print(f"\nРекомендация: c = {best_c} операторов")
        print(f"Значение критерия: {val:.2f}")
        # Для среднего сценария (или для каждого) покажем SLA
        avg_lambda = float(np.mean(lambdas))
        print_sla_details(avg_lambda, mu, best_c, sla)

        results[crit] = {"c": best_c, "value": val}

    # Вывод общей SLA-картины
    print_sla_landscape(lambdas, strategies, mu, sla)

    # Если запрошена визуализация
    if plot:
        try:
            # Построим графики
            plot_sla_landscape(lambdas, strategies, mu, sla, save_path='reports/sla_landscape.png')
            plot_cost_matrix(matrix, strategies, lambdas, save_path='reports/cost_matrix.png')
            print("\nГрафики сохранены в папку reports/")
        except ImportError as e:
            print(f"\nВизуализация не доступна: {e}. Установите matplotlib.")
        except Exception as e:
            print(f"\nОшибка при построении графиков: {e}")

    # Сравнение результатов всех критериев (если show_all)
    if show_all:
        print("\n" + "="*50)
        print("Сводка рекомендаций по критериям:")
        for crit, res in results.items():
            print(f"  {crit.capitalize():8}: c = {res['c']:2d}, значение = {res['value']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="CallCenterOpt - оптимизатор штата")
    parser.add_argument(
        "--config", default="config/scenario.yaml", help="Путь к YAML конфигурации"
    )
    parser.add_argument(
        "--criterion",
        choices=["wald", "hurwicz", "bayes", "savage"],
        help="Переопределить критерий"
    )
    parser.add_argument(
        "--all", action="store_true", help="Сравнить все четыре критерия"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Построить графики (требуется matplotlib)"
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        run(cfg, criterion=args.criterion, show_all=args.all, plot=args.plot)
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
