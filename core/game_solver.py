"""
Модуль игр: принятие решений в условиях неопределенности.

Классические критерии:
  - Вальд (максимин): пессимистический
  - Гурвиц (альфа-критерий): взвешенный оптимизм/пессимизм
  - Байес (ожидаемое значение): использует вероятности сценариев
  - Сэвидж (минимакс сожаления): минимизация максимального упущенного выигрыша

Дополнительно: функция затрат может включать штраф за длину очереди L_q.
"""

import numpy as np
from typing import List, Tuple

from core.queue_math import calculate_mm_c_metrics


def build_cost_matrix(
        lambda_scenarios: List[float],
        strategies: List[int],
        mu: float,
        salary_per_op: float,
        penalty_factor: float,
        penalty_lq: float = 0.0,
) -> np.ndarray:
    """
    Построение матрицы затрат для игры с природой.

    Затраты = c * salary_per_op + penalty_factor * P_wait + penalty_lq * L_q.

    Args:
        lambda_scenarios: список интенсивностей нагрузки (сценарии природы)
        strategies: список возможных количеств операторов
        mu: производительность одного оператора
        salary_per_op: зарплата оператора (тыс. руб.)
        penalty_factor: штраф за единицу вероятности ожидания
        penalty_lq: штраф за единицу длины очереди (опционально)

    Returns:
        матрица затрат размером (len(strategies) x len(lambda_scenarios))
    """
    rows = len(strategies)
    cols = len(lambda_scenarios)
    matrix = np.zeros((rows, cols))

    for i, c in enumerate(strategies):
        for j, lam in enumerate(lambda_scenarios):
            metrics = calculate_mm_c_metrics(lam, mu, c)
            # Если система нестабильна, L_q = inf, тогда затраты становятся бесконечными
            # Это приведёт к тому, что такие стратегии не будут выбраны.
            penalty_lq_term = penalty_lq * metrics.l_q if metrics.stable else np.inf
            penalty_pwait_term = penalty_factor * metrics.p_wait
            matrix[i, j] = c * salary_per_op + penalty_pwait_term + penalty_lq_term

    return matrix


def apply_wald(matrix: np.ndarray) -> Tuple[int, float]:
    """Критерий Вальда: выбор стратегии с минимальными затратами в худшем случае."""
    worst_case = matrix.max(axis=1)
    best_idx = int(np.argmin(worst_case))
    return best_idx, float(worst_case[best_idx])


def apply_hurwicz(matrix: np.ndarray, alpha: float) -> Tuple[int, float]:
    """
    Критерий Гурвица: H(стратегия) = alpha * лучший случай + (1 - alpha) * худший случай.
    alpha = 1 → оптимист, alpha = 0 → пессимист (Вальд).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha должно быть в [0, 1]")
    best_case = matrix.min(axis=1)
    worst_case = matrix.max(axis=1)
    hurwicz = alpha * best_case + (1.0 - alpha) * worst_case
    best_idx = int(np.argmin(hurwicz))
    return best_idx, float(hurwicz[best_idx])


def apply_bayes(matrix: np.ndarray, probabilities: List[float]) -> Tuple[int, float]:
    """Критерий Байеса: минимизация ожидаемых затрат."""
    probs = np.array(probabilities, dtype=float)
    if not np.isclose(probs.sum(), 1.0, atol=1e-3):
        raise ValueError("Вероятности сценариев должны в сумме давать 1.0")
    if np.any(probs < 0):
        raise ValueError("Все вероятности должны быть неотрицательными")

    expected = matrix @ probs
    best_idx = int(np.argmin(expected))
    return best_idx, float(expected[best_idx])


def apply_savage(matrix: np.ndarray) -> Tuple[int, float]:
    """
    Критерий Сэвиджа (минимакс сожаления).

    Для каждого сценария (столбца) вычисляется минимальная стоимость среди всех стратегий.
    Сожаление = стоимость стратегии - минимальная стоимость в этом сценарии.
    Затем для каждой стратегии берётся максимальное сожаление.
    Выбирается стратегия с минимальным максимальным сожалением.
    """
    # Минимальные затраты по каждому сценарию (по столбцам)
    min_per_scenario = matrix.min(axis=0, keepdims=True)
    # Матрица сожалений (размерность та же)
    regret_matrix = matrix - min_per_scenario
    # Максимальное сожаление для каждой стратегии
    max_regret = regret_matrix.max(axis=1)
    best_idx = int(np.argmin(max_regret))
    return best_idx, float(max_regret[best_idx])


def full_analysis(
        lambda_scenarios: List[float],
        strategies: List[int],
        mu: float,
        salary_per_op: float,
        penalty_factor: float,
        probabilities: List[float],
        alpha: float = 0.6,
        penalty_lq: float = 0.0,
) -> dict:
    """
    Запуск всех четырёх критериев и возврат результатов.
    """
    matrix = build_cost_matrix(
        lambda_scenarios,
        strategies,
        mu,
        salary_per_op,
        penalty_factor,
        penalty_lq,
    )

    wald_idx, wald_val = apply_wald(matrix)
    hurwicz_idx, hurw_val = apply_hurwicz(matrix, alpha)
    bayes_idx, bayes_val = apply_bayes(matrix, probabilities)
    savage_idx, savage_val = apply_savage(matrix)

    return {
        "matrix": matrix,
        "wald": {"c": strategies[wald_idx], "cost": wald_val},
        "hurwicz": {"c": strategies[hurwicz_idx], "cost": hurw_val, "alpha": alpha},
        "bayes": {
            "c": strategies[bayes_idx],
            "cost": bayes_val,
            "probs": probabilities,
        },
        "savage": {"c": strategies[savage_idx], "cost": savage_val},
    }
