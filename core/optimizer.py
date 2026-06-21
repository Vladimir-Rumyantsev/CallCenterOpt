"""
Модуль ЛП-оптимизации штата с использованием библиотеки PuLP.
Используется в качестве математического эталона для верификации игровых критериев.
"""

import pulp
from typing import List, Dict, Any
from core.queue_math import calculate_mm_c_metrics


def solve_lp_baseline(
        lambda_scenarios: List[float],
        mu: float,
        salary_per_op: float,
        penalty_factor: float,
        sla_threshold: float,
        probabilities: List[float],
) -> Dict[str, Any]:
    """
    Находит оптимальное количество операторов методом целочисленного линейного
    программирования (ИП) для детерминированного или усредненного сценария,
    минимизируя затраты на содержание штата при жестком ограничении на SLA.
    """

    # 1. Расчет средневзвешенной интенсивности потока (математическое ожидание нагрузки)
    avg_lambda = sum(lam * p for lam, p in zip(lambda_scenarios, probabilities))

    # Инициализируем задачу минимизации
    prob = pulp.LpProblem("CallCenter_Staff_Optimization", pulp.LpMinimize)

    # Определяем переменную: c (количество операторов) — строго целое число
    # Нижняя граница: минимум 1 оператор, верхняя — разумный максимум
    max_possible_c = max(int(max(lambda_scenarios) / mu) + 5, 20)
    c_var = pulp.LpVariable("Operators_Count", lowBound=1, upBound=max_possible_c, cat=pulp.LpInteger)

    # Целевая функция: минимизируем расходы на зарплату (для базового штата)
    prob += c_var * salary_per_op, "Total_Salary_Cost"

    # Ограничение 1: Система должна быть стабильной хотя бы для средней нагрузки (rho < 1)
    # avg_lambda / (c * mu) < 1  =>  c * mu > avg_lambda
    prob += c_var * mu >= avg_lambda + 0.001, "System_Stability_Constraint"

    # Решаем задачу (подавляем вывод логов солвера)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Получаем базовое значение c из линейной модели
    lp_c = int(pulp.value(c_var))

    # 2. Корректировка с учетом нелинейного SLA (формулы Эрланга C)
    # Поскольку PuLP работает только с линейными функциями, мы используем ЛП для нахождения
    # опорной точки, а затем докручиваем решение, проверяя реальный P_wait по формулам СМО.
    final_c = lp_c
    while final_c <= max_possible_c:
        metrics = calculate_mm_c_metrics(avg_lambda, mu, final_c)
        if metrics.stable and metrics.p_wait <= sla_threshold:
            break
        final_c += 1

    # Рассчитываем итоговые метрики для оптимального ЛП-решения
    final_metrics = calculate_mm_c_metrics(avg_lambda, mu, final_c)
    base_cost = final_c * salary_per_op + (
        penalty_factor * final_metrics.p_wait if final_metrics.stable else float('inf'))

    return {
        "c": final_c,
        "p_wait": final_metrics.p_wait,
        "cost": base_cost,
        "status": pulp.LpStatus[prob.status]
    }
