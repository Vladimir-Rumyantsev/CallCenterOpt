"""
Математическое ядро: метрики очереди M/M/c по формуле Эрланга C.
"""

import math
from dataclasses import dataclass


@dataclass
class QueueMetrics:
    rho: float
    p0: float
    p_wait: float
    l_q: float
    w: float
    stable: bool


def erlang_c(lambda_rate: float, mu_rate: float, c: int) -> float:
    """Вычисление вероятности ожидания для очереди M/M/c."""
    if c <= 0 or mu_rate <= 0:
        raise ValueError("c и mu_rate должны быть положительными")
    rho = lambda_rate / (c * mu_rate)
    if rho >= 1.0:
        return 1.0

    a = lambda_rate / mu_rate

    sum_terms = sum(a**n / math.factorial(n) for n in range(c))
    last_term = a**c / (math.factorial(c) * (1.0 - rho))
    p0 = 1.0 / (sum_terms + last_term)
    return last_term * p0


def calculate_mm_c_metrics(lambda_rate: float, mu_rate: float, c: int) -> QueueMetrics:
    """Расчет характеристик очереди M/M/c."""
    if c <= 0 or mu_rate <= 0:
        raise ValueError("c и mu_rate должны быть положительными")
    if lambda_rate < 0:
        raise ValueError("lambda_rate должно быть неотрицательным")

    rho = lambda_rate / (c * mu_rate)
    stable = rho < 1.0

    if not stable:
        return QueueMetrics(
            rho=rho, p0=0.0, p_wait=1.0, l_q=float("inf"), w=float("inf"), stable=False
        )

    a = lambda_rate / mu_rate

    sum_terms = sum(a**n / math.factorial(n) for n in range(c))
    last_term = a**c / (math.factorial(c) * (1.0 - rho))
    p0 = 1.0 / (sum_terms + last_term)

    p_wait = last_term * p0
    l_q = (rho * p_wait) / (1.0 - rho)
    w = l_q / lambda_rate if lambda_rate > 0 else 0.0

    return QueueMetrics(rho=rho, p0=p0, p_wait=p_wait, l_q=l_q, w=w, stable=True)
