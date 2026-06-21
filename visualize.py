"""
Визуализация результатов для CallCenterOpt.

Требует: matplotlib, numpy.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from core.queue_math import calculate_mm_c_metrics


def plot_sla_landscape(
    lambdas: list,
    strategies: list,
    mu: float,
    sla_threshold: float,
    save_path: str = None,
) -> None:
    """
    Построение двух графиков:
      1. P_wait vs c для каждого λ.
      2. Тепловая карта статуса SLA (OK/FAIL) по комбинациям (c, λ).
    """
    # Сортируем для красоты
    lambdas = sorted(lambdas)
    strategies = sorted(strategies)

    # Подготовка данных для графика P_wait
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # График P_wait vs c
    for lam in lambdas:
        p_wait_vals = []
        for c in strategies:
            m = calculate_mm_c_metrics(lam, mu, c)
            p_wait_vals.append(m.p_wait if m.stable else 1.0)
        ax1.plot(strategies, p_wait_vals, marker='o', label=f'λ={lam:.1f}')
    ax1.axhline(y=sla_threshold, color='r', linestyle='--', label=f'SLA = {sla_threshold:.2f}')
    ax1.set_xlabel('Количество операторов c')
    ax1.set_ylabel('Вероятность ожидания P_wait')
    ax1.set_title('Зависимость P_wait от числа операторов')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # Тепловая карта SLA статуса
    # Построим матрицу: строки — стратегии, столбцы — сценарии λ
    sla_matrix = np.zeros((len(strategies), len(lambdas)), dtype=int)
    for i, c in enumerate(strategies):
        for j, lam in enumerate(lambdas):
            m = calculate_mm_c_metrics(lam, mu, c)
            # 1 — OK (P_wait <= threshold и стабильно), 0 — FAIL
            sla_matrix[i, j] = 1 if (m.stable and m.p_wait <= sla_threshold) else 0

    # Цветовая карта: красный (0) и зелёный (1)
    cmap = ListedColormap(['red', 'green'])
    bounds = [-0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax2.imshow(sla_matrix, cmap=cmap, norm=norm, aspect='auto')

    # Настройка осей
    ax2.set_xticks(np.arange(len(lambdas)))
    ax2.set_yticks(np.arange(len(strategies)))
    ax2.set_xticklabels([f'{lam:.0f}' for lam in lambdas])
    ax2.set_yticklabels(strategies)
    ax2.set_xlabel('Сценарий нагрузки λ')
    ax2.set_ylabel('Количество операторов c')
    ax2.set_title('Статус SLA (OK/FAIL)')

    # Добавим рамки вокруг ячеек
    for i in range(len(strategies)):
        for j in range(len(lambdas)):
            text = 'OK' if sla_matrix[i, j] == 1 else 'FAIL'
            ax2.text(j, i, text, ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nГрафик сохранён в {save_path}")
    else:
        plt.show()


def plot_cost_matrix(
    matrix: np.ndarray,
    strategies: list,
    lambdas: list,
    save_path: str = None,
) -> None:
    """
    Тепловая карта матрицы затрат.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')

    # Добавляем значения в ячейки
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Проверяем на inf (нестабильная система)
            val = matrix[i, j]
            if np.isinf(val):
                text = '∞'
            else:
                text = f'{val:.1f}'
            ax.text(j, i, text, ha='center', va='center', color='white' if val < matrix.max()/2 else 'black')

    ax.set_xticks(np.arange(len(lambdas)))
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_xticklabels([f'λ={lam:.0f}' for lam in lambdas])
    ax.set_yticklabels(strategies)
    ax.set_xlabel('Сценарий нагрузки')
    ax.set_ylabel('Количество операторов c')
    ax.set_title('Матрица затрат (тыс. руб./месяц)')

    plt.colorbar(im, ax=ax, label='Затраты')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nГрафик сохранён в {save_path}")
    else:
        plt.show()
