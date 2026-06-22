"""
Тесты для модуля game_solver.
"""

import numpy as np
import pytest
from core.game_solver import (
    build_cost_matrix,
    apply_wald,
    apply_hurwicz,
    apply_bayes,
    apply_savage,
    full_analysis,
)

LAMBDAS = [10, 30, 60]
STRATEGIES = [2, 4, 6, 8, 10]
MU = 15.0
SALARY = 40.0
PENALTY = 200.0
PENALTY_LQ = 0.0
PROBS = [0.2, 0.5, 0.3]


@pytest.fixture
def cost_matrix():
    """Базовая матрица затрат (без штрафа за L_q)."""
    return build_cost_matrix(LAMBDAS, STRATEGIES, MU, SALARY, PENALTY, PENALTY_LQ)


@pytest.fixture
def cost_matrix_with_lq():
    """Матрица затрат со штрафом за L_q."""
    return build_cost_matrix(LAMBDAS, STRATEGIES, MU, SALARY, PENALTY, penalty_lq=10.0)


class TestBuildCostMatrix:

    def test_shape(self, cost_matrix):
        assert cost_matrix.shape == (len(STRATEGIES), len(LAMBDAS))

    def test_all_positive(self, cost_matrix):
        assert np.all(cost_matrix >= 0)

    def test_more_operators_increase_salary_cost(self):
        mat = build_cost_matrix([0.0], STRATEGIES, MU, SALARY, penalty_factor=0.0, penalty_lq=0.0)
        for i in range(len(STRATEGIES) - 1):
            assert mat[i + 1, 0] > mat[i, 0]

    def test_higher_lambda_increases_cost(self, cost_matrix):
        idx_c5 = STRATEGIES.index(4)
        assert cost_matrix[idx_c5, 2] >= cost_matrix[idx_c5, 0]

    def test_enough_operators_cap_penalty(self):
        mat = build_cost_matrix([20.0], [20], MU, SALARY, PENALTY, 0.0)
        m = mat[0, 0]
        assert abs(m - 20 * SALARY) < 1.0

    def test_penalty_lq_increases_cost(self):
        """Проверка, что ненулевой penalty_lq увеличивает затраты (для стабильных систем)."""
        mat0 = build_cost_matrix([20.0], [5], MU, SALARY, PENALTY, penalty_lq=0.0)
        mat1 = build_cost_matrix([20.0], [5], MU, SALARY, PENALTY, penalty_lq=10.0)
        assert mat1[0, 0] > mat0[0, 0]

    def test_penalty_lq_inf_for_unstable(self):
        """При нестабильной системе L_q = inf, затраты должны стать inf."""
        mat = build_cost_matrix([100.0], [3], MU, SALARY, PENALTY, penalty_lq=10.0)
        assert np.isinf(mat[0, 0])


class TestWald:

    def test_returns_tuple(self, cost_matrix):
        result = apply_wald(cost_matrix)
        assert isinstance(result, tuple) and len(result) == 2

    def test_index_in_range(self, cost_matrix):
        idx, _ = apply_wald(cost_matrix)
        assert 0 <= idx < len(STRATEGIES)

    def test_pessimistic_picks_minimax(self, cost_matrix):
        idx, val = apply_wald(cost_matrix)
        worst_cases = cost_matrix.max(axis=1)
        assert val == pytest.approx(worst_cases[idx], abs=1e-6)
        assert val == pytest.approx(worst_cases.min(), abs=1e-6)

    def test_consistent_with_manual(self, cost_matrix):
        worst = cost_matrix.max(axis=1)
        expected_idx = int(np.argmin(worst))
        idx, val = apply_wald(cost_matrix)
        assert idx == expected_idx
        assert abs(val - worst[expected_idx]) < 1e-9


class TestHurwicz:

    def test_alpha_zero_equals_wald(self, cost_matrix):
        wald_idx, _ = apply_wald(cost_matrix)
        hurw_idx, _ = apply_hurwicz(cost_matrix, alpha=0.0)
        assert wald_idx == hurw_idx

    def test_alpha_one_optimistic(self, cost_matrix):
        idx, val = apply_hurwicz(cost_matrix, alpha=1.0)
        best = cost_matrix.min(axis=1)
        assert idx == int(np.argmin(best))

    def test_alpha_midpoint(self, cost_matrix):
        idx, val = apply_hurwicz(cost_matrix, alpha=0.6)
        assert 0 <= idx < len(STRATEGIES)
        assert val > 0

    def test_invalid_alpha_raises(self, cost_matrix):
        with pytest.raises(ValueError):
            apply_hurwicz(cost_matrix, alpha=1.5)
        with pytest.raises(ValueError):
            apply_hurwicz(cost_matrix, alpha=-0.1)


class TestBayes:

    def test_uniform_probs(self, cost_matrix):
        uniform = [1.0 / len(LAMBDAS)] * len(LAMBDAS)
        idx, val = apply_bayes(cost_matrix, uniform)
        expected_costs = cost_matrix @ np.array(uniform)
        assert idx == int(np.argmin(expected_costs))
        assert abs(val - expected_costs[idx]) < 1e-9

    def test_custom_probs(self, cost_matrix):
        idx, val = apply_bayes(cost_matrix, PROBS)
        assert 0 <= idx < len(STRATEGIES)

    def test_probs_not_summing_to_one_raises(self, cost_matrix):
        with pytest.raises(ValueError):
            apply_bayes(cost_matrix, [0.3, 0.3, 0.3])

    def test_negative_prob_raises(self, cost_matrix):
        with pytest.raises(ValueError):
            apply_bayes(cost_matrix, [-0.1, 0.6, 0.5])


class TestSavage:
    """Тесты для критерия Сэвиджа (минимакс сожаления)."""

    def test_returns_tuple(self, cost_matrix):
        result = apply_savage(cost_matrix)
        assert isinstance(result, tuple) and len(result) == 2

    def test_index_in_range(self, cost_matrix):
        idx, _ = apply_savage(cost_matrix)
        assert 0 <= idx < len(STRATEGIES)

    def test_regret_matrix_calculation(self, cost_matrix):
        """Проверяем, что сожаление равно cost - min_по_столбцу."""
        # Вычислим вручную для первой строки
        min_per_col = cost_matrix.min(axis=0)
        # Применим apply_savage, внутри она строит regret_matrix,
        # но мы не можем её достать. Проверим только логику через вызов.
        idx, val = apply_savage(cost_matrix)
        # Для найденной стратегии val должно быть равно max regret по строке
        max_regret = (cost_matrix - min_per_col).max(axis=1)
        assert val == pytest.approx(max_regret[idx], abs=1e-6)

    def test_savage_on_known_matrix(self):
        """
        Тест на известной матрице:
        Стратегии A и B, сценарии 1 и 2.
        Матрица: [[10, 20], [15, 15]]
        Минимальные по столбцам: [10, 15]
        Сожаление: [[0, 5], [5, 0]]
        Максимальные сожаления: [5, 5] → обе стратегии дают 5 → выбираем первую (индекс 0).
        """
        known_matrix = np.array([[10.0, 20.0], [15.0, 15.0]])
        idx, val = apply_savage(known_matrix)
        assert idx == 0
        assert val == 5.0

    def test_savage_with_penalty_lq(self, cost_matrix_with_lq):
        """Проверяем, что Сэвидж работает и с матрицей, содержащей inf (нестабильные системы)."""
        idx, val = apply_savage(cost_matrix_with_lq)
        assert 0 <= idx < len(STRATEGIES)
        assert not np.isnan(val)
        # В этой матрице могут быть inf, но apply_savage должна корректно работать
        # (inf в сожалении не должны влиять, если только все стратегии не дают inf)


class TestFullAnalysis:

    def test_returns_all_keys(self):
        result = full_analysis(LAMBDAS, STRATEGIES, MU, SALARY, PENALTY, PROBS)
        assert "matrix" in result
        assert "wald" in result
        assert "hurwicz" in result
        assert "bayes" in result
        assert "savage" in result

    def test_all_recommended_c_in_strategies(self):
        result = full_analysis(LAMBDAS, STRATEGIES, MU, SALARY, PENALTY, PROBS)
        for key in ("wald", "hurwicz", "bayes", "savage"):
            assert result[key]["c"] in STRATEGIES

    def test_hurwicz_alpha_stored(self):
        result = full_analysis(
            LAMBDAS, STRATEGIES, MU, SALARY, PENALTY, PROBS, alpha=0.7
        )
        assert result["hurwicz"]["alpha"] == 0.7

    def test_savage_in_full_analysis(self, cost_matrix):
        """Проверяем, что savage в full_analysis возвращает тот же результат, что и apply_savage."""
        result = full_analysis(LAMBDAS, STRATEGIES, MU, SALARY, PENALTY, PROBS)
        savage_idx, savage_val = apply_savage(result["matrix"])
        assert result["savage"]["c"] == STRATEGIES[savage_idx]
        assert abs(result["savage"]["cost"] - savage_val) < 1e-6
