"""
Тесты для модуля queue_math (формулы M/M/c).
"""

import pytest
from core.queue_math import calculate_mm_c_metrics, erlang_c, QueueMetrics


class TestErlangC:

    def test_single_server_light_load(self):
        p = erlang_c(5.0, 10.0, 1)
        assert abs(p - 0.5) < 1e-6

    def test_single_server_heavy_load(self):
        p = erlang_c(9.0, 10.0, 1)
        assert abs(p - 0.9) < 1e-4

    def test_unstable_system_returns_one(self):
        p = erlang_c(100.0, 10.0, 1)
        assert p == 1.0

    def test_multiserver_reduces_wait(self):
        p1 = erlang_c(8.0, 10.0, 1)
        p2 = erlang_c(8.0, 10.0, 2)
        assert p2 < p1

    def test_zero_load(self):
        p = erlang_c(0.0, 10.0, 3)
        assert p == pytest.approx(0.0, abs=1e-9)

    def test_invalid_c_raises(self):
        with pytest.raises(ValueError):
            erlang_c(5.0, 10.0, 0)

    def test_invalid_mu_raises(self):
        with pytest.raises(ValueError):
            erlang_c(5.0, 0.0, 1)


class TestCalculateMMC:

    def test_basic_mm1(self):
        m = calculate_mm_c_metrics(10.0, 20.0, 1)
        assert m.stable is True
        assert abs(m.rho - 0.5) < 1e-6
        assert abs(m.p_wait - 0.5) < 1e-6
        assert abs(m.l_q - 0.5) < 1e-4

    def test_mm1_waiting_time(self):
        m = calculate_mm_c_metrics(10.0, 20.0, 1)
        assert abs(m.w - 0.05) < 1e-6

    def test_unstable_system(self):
        m = calculate_mm_c_metrics(50.0, 10.0, 3)
        assert m.stable is False
        assert m.p_wait == 1.0
        assert m.l_q == float("inf")

    def test_three_servers(self):
        m = calculate_mm_c_metrics(20.0, 12.0, 3)
        assert m.stable is True
        assert abs(m.rho - 20.0 / 36.0) < 1e-6
        assert 0.0 < m.p_wait < 1.0

    def test_high_capacity_near_zero_wait(self):
        m = calculate_mm_c_metrics(20.0, 12.0, 10)
        assert m.stable is True
        assert m.p_wait < 0.01

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            calculate_mm_c_metrics(-1.0, 10.0, 2)

    def test_returns_dataclass(self):
        m = calculate_mm_c_metrics(5.0, 10.0, 1)
        assert isinstance(m, QueueMetrics)
        assert hasattr(m, "rho")
        assert hasattr(m, "p_wait")
        assert hasattr(m, "l_q")
        assert hasattr(m, "w")

    def test_p0_plus_pwait_le_one(self):
        m = calculate_mm_c_metrics(15.0, 10.0, 3)
        assert 0.0 <= m.p0 <= 1.0
        assert 0.0 <= m.p_wait <= 1.0

    def test_sla_reference_case_c9(self):
        m = calculate_mm_c_metrics(50.0, 12.0, 9)
        assert m.stable is True
        assert m.p_wait < 0.05

    def test_borderline_case_c7(self):
        m = calculate_mm_c_metrics(50.0, 12.0, 7)
        assert m.stable is True
        assert 0.10 < m.p_wait < 0.25
