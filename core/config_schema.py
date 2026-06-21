"""
Схема валидации конфигурации с использованием Pydantic.
"""
from typing import List
from pydantic import BaseModel, Field, model_validator


class CallCenterConfig(BaseModel):
    # Интенсивности входящего потока (сценарии природы)
    lambda_scenarios: List[float] = Field(..., min_items=1, description="Список интенсивностей lambda >= 0")

    # Производительность одного оператора
    mu: float = Field(..., gt=0, description="Производительность оператора mu должна быть > 0")

    # Стоимость одного оператора
    salary_per_operator: float = Field(..., ge=0, description="Зарплата должна быть >= 0")

    # Штраф за единицу вероятности ожидания
    penalty_factor: float = Field(..., ge=0, description="Штраф за P_wait должен быть >= 0")

    # Стратегии найма (количество операторов)
    strategies: List[int] = Field(..., min_items=1, description="Список стратегий (число операторов)")

    # Критерий по умолчанию
    criterion: str = Field("hurwicz", description="Критерий принятия решений")

    # Коэффициент оптимизма для Гурвица
    alpha: float = Field(0.6, ge=0.0, le=1.0, description="Коэффициент альфа в диапазоне [0, 1]")

    # Вероятности сценариев для Байеса
    probabilities: List[float] = Field(..., min_items=1, description="Вероятности сценариев природы")

    # Порог SLA
    sla_p_wait: float = Field(0.15, ge=0.0, le=1.0, description="Порог SLA P_wait в диапазоне [0, 1]")

    # Опциональный штраф за длину очереди L_q
    penalty_lq: float = Field(0.0, ge=0.0, description="Штраф за единицу длины очереди L_q")

    @model_validator(mode="after")
    def validate_business_logic(self) -> "CallCenterConfig":
        # 1. Проверка элементов lambda
        if any(lam < 0 for lam in self.lambda_scenarios):
            raise ValueError("Все значения lambda_scenarios должны быть неотрицательными.")

        # 2. Проверка элементов strategies
        if any(c <= 0 for c in self.strategies):
            raise ValueError("Количество операторов в стратегиях должно быть строго > 0.")

        # 3. Проверка соответствия размерностей lambda и probabilities
        if len(self.lambda_scenarios) != len(self.probabilities):
            raise ValueError(
                f"Размерность probabilities ({len(self.probabilities)}) "
                f"должна совпадать с lambda_scenarios ({len(self.lambda_scenarios)})."
            )

        # 4. Проверка суммы вероятностей (с допуском на точность float)
        if abs(sum(self.probabilities) - 1.0) > 1e-5:
            raise ValueError(
                f"Сумма вероятностей (probabilities) должна быть равна 1.0. Сейчас: {sum(self.probabilities)}")

        # 5. Валидация названия критерия
        valid_criteria = {"wald", "hurwicz", "bayes", "savage"}
        if self.criterion.lower() not in valid_criteria:
            raise ValueError(f"Неизвестный критерий '{self.criterion}'. Допустимые: {valid_criteria}")

        return self
