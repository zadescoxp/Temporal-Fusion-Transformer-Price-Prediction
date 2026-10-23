from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from model.config import HORIZONS, normalize_ticker


class PredictRequest(BaseModel):
    ticker: str = "AAPL"
    horizons: list[int] = Field(default_factory=lambda: list(HORIZONS))

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, value: str) -> str:
        return normalize_ticker(value)

    @field_validator("horizons")
    @classmethod
    def _horizons(cls, value: list[int]) -> list[int]:
        unknown = [item for item in value if item not in HORIZONS]
        if unknown:
            raise ValueError(f"Horizons must be chosen from {list(HORIZONS)}.")
        if not value:
            raise ValueError("Request at least one horizon.")
        return value


class TrainRequest(BaseModel):
    ticker: str = "AAPL"
    max_epochs: int = Field(default=30, ge=1, le=100)
    batch_size: int = Field(default=32, ge=4, le=128)
    accelerator: Literal["auto", "mps", "cpu"] = "auto"
    smoke: bool = False

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, value: str) -> str:
        return normalize_ticker(value)
