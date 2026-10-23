from fastapi import APIRouter, HTTPException

from app.api.routes.train import training_busy
from app.core.model_loader import read_json
from app.schemas import PredictRequest
from app.services.forecast_service import forecast_payload
from model.config import normalize_ticker

router = APIRouter()


@router.post("/predict")
def predict(body: PredictRequest) -> dict:
    if training_busy():
        raise HTTPException(status_code=409, detail="Training is still running.")
    return forecast_payload(body.ticker, body.horizons)


@router.get("/interpretability/{ticker}")
def interpretability(ticker: str) -> dict:
    ticker = normalize_ticker(ticker)
    interpretation = read_json(ticker, "interpretability.json")
    try:
        evaluation = read_json(ticker, "evaluation.json")
    except Exception:
        evaluation = None
    try:
        meta = read_json(ticker, "meta.json")
    except Exception:
        meta = {}
    return {
        "ticker": ticker,
        "smoke": bool(meta.get("smoke", False)),
        "costs": meta.get("costs"),
        "split": {
            "train_end_date": meta.get("train_end_date"),
            "validation_start_date": meta.get("validation_start_date"),
            "validation_end_date": meta.get("validation_end_date"),
            "oos_start_date": meta.get("oos_start_date"),
        },
        "encoder_variables": interpretation.get("encoder_variables", []),
        "decoder_variables": interpretation.get("decoder_variables", []),
        "static_variables": interpretation.get("static_variables", []),
        "attention": interpretation.get("attention", []),
        "evaluation": evaluation,
        "error": interpretation.get("error"),
    }
