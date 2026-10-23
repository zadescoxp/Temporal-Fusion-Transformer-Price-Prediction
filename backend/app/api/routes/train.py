import threading

from fastapi import APIRouter, HTTPException

from app.core.model_loader import invalidate
from app.schemas import TrainRequest
from model.train import train_ticker

router = APIRouter()
_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_STATE: dict = {
    "running": False,
    "ticker": None,
    "phase": "idle",
    "message": "",
    "epoch": None,
    "max_epochs": None,
    "batch": None,
    "batches": None,
    "train_loss": None,
    "val_loss": None,
    "progress": 0.0,
    "error": None,
}


def _write(payload: dict | None = None, **patch) -> None:
    with _STATE_LOCK:
        if payload:
            _STATE.update(payload)
        if patch:
            _STATE.update(patch)


def _read() -> dict:
    with _STATE_LOCK:
        return dict(_STATE)


def training_busy() -> bool:
    return _LOCK.locked()


@router.get("/train/status")
def train_status() -> dict:
    return _read()


@router.post("/train")
def train(body: TrainRequest) -> dict:
    if not _LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Training is already running.")
    _write(
        running=True,
        ticker=body.ticker,
        phase="preparing",
        message="Preparing price history",
        epoch=0,
        max_epochs=1 if body.smoke else body.max_epochs,
        batch=None,
        batches=None,
        train_loss=None,
        val_loss=None,
        progress=0.02,
        error=None,
    )
    try:
        result = train_ticker(
            ticker=body.ticker,
            max_epochs=body.max_epochs,
            batch_size=body.batch_size,
            accelerator=body.accelerator,
            smoke=body.smoke,
            on_progress=_write,
        )
        invalidate(body.ticker)
        _write(running=False, phase="done", message="Checkpoint ready", progress=1.0, error=None)
        return result
    except Exception as exc:
        _write(running=False, phase="failed", message="Training failed", error=str(exc))
        raise
    finally:
        _LOCK.release()
