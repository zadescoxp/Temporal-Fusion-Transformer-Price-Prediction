import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import data, predict, train
from model.errors import DataError, ModelNotFoundError

app = FastAPI(title="TFT Finance", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(data.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(train.router, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(ModelNotFoundError)
async def missing_model(_request, exc: ModelNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(DataError)
async def bad_data(_request, exc: DataError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def bad_value(_request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
