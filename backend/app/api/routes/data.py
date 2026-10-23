from fastapi import APIRouter, Query

from app.services.data_service import history_payload

router = APIRouter()


@router.get("/data/{ticker}")
def get_data(
    ticker: str,
    n: int = Query(default=756, ge=30, le=5000),
    refresh: bool = False,
) -> dict:
    return history_payload(ticker, n, refresh=refresh)
