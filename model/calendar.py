"""NYSE regular holidays and the next trading session.

Holiday rules are known before the session, so flags derived from them are
valid known-future covariates. One-off closures (weather, funerals) are not
in this calendar. Juneteenth is included from 2022, when it became an NYSE
holiday.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 15)


def easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    shift = (weekday - first.weekday()) % 7
    return first + timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    return cursor - timedelta(days=(cursor.weekday() - weekday) % 7)


def _observe(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def nyse_holidays_for_year(year: int) -> list[date]:
    days = [
        _observe(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observe(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observe(date(year, 12, 25)),
    ]
    if year >= 2022:
        days.append(_observe(date(year, 6, 19)))
    return days


@lru_cache(maxsize=8)
def holiday_set(start_year: int, end_year: int) -> frozenset[date]:
    found: set[date] = set()
    for year in range(start_year, end_year + 1):
        found.update(nyse_holidays_for_year(year))
    return frozenset(found)


def _holidays_around(day: date) -> frozenset[date]:
    return holiday_set(day.year - 1, day.year + 1)


def _is_trading_day(day: date, holidays: frozenset[date]) -> bool:
    return day.weekday() < 5 and day not in holidays


def _neighbor(day: date, holidays: frozenset[date], step: int) -> date:
    cursor = day + timedelta(days=step)
    for _ in range(12):
        if _is_trading_day(cursor, holidays):
            return cursor
        cursor += timedelta(days=step)
    return cursor


def _span_has_holiday(start: date, end: date, holidays: frozenset[date]) -> bool:
    cursor = start + timedelta(days=1)
    while cursor < end:
        if cursor in holidays:
            return True
        cursor += timedelta(days=1)
    return False


def is_holiday_adjacent(day: date) -> bool:
    """True when a full NYSE closure sits against this trading session.

    The holiday itself is not a row in the price series, so the covariate
    marks the sessions on either side of the closure. Weekends alone do not
    count. The exchange calendar is public before the session.
    """
    holidays = _holidays_around(day)
    if not _is_trading_day(day, holidays):
        return False
    previous_session = _neighbor(day, holidays, -1)
    next_session = _neighbor(day, holidays, 1)
    return _span_has_holiday(previous_session, day, holidays) or _span_has_holiday(
        day, next_session, holidays
    )


def next_trading_days(after: date, count: int) -> list[date]:
    if count < 1:
        return []
    holidays = holiday_set(after.year - 1, after.year + 2)
    days: list[date] = []
    cursor = after + timedelta(days=1)
    while len(days) < count:
        if _is_trading_day(cursor, holidays):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def last_complete_session(now: datetime | None = None) -> date:
    """Most recent regular session whose cash close has already printed."""
    moment = now or datetime.now(ET)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=ET)
    else:
        moment = moment.astimezone(ET)
    day = moment.date()
    holidays = _holidays_around(day)
    trading_today = _is_trading_day(day, holidays)
    if trading_today and moment.time() >= MARKET_CLOSE:
        return day
    cursor = day - timedelta(days=1) if trading_today else day
    while not _is_trading_day(cursor, holidays):
        cursor -= timedelta(days=1)
    return cursor
