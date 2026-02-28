"""
경기선행지수 데이터 수집 모듈
- OECD 경기선행지수: 주가지수, 단기금리, 장단기 금리차
- 컨퍼런스보드 경기선행지수: S&P500, 실질 M2, 장단기 금리차
- 한국 경기선행지수: 경제심리지수, 국제원자재가격, KOSPI, 장단기 금리차
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os

# FRED API 설정
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# 한국은행 ECOS API 설정
BOK_API_KEY = os.getenv("BOK_API_KEY", "")


def get_fred_series(series_id: str, start_date: str = None) -> pd.DataFrame:
    """FRED에서 시계열 데이터 조회."""
    if not FRED_API_KEY:
        return pd.DataFrame()

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "observations" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["observations"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "value"]].dropna()
        df = df.set_index("date")
        return df
    except Exception as e:
        print(f"FRED API 오류 ({series_id}): {e}")
        return pd.DataFrame()


def get_bok_series(stat_code: str, item_code: str, start_date: str = None) -> pd.DataFrame:
    """한국은행 ECOS API에서 시계열 데이터 조회."""
    if not BOK_API_KEY:
        return pd.DataFrame()

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m")
    else:
        start_date = start_date.replace("-", "")[:6]

    end_date = datetime.now().strftime("%Y%m")

    # ECOS API URL
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{BOK_API_KEY}/json/kr/1/1000/{stat_code}/M/{start_date}/{end_date}/{item_code}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "StatisticSearch" not in data:
            return pd.DataFrame()

        rows = data["StatisticSearch"].get("row", [])
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["TIME"] + "01", format="%Y%m%d")
        df["value"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
        df = df[["date", "value"]].dropna()
        df = df.set_index("date")
        return df
    except Exception as e:
        print(f"BOK API 오류 ({stat_code}/{item_code}): {e}")
        return pd.DataFrame()


def fetch_oecd_indicators() -> dict:
    """
    OECD 경기선행지수 구성요소 조회
    - 주가지수: S&P 500 (FRED)
    - 단기금리: 3개월 국채 금리 (FRED)
    - 장단기 금리차: 10년-3개월 (FRED)
    """
    result = {}

    # S&P 500 (주가지수 대리변수)
    sp500 = get_fred_series("SP500")
    if not sp500.empty:
        result["주가지수 (S&P 500)"] = sp500

    # 단기금리: 3개월 국채
    tb3ms = get_fred_series("TB3MS")
    if not tb3ms.empty:
        result["단기금리 (3개월 국채)"] = tb3ms

    # 장단기 금리차: 10년 - 3개월
    t10y3m = get_fred_series("T10Y3M")
    if not t10y3m.empty:
        result["장단기 금리차 (10Y-3M)"] = t10y3m

    return result


def fetch_conference_board_indicators() -> dict:
    """
    컨퍼런스보드 경기선행지수 구성요소 조회
    - S&P 500
    - 실질 M2 통화공급
    - 장단기 금리차
    """
    result = {}

    # S&P 500
    sp500 = get_fred_series("SP500")
    if not sp500.empty:
        result["S&P 500"] = sp500

    # 실질 M2 (M2SL / CPIAUCSL * 100)
    m2 = get_fred_series("M2SL")
    cpi = get_fred_series("CPIAUCSL")
    if not m2.empty and not cpi.empty:
        # 월말 기준으로 정렬
        m2_monthly = m2.resample("ME").last()
        cpi_monthly = cpi.resample("ME").last()
        merged = m2_monthly.join(cpi_monthly, lsuffix="_m2", rsuffix="_cpi", how="inner")
        if not merged.empty:
            merged["value"] = merged["value_m2"] / merged["value_cpi"] * 100
            result["실질 M2 통화공급"] = merged[["value"]]

    # 장단기 금리차
    t10y3m = get_fred_series("T10Y3M")
    if not t10y3m.empty:
        result["장단기 금리차 (10Y-3M)"] = t10y3m

    return result


def fetch_korea_indicators() -> dict:
    """
    한국 경기선행지수 구성요소 조회
    - 경제심리지수 (BOK)
    - 국제원자재가격지수 (BOK)
    - KOSPI (FinanceDataReader)
    - 장단기 금리차 (BOK)
    """
    import FinanceDataReader as fdr

    result = {}
    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")

    # KOSPI
    try:
        kospi = fdr.DataReader("KS11", start_date)
        if not kospi.empty and "Close" in kospi.columns:
            kospi_df = kospi[["Close"]].copy()
            kospi_df.columns = ["value"]
            result["KOSPI"] = kospi_df
    except Exception as e:
        print(f"KOSPI 조회 오류: {e}")

    # 경제심리지수 (BOK: 513Y001 / X)
    esi = get_bok_series("513Y001", "X")
    if not esi.empty:
        result["경제심리지수"] = esi

    # 국제원자재가격지수 (BOK: 901Y062 / P)
    commodity = get_bok_series("901Y062", "P")
    if not commodity.empty:
        result["국제원자재가격지수"] = commodity

    # 장단기 금리차: 국고채 10년 - 통안채 1년 (또는 국고채 3년)
    # 국고채 10년: 817Y002 / 010500000
    # 국고채 3년: 817Y002 / 010200000
    kr_10y = get_bok_series("817Y002", "010500000")
    kr_3y = get_bok_series("817Y002", "010200000")
    if not kr_10y.empty and not kr_3y.empty:
        merged = kr_10y.join(kr_3y, lsuffix="_10y", rsuffix="_3y", how="inner")
        if not merged.empty:
            merged["value"] = merged["value_10y"] - merged["value_3y"]
            result["장단기 금리차 (10Y-3Y)"] = merged[["value"]]

    return result


def get_latest_values(indicators: dict) -> pd.DataFrame:
    """각 지표의 최신값과 전월 대비 변화 계산."""
    rows = []
    for name, df in indicators.items():
        if df.empty:
            continue
        series = df.copy()
        if isinstance(series.index, pd.DatetimeIndex):
            series = series.sort_index()
            # 일/주 단위라도 월말 기준으로 비교하도록 월말 리샘플링
            series = series.resample("ME").last().dropna()

        if series.empty:
            continue

        latest = series.iloc[-1]["value"]
        latest_date = series.index[-1]

        # 전월 대비 변화
        prev_value = None
        change = None
        change_pct = None

        if len(series) >= 2:
            prev_value = series.iloc[-2]["value"]
            change = latest - prev_value
            if prev_value != 0:
                change_pct = (change / abs(prev_value)) * 100

        rows.append({
            "지표": name,
            "최신값": latest,
            "기준일": latest_date.strftime("%Y-%m-%d"),
            "전월값": prev_value,
            "변화": change,
            "변화율(%)": change_pct,
        })

    return pd.DataFrame(rows)


def get_indicator_history(indicators: dict, months: int = 36) -> pd.DataFrame:
    """지표별 시계열 데이터를 하나의 DataFrame으로 병합."""
    if not indicators:
        return pd.DataFrame()

    dfs = []
    for name, df in indicators.items():
        if df.empty:
            continue
        temp = df.copy()
        temp.columns = [name]
        dfs.append(temp)

    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="outer")

    # 최근 N개월만
    cutoff = datetime.now() - timedelta(days=30 * months)
    merged = merged[merged.index >= cutoff]

    return merged.sort_index()
