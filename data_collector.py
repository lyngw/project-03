import os
import time
import pickle
from datetime import datetime

import pandas as pd
import OpenDartReader
import FinanceDataReader as fdr

from config import DART_API_KEY, CACHE_DIR


def get_dart_reader(api_key: str | None = None) -> OpenDartReader:
    key = api_key or DART_API_KEY
    if not key:
        raise ValueError("DART API 키가 설정되지 않았습니다.")
    return OpenDartReader(key)


def get_corp_list(dart: OpenDartReader) -> pd.DataFrame:
    """KRX 실제 거래 중인 상장기업 리스트 반환 (DART corp_code 매핑 포함)."""
    cache_path = os.path.join(CACHE_DIR, "corp_list.pkl")
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if (time.time() - mtime) < 86400:  # 1일 캐시
            return pd.read_pickle(cache_path)

    # KRX 실제 상장 종목 기준
    krx = fdr.StockListing("KRX")
    krx_codes = set(krx["Code"].values)

    # DART corp_codes에서 KRX에 있는 것만 필터
    corp_list = dart.corp_codes
    corp_list["stock_code_clean"] = corp_list["stock_code"].astype(str).str.strip()
    listed = corp_list[corp_list["stock_code_clean"].isin(krx_codes)].copy()
    listed = listed.drop(columns=["stock_code_clean"])

    listed.to_pickle(cache_path)
    return listed


def fetch_financial_statements(
    dart: OpenDartReader,
    corp_code: str,
    corp_name: str,
    years: int = 10,
) -> pd.DataFrame:
    """한 기업의 최근 N년 재무제표(연결) 수집."""
    last_year = datetime.now().year - 1
    all_data = []

    for year in range(last_year - years + 1, last_year + 1):
        try:
            # 연결재무제표 우선, 없으면 개별
            fs = dart.finstate(corp_code, year, reprt_code="11011")  # 사업보고서
            if fs is not None and not fs.empty:
                fs["year"] = year
                fs["corp_code"] = corp_code
                fs["corp_name"] = corp_name
                all_data.append(fs)
            time.sleep(0.15)  # API 호출 제한 관리
        except Exception:
            continue

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def fetch_all_financials(
    dart: OpenDartReader,
    corp_list: pd.DataFrame,
    years: int = 10,
    progress_callback=None,
) -> pd.DataFrame:
    """전체 상장기업 재무제표 수집 (증분 업데이트 지원).

    기존 캐시가 있으면 미수집 기업·연도만 추가 수집한다.
    """
    cache_path = os.path.join(CACHE_DIR, "all_financials.pkl")
    # 사업보고서는 다음해 3월 제출이므로, 현재 연도는 제외
    last_year = datetime.now().year - 1

    # 기존 캐시 로드
    existing = pd.DataFrame()
    if os.path.exists(cache_path):
        existing = pd.read_pickle(cache_path)

    # 이미 수집된 (기업, 연도) 조합
    collected = set()
    if not existing.empty:
        collected = set(zip(existing["corp_code"], existing["year"]))

    all_new = []
    total = len(corp_list)

    for i, (idx, row) in enumerate(corp_list.iterrows()):
        corp_code = row["corp_code"]
        corp_name = row.get("corp_name", row.get("corp_cls", ""))
        stock_code = row.get("stock_code", "")

        if progress_callback:
            progress_callback(i, total, corp_name)

        # 미수집 연도만 파악 (현재 연도 제외)
        need_years = [
            y for y in range(last_year - years + 1, last_year + 1)
            if (corp_code, y) not in collected
        ]

        if not need_years:
            continue

        for year in need_years:
            try:
                fs = dart.finstate(corp_code, year, reprt_code="11011")
                if fs is not None and not fs.empty:
                    fs["year"] = year
                    fs["corp_code"] = corp_code
                    fs["corp_name"] = corp_name
                    fs["stock_code"] = stock_code
                    all_new.append(fs)
                time.sleep(0.15)
            except Exception:
                continue

    # 기존 + 신규 병합
    if all_new:
        new_df = pd.concat(all_new, ignore_index=True)
        result = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    elif not existing.empty:
        result = existing
    else:
        return pd.DataFrame()

    result.to_pickle(cache_path)
    return result


def fetch_all_detail_financials(
    dart: OpenDartReader,
    base_financials: pd.DataFrame,
    years: int = 10,
    progress_callback=None,
) -> pd.DataFrame:
    """기존 수집 성공 기업에 대해 finstate_all(전체 재무제표)을 증분 수집."""
    cache_path = os.path.join(CACHE_DIR, "all_financials_detail.pkl")
    current_year = datetime.now().year

    # 기존 캐시 로드
    existing = pd.DataFrame()
    if os.path.exists(cache_path):
        existing = pd.read_pickle(cache_path)

    collected = set()
    if not existing.empty:
        collected = set(zip(existing["corp_code"], existing["year"]))

    # 기존 요약 재무제표에서 수집 성공한 기업·연도 목록
    targets = base_financials[["corp_code", "corp_name", "stock_code", "year"]].drop_duplicates()
    need = targets[~targets.apply(lambda r: (r["corp_code"], r["year"]) in collected, axis=1)]

    all_new = []
    total = len(need)

    for i, (idx, row) in enumerate(need.iterrows()):
        corp_code = row["corp_code"]
        corp_name = row["corp_name"]
        stock_code = row.get("stock_code", "")
        year = int(row["year"])

        if progress_callback:
            progress_callback(i, total, f"{corp_name} ({year})")

        try:
            fs = dart.finstate_all(corp_code, year, reprt_code="11011")
            if fs is not None and not fs.empty:
                fs["year"] = year
                fs["corp_code"] = corp_code
                fs["corp_name"] = corp_name
                fs["stock_code"] = stock_code
                all_new.append(fs)
            time.sleep(0.15)
        except Exception:
            continue

    if all_new:
        new_df = pd.concat(all_new, ignore_index=True)
        result = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    elif not existing.empty:
        result = existing
    else:
        return pd.DataFrame()

    result.to_pickle(cache_path)
    return result


def fetch_stock_price(stock_code: str, years: int = 10) -> pd.DataFrame:
    """개별 종목의 주가 데이터 조회."""
    start = f"{datetime.now().year - years}-01-01"
    try:
        df = fdr.DataReader(stock_code, start)
        return df
    except Exception:
        return pd.DataFrame()
