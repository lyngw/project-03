import pandas as pd
import numpy as np


def _extract_amount(df: pd.DataFrame, account_names: list[str], fs_div: str = "CFS") -> float | None:
    """재무제표 DataFrame에서 특정 계정과목의 금액을 추출.

    fs_div: CFS(연결), OFS(개별)
    """
    # 띄어쓰기 제거한 계정과목명으로 매칭
    df = df.copy()
    df["_account_clean"] = df["account_nm"].str.replace(" ", "", regex=False)
    clean_names = [n.replace(" ", "") for n in account_names]

    # 연결 우선, 없으면 개별, fs_div 없는 행도 포함
    for div in [fs_div, "CFS", "OFS", None]:
        if "fs_div" not in df.columns or div is None:
            subset = df
        else:
            subset = df[(df["fs_div"] == div) | (df["fs_div"].isna())]
        for name in clean_names:
            # 1차: 정확히 일치하는 항목
            exact = subset[subset["_account_clean"] == name]
            if not exact.empty:
                for col in ["thstrm_amount", "thstrm_dt"]:
                    if col in exact.columns:
                        val = exact.iloc[0][col]
                        if pd.notna(val):
                            return _parse_number(val)
            # 2차: 부분 매칭 (단, 앞에 다른 수식어가 붙지 않은 것 우선)
            contains = subset[subset["_account_clean"].str.contains(name, na=False, regex=False)]
            if not contains.empty:
                # 계정명이 검색어로 시작하는 것 우선
                starts = contains[contains["_account_clean"].str.startswith(name)]
                pick = starts if not starts.empty else contains
                for col in ["thstrm_amount", "thstrm_dt"]:
                    if col in pick.columns:
                        val = pick.iloc[0][col]
                        if pd.notna(val):
                            return _parse_number(val)
    return None


def _parse_number(val) -> float | None:
    """문자열/숫자를 float로 변환."""
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    if isinstance(val, str):
        val = val.replace(",", "").replace(" ", "").strip()
        if val in ("", "-"):
            return None
        try:
            return float(val)
        except ValueError:
            return None
    return None


def compute_metrics_for_year(year_df: pd.DataFrame) -> dict:
    """한 기업의 1개년 재무제표에서 ROCE와 부채비율 계산.

    ROCE = EBIT / (자기자본 - 순현금)
    순현금 = 현금및현금성자산 - 단기차입금
    부채비율 = 총부채 / 자기자본 × 100
    """
    ebit = _extract_amount(year_df, ["영업이익", "영업손익"])

    # 재무상태표 항목
    total_assets = _extract_amount(year_df, ["자산총계"])
    total_liabilities = _extract_amount(year_df, ["부채총계"])
    total_equity = _extract_amount(year_df, ["자본총계"])

    cash = _extract_amount(year_df, ["현금및현금성자산", "현금및현금등가물"])
    short_term_borrowings = _extract_amount(year_df, ["단기차입금"])

    # 자기자본: 직접 값 또는 계산
    if total_equity is None and total_assets is not None and total_liabilities is not None:
        total_equity = total_assets - total_liabilities

    # 순현금 계산
    net_cash = 0.0
    if cash is not None:
        net_cash = cash
    if short_term_borrowings is not None:
        net_cash -= short_term_borrowings

    # 자본잠식 판정
    capital_impaired = total_equity is not None and total_equity <= 0

    # ROCE 계산
    roce = None
    if not capital_impaired and ebit is not None and total_equity is not None:
        capital_employed = total_equity - net_cash
        if capital_employed != 0:
            roce = ebit / capital_employed * 100  # 퍼센트

    # 부채비율
    debt_ratio = None
    if not capital_impaired and total_liabilities is not None and total_equity is not None and total_equity != 0:
        debt_ratio = total_liabilities / total_equity * 100

    return {
        "ebit": ebit,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "total_equity": total_equity,
        "cash": cash,
        "short_term_borrowings": short_term_borrowings,
        "net_cash": net_cash,
        "roce": roce,
        "debt_ratio": debt_ratio,
        "capital_impaired": capital_impaired,
    }


def compute_all_metrics(financials: pd.DataFrame) -> pd.DataFrame:
    """전체 재무제표 데이터에서 기업별·연도별 지표 계산."""
    if financials.empty:
        return pd.DataFrame()

    results = []
    grouped = financials.groupby(["corp_code", "corp_name", "year"])

    for (corp_code, corp_name, year), group in grouped:
        metrics = compute_metrics_for_year(group)
        metrics["corp_code"] = corp_code
        metrics["corp_name"] = corp_name
        metrics["year"] = year
        if "stock_code" in group.columns:
            metrics["stock_code"] = group["stock_code"].iloc[0]
        results.append(metrics)

    return pd.DataFrame(results)


def pivot_roce(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """ROCE를 기업×연도 피벗 테이블로 변환. 자본잠식은 표시."""
    if metrics_df.empty:
        return pd.DataFrame()
    # 자본잠식 연도는 ROCE를 특수값으로 표시
    df = metrics_df.copy()
    pivot = df.pivot_table(
        index=["corp_code", "corp_name"],
        columns="year",
        values="roce",
    )
    # 자본잠식 피벗
    impaired_pivot = df.pivot_table(
        index=["corp_code", "corp_name"],
        columns="year",
        values="capital_impaired",
        aggfunc="max",
    )
    pivot.columns = [f"ROCE_{int(y)}" for y in pivot.columns]
    impaired_pivot.columns = [f"imp_{int(y)}" for y in impaired_pivot.columns]
    result = pivot.reset_index()
    result = result.join(impaired_pivot.reset_index(drop=True))
    return result


def pivot_debt_ratio(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """부채비율을 기업×연도 피벗 테이블로 변환."""
    if metrics_df.empty:
        return pd.DataFrame()
    df = metrics_df.copy()
    pivot = df.pivot_table(
        index=["corp_code", "corp_name"],
        columns="year",
        values="debt_ratio",
    )
    impaired_pivot = df.pivot_table(
        index=["corp_code", "corp_name"],
        columns="year",
        values="capital_impaired",
        aggfunc="max",
    )
    pivot.columns = [f"부채비율_{int(y)}" for y in pivot.columns]
    impaired_pivot.columns = [f"imp_{int(y)}" for y in impaired_pivot.columns]
    result = pivot.reset_index()
    result = result.join(impaired_pivot.reset_index(drop=True))
    return result
