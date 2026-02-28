import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
import os
import pickle

from config import DART_API_KEY, CACHE_DIR


class _NumpyCompatUnpickler(pickle.Unpickler):
    """numpy 버전 호환 unpickler (numpy._core → numpy.core 매핑)"""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _read_pickle_compat(path):
    """numpy 호환 pickle 로드 (pd.read_pickle 대체)"""
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()
from data_collector import get_dart_reader, get_corp_list, fetch_all_financials, fetch_all_detail_financials, fetch_quarterly_net_cash, fetch_stock_price
from analyzer import compute_all_metrics, pivot_roce, pivot_debt_ratio, pivot_net_cash, compute_quarterly_net_cash
from leading_indicators import (
    fetch_oecd_indicators, fetch_conference_board_indicators, fetch_korea_indicators,
    get_latest_values, get_indicator_history
)
from ai_regime import (
    KOSPI200_SECTOR_MAP, build_sector_dataframe,
    fetch_sector_prices, compute_regime_momentum,
    plot_sensitivity_bar, plot_position_treemap, plot_regime_scatter,
    plot_displacement_spiral, plot_scenario_timeline,
)

st.set_page_config(page_title="퀀트투자 분석 툴", layout="wide")

# --- 영문 약어 → 한글 변환 (검색용) ---
ABBREV_MAP = {
    "SK": "에스케이", "LG": "엘지", "KT": "케이티", "CJ": "씨제이",
    "GS": "지에스", "LS": "엘에스", "HD": "에이치디", "DB": "디비",
    "KB": "케이비", "NH": "엔에이치", "DL": "디엘", "HL": "에이치엘",
    "OCI": "오씨아이", "BGF": "비지에프", "KCC": "케이씨씨",
    "SDN": "에스디엔", "KPX": "케이피엑스", "HLB": "에이치엘비",
    "JW": "제이더블유", "AK": "에이케이", "DI": "디아이",
}

def expand_search(query: str) -> list[str]:
    """검색어를 확장: 영문 약어 → 한글 변환 포함."""
    if not query:
        return []
    queries = [query]
    upper = query.upper()
    # 영문 약어가 포함되어 있으면 한글 버전도 추가
    for eng, kor in ABBREV_MAP.items():
        if eng in upper:
            converted = query.replace(eng, kor).replace(eng.lower(), kor)
            if converted not in queries:
                queries.append(converted)
    # 한글이 포함되어 있으면 영문 버전도 추가 (역방향)
    for eng, kor in ABBREV_MAP.items():
        if kor in query:
            converted = query.replace(kor, eng)
            if converted not in queries:
                queries.append(converted)
    return queries

def search_filter(df: pd.DataFrame, col: str, query: str) -> pd.DataFrame:
    """확장된 검색어로 DataFrame 필터링."""
    if not query:
        return df
    queries = expand_search(query)
    mask = pd.Series(False, index=df.index)
    for q in queries:
        mask |= df[col].str.contains(q, na=False, case=False)
    return df[mask]

# --- 모바일 최적화 CSS ---
st.markdown("""
<style>
/* 모바일: 넓은 테이블 가로 스크롤 */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        width: 100% !important;
        flex: 1 1 100% !important;
    }
    .stDataFrame > div {
        overflow-x: auto !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("한국 증시 퀀트투자 분석 툴")
st.caption("ROCE · 부채비율 10년 추이 분석")

# --- 사이드바 ---
api_key = DART_API_KEY
collect_btn = False
collect_detail_btn = False
collect_qcash_btn = False

with st.sidebar:
    st.header("설정")
    years = st.slider("분석 기간 (년)", 3, 10, 10)

    st.divider()
    with st.expander("산출 공식 안내"):
        st.markdown("""
**ROCE (Return on Capital Employed)**
$$
ROCE = \\frac{EBIT}{자기자본 - 순현금} \\times 100
$$
- EBIT = 영업이익
- 자기자본 = 자본총계
- 순현금 = 현금및현금성자산 - 단기차입금
- 분모 = 영업에 실제 투입된 자본

**부채비율**
$$
부채비율 = \\frac{총부채}{자기자본} \\times 100
$$

**후행 PER (TTM)**
$$
PER = \\frac{현재 주가}{주당순이익(EPS)}
$$
- EPS = TTM 당기순이익 / 발행주식수
- TTM 당기순이익 = 최근 4개 분기 당기순이익 합산
- 지배기업 귀속 당기순이익 기준
""")

    if os.path.exists(os.path.join(CACHE_DIR, "all_financials.pkl")):
        st.success("캐시된 데이터가 있습니다")

    # --- 관리자 모드 ---
    st.divider()
    with st.expander("관리자 모드"):
        admin_pw = st.text_input("관리자 비밀번호", type="password", key="admin_pw")
        _correct_pw = os.getenv("ADMIN_PASSWORD", "")
        try:
            _secret_pw = st.secrets.get("ADMIN_PASSWORD", "")
            if _secret_pw:
                _correct_pw = _secret_pw
        except Exception:
            pass

        if admin_pw and admin_pw == _correct_pw:
            st.success("관리자 인증 완료")
            collect_btn = st.button("데이터 수집/갱신 (요약)", type="primary", use_container_width=True)
            collect_detail_btn = st.button("전체 재무제표 수집", use_container_width=True,
                                            help="수집 성공 기업의 세부 계정과목(현금, 단기차입금 등)을 추가 수집")
            collect_qcash_btn = st.button("최신 분기 순현금 수집", use_container_width=True,
                                          help="직전 연도 최신 분기(3Q→반기→1Q) 순현금 수집")
            if os.path.exists(os.path.join(CACHE_DIR, "all_financials.pkl")):
                if st.button("캐시 초기화"):
                    for f in os.listdir(CACHE_DIR):
                        os.remove(os.path.join(CACHE_DIR, f))
                    st.cache_data.clear()
                    st.rerun()
        elif admin_pw:
            st.error("비밀번호가 올바르지 않습니다.")

# --- 데이터 수집 ---
if collect_btn:
    if not api_key:
        st.error("DART API 키를 입력해주세요.")
        st.stop()

    dart = get_dart_reader(api_key)

    with st.spinner("상장기업 리스트 조회 중..."):
        corp_list = get_corp_list(dart)
    st.info(f"총 {len(corp_list)}개 상장기업 확인")

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(idx, total, name):
        progress_bar.progress(min(idx / max(total, 1), 1.0))
        status_text.text(f"[{idx+1}/{total}] {name} 수집 중...")

    with st.spinner("재무제표 수집 중 (시간이 걸립니다)..."):
        financials = fetch_all_financials(dart, corp_list, years, progress_callback=update_progress)

    progress_bar.empty()
    status_text.empty()

    if financials.empty:
        st.warning("수집된 데이터가 없습니다.")
        st.stop()

    st.success(f"재무데이터 수집 완료: {financials['corp_name'].nunique()}개 기업")
    # 수집 후 지표 캐시 갱신
    st.cache_data.clear()

# --- 전체 재무제표 수집 ---
if collect_detail_btn:
    if not api_key:
        st.error("DART API 키를 입력해주세요.")
        st.stop()

    base_cache = os.path.join(CACHE_DIR, "all_financials.pkl")
    if not os.path.exists(base_cache):
        st.error("먼저 '데이터 수집/갱신 (요약)' 을 실행해주세요.")
        st.stop()

    dart = get_dart_reader(api_key)
    base_fin = _read_pickle_compat(base_cache)

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_detail_progress(idx, total, name):
        progress_bar.progress(min((idx + 1) / max(total, 1), 1.0))
        status_text.text(f"[{idx+1}/{total}] {name} 전체 재무제표 수집 중...")

    with st.spinner("전체 재무제표 수집 중..."):
        detail = fetch_all_detail_financials(dart, base_fin, years, progress_callback=update_detail_progress)

    progress_bar.empty()
    status_text.empty()

    if not detail.empty:
        st.success(f"전체 재무제표 수집 완료: {detail['corp_name'].nunique()}개 기업")
        st.cache_data.clear()
    else:
        st.warning("수집된 데이터가 없습니다.")

# --- 최신 분기 순현금 수집 ---
if collect_qcash_btn:
    if not api_key:
        st.error("DART API 키를 입력해주세요.")
        st.stop()

    dart = get_dart_reader(api_key)
    corp_list_cache = os.path.join(CACHE_DIR, "corp_list.pkl")
    if not os.path.exists(corp_list_cache):
        st.error("먼저 '데이터 수집/갱신 (요약)' 을 실행해주세요.")
        st.stop()

    corp_list = _read_pickle_compat(corp_list_cache)
    from datetime import datetime as _dt_tmp
    target_year = _dt_tmp.now().year - 1

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_qcash_progress(idx, total, name):
        progress_bar.progress(min((idx + 1) / max(total, 1), 1.0))
        status_text.text(f"[{idx+1}/{total}] {name} 분기 순현금 수집 중...")

    with st.spinner(f"{target_year}년 최신 분기 순현금 수집 중..."):
        qcash = fetch_quarterly_net_cash(dart, corp_list, target_year, progress_callback=update_qcash_progress)

    progress_bar.empty()
    status_text.empty()

    if not qcash.empty:
        st.success(f"분기 순현금 수집 완료: {qcash['corp_name'].nunique()}개 기업")
        st.cache_data.clear()
    else:
        st.warning("수집된 데이터가 없습니다.")

# --- 캐시 데이터 로드 & 분석 ---
from datetime import datetime as _dt

cache_path = os.path.join(CACHE_DIR, "all_financials.pkl")
metrics_cache_path = os.path.join(CACHE_DIR, "metrics.pkl")

# metrics.pkl만 있어도 조회/분석 가능
if not os.path.exists(cache_path) and not os.path.exists(metrics_cache_path):
    st.info("좌측 사이드바에서 관리자 모드로 '데이터 수집/갱신' 버튼을 눌러 데이터를 먼저 수집해주세요.")
    st.stop()

@st.cache_data(ttl=3600)
def load_financials():
    """재무제표 로드 + 병합 + 연도 필터 (1시간 메모리 캐시)."""
    if not os.path.exists(cache_path):
        return pd.DataFrame()
    fin = _read_pickle_compat(cache_path)
    detail_cache = os.path.join(CACHE_DIR, "all_financials_detail.pkl")
    if os.path.exists(detail_cache):
        detail = _read_pickle_compat(detail_cache)
        # BS(재무상태표)와 CIS(손익계산서)만 사용 (SCE자본변동표, CF현금흐름표 제외)
        # SCE에 동일 계정명이 다른 값으로 존재하여 덮어쓰기 문제 발생 방지
        if "sj_div" in detail.columns:
            detail = detail[detail["sj_div"].isin(["BS", "CIS"])]
        dedup_cols = ["corp_code", "year", "account_nm"]
        if "fs_div" in detail.columns and "fs_div" in fin.columns:
            dedup_cols.append("fs_div")
        fin = pd.concat([fin, detail], ignore_index=True).drop_duplicates(
            subset=dedup_cols, keep="last"
        )
    cutoff = _dt.now().year - 1
    fin = fin[fin["year"] <= cutoff]
    return fin

@st.cache_data(ttl=3600)
def load_metrics(_fin_hash):
    """지표 계산 결과 캐시. 디스크 캐시도 활용."""
    fin_mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0
    detail_cache = os.path.join(CACHE_DIR, "all_financials_detail.pkl")
    detail_mtime = os.path.getmtime(detail_cache) if os.path.exists(detail_cache) else 0

    # 디스크 캐시가 재무제표보다 최신이면 바로 로드
    if os.path.exists(metrics_cache_path):
        m_mtime = os.path.getmtime(metrics_cache_path)
        if m_mtime > fin_mtime and m_mtime > detail_mtime:
            return _read_pickle_compat(metrics_cache_path)

    # 재계산 (원본 재무제표 필요)
    fin = load_financials()
    if fin.empty:
        return pd.DataFrame()
    m = compute_all_metrics(fin)
    m.to_pickle(metrics_cache_path)
    return m

financials = load_financials()
# 해시값으로 데이터 변경 감지
_fin_hash = os.path.getmtime(cache_path) if os.path.exists(cache_path) else os.path.getmtime(metrics_cache_path)
metrics = load_metrics(_fin_hash)

if metrics.empty:
    st.warning("지표 계산 결과가 없습니다.")
    st.stop()

# --- 탭 구성 ---
tab_roce, tab_debt, tab_netcash, tab_detail, tab_leading, tab_regime = st.tabs(["ROCE 스프레드시트", "부채비율 스프레드시트", "순현금 추이", "개별 기업 분석", "경기선행지수", "🤖 AI Regime"])

# --- 순현금 추이 탭 ---
with tab_netcash:
    st.subheader("전체 기업 순현금 추이 (억원)")
    st.caption("순현금 = 현금및현금성자산 - 단기차입금")
    netcash_pivot = pivot_net_cash(metrics)
    if not netcash_pivot.empty:
        nc_cols = [c for c in netcash_pivot.columns if c.startswith("순현금_")]
        for c in nc_cols:
            netcash_pivot[c] = netcash_pivot[c] / 1e8

        # 분기 순현금 데이터 병합 (연간 데이터 없는 기업만)
        latest_annual_year = int(metrics["year"].max())
        qcash_cache = os.path.join(CACHE_DIR, f"quarterly_cash_{latest_annual_year + 1}.pkl")
        if not os.path.exists(qcash_cache):
            qcash_cache = os.path.join(CACHE_DIR, f"quarterly_cash_{latest_annual_year}.pkl")
        if os.path.exists(qcash_cache):
            qcash_raw = _read_pickle_compat(qcash_cache)
            qcash_year = int(os.path.basename(qcash_cache).replace("quarterly_cash_", "").replace(".pkl", ""))
            qcash_metrics = compute_quarterly_net_cash(qcash_raw)
            if not qcash_metrics.empty:
                qcash_metrics["net_cash_q"] = qcash_metrics["net_cash"] / 1e8
                q_col = f"순현금_{qcash_year}(최신분기)"
                q_map = qcash_metrics.set_index("corp_code")[["net_cash_q", "quarter"]]
                netcash_pivot = netcash_pivot.merge(q_map, left_on="corp_code", right_index=True, how="left")
                # 연간 데이터가 없는 기업만 분기 값 표시, 있으면 그대로 분기도 표시
                netcash_pivot[q_col] = netcash_pivot["net_cash_q"]
                # 분기 라벨 추가
                netcash_pivot[q_col + "_분기"] = netcash_pivot["quarter"]
                netcash_pivot = netcash_pivot.drop(columns=["net_cash_q", "quarter"])
                nc_cols = [c for c in netcash_pivot.columns if c.startswith("순현금_") and not c.endswith("_분기")]

        col1, col2 = st.columns(2)
        with col1:
            search_nc = st.text_input("기업명 검색 (순현금)", key="nc_search")
        with col2:
            if nc_cols:
                sort_order = st.selectbox("정렬 기준", ["최신 순현금 높은순", "최신 순현금 낮은순"], key="nc_sort")

        filtered_nc = netcash_pivot.copy()
        if search_nc:
            filtered_nc = search_filter(filtered_nc, "corp_name", search_nc)
        if nc_cols:
            latest_nc = filtered_nc[nc_cols].ffill(axis=1).iloc[:, -1]
            ascending = sort_order == "최신 순현금 낮은순"
            filtered_nc = filtered_nc.assign(_sort=latest_nc).sort_values("_sort", ascending=ascending).drop(columns="_sort")

        # 분기 라벨 컬럼 제거 (표시용)
        display_nc_cols = [c for c in filtered_nc.columns if not c.endswith("_분기")]
        fmt_cols = {c: "{:,.0f}" for c in display_nc_cols if c.startswith("순현금_")}

        st.dataframe(
            filtered_nc[display_nc_cols].style.format(fmt_cols, na_rep="-"),
            use_container_width=True,
            height=600,
            hide_index=True,
        )
        st.download_button(
            "CSV 다운로드",
            filtered_nc[display_nc_cols].to_csv(index=False).encode("utf-8-sig"),
            "net_cash_all.csv",
            "text/csv",
            key="dl_netcash",
        )

        # --- 주목할 기업: 순현금 급증 ---
        st.divider()
        st.subheader("주목할 기업 — 순현금 급증")
        st.caption("최신 순현금이 이전 3개년 평균 대비 30% 이상 증가한 기업")

        # 연간 순현금 컬럼에서 이전 3년 평균 계산
        annual_nc_cols = [c for c in nc_cols if "(최신분기)" not in c]
        q_col_name = [c for c in nc_cols if "(최신분기)" in c]

        if len(annual_nc_cols) >= 4:
            prev_3y_cols = annual_nc_cols[-4:-1]  # 최신 연간 제외, 그 이전 3년
            latest_col = annual_nc_cols[-1]
        elif len(annual_nc_cols) >= 2:
            prev_3y_cols = annual_nc_cols[:-1]
            latest_col = annual_nc_cols[-1]
        else:
            prev_3y_cols = []
            latest_col = annual_nc_cols[-1] if annual_nc_cols else None

        if prev_3y_cols and latest_col:
            spotlight = netcash_pivot.copy()
            spotlight["이전3년평균"] = spotlight[prev_3y_cols].mean(axis=1)
            # 최신값: 분기 데이터가 있으면 분기 우선, 없으면 연간
            if q_col_name:
                spotlight["최신순현금"] = spotlight[q_col_name[0]].fillna(spotlight[latest_col])
                spotlight["기준"] = spotlight.apply(
                    lambda r: r.get(q_col_name[0] + "_분기", "") if pd.notna(r[q_col_name[0]]) else f"{latest_col.replace('순현금_', '')}년 연간",
                    axis=1,
                )
            else:
                spotlight["최신순현금"] = spotlight[latest_col]
                spotlight["기준"] = f"{latest_col.replace('순현금_', '')}년 연간"

            # 3년 평균 양수 & 최신값 존재
            valid = spotlight.dropna(subset=["최신순현금", "이전3년평균"])
            valid = valid[valid["이전3년평균"] > 0]
            valid["증가율(%)"] = (valid["최신순현금"] - valid["이전3년평균"]) / valid["이전3년평균"] * 100

            threshold = st.slider("최소 증가율 (%)", 30, 500, 30, step=10, key="nc_threshold")
            notable = valid[valid["증가율(%)"] >= threshold].sort_values("증가율(%)", ascending=False)

            st.metric("해당 기업 수", f"{len(notable)}개")

            if not notable.empty:
                notable_display = notable[["corp_name", "이전3년평균", "최신순현금", "증가율(%)", "기준"]].copy()
                notable_display.columns = ["기업명", "이전3년평균(억)", "최신순현금(억)", "증가율(%)", "기준"]

                st.dataframe(
                    notable_display.style.format({
                        "이전3년평균(억)": "{:,.0f}",
                        "최신순현금(억)": "{:,.0f}",
                        "증가율(%)": "{:,.0f}",
                    }, na_rep="-"),
                    use_container_width=True,
                    height=500,
                    hide_index=True,
                )
                st.download_button(
                    "주목 기업 CSV 다운로드",
                    notable_display.to_csv(index=False).encode("utf-8-sig"),
                    "notable_net_cash.csv",
                    "text/csv",
                    key="dl_notable_nc",
                )
            else:
                st.info("조건에 해당하는 기업이 없습니다.")

# --- ROCE 탭 ---
with tab_roce:
    st.subheader("전체 기업 ROCE 추이 (%)")
    roce_pivot = pivot_roce(metrics)
    if not roce_pivot.empty:
        # 필터
        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("기업명 검색 (ROCE)", key="roce_search")
        with col2:
            roce_cols = [c for c in roce_pivot.columns if c.startswith("ROCE_")]
            if roce_cols:
                latest = roce_cols[-1]
                min_roce = st.number_input("최소 ROCE (%)", value=-100.0, key="min_roce")

        filtered = roce_pivot.copy()
        if search:
            filtered = search_filter(filtered, "corp_name", search)
        if roce_cols:
            # 각 기업의 가장 최근 ROCE 값으로 필터링 (NaN 무시)
            latest_roce = filtered[roce_cols].ffill(axis=1).iloc[:, -1]
            filtered = filtered[latest_roce >= min_roce]
            filtered = filtered.assign(_sort=latest_roce).sort_values("_sort", ascending=False).drop(columns="_sort")

        # 자본잠식 셀을 "자본잠식" 텍스트로 치환
        imp_cols = [c for c in filtered.columns if c.startswith("imp_")]
        display_roce = filtered[["corp_code", "corp_name"] + roce_cols].copy()
        for rc in roce_cols:
            year_str = rc.replace("ROCE_", "")
            ic = f"imp_{year_str}"
            if ic in filtered.columns:
                mask = filtered[ic].fillna(False).astype(bool)
                display_roce.loc[mask, rc] = None
                display_roce[rc] = display_roce[rc].apply(
                    lambda v, m=mask: v if not pd.isna(v) else None
                )

        # 자본잠식 표시용 포맷 함수
        def roce_formatter(v, imp_flags, col):
            year_str = col.replace("ROCE_", "")
            ic = f"imp_{year_str}"
            if ic in imp_flags.columns:
                idx = v.name if hasattr(v, 'name') else None
            return v

        # 자본잠식 셀 스타일링
        def style_impaired_roce(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for rc in roce_cols:
                year_str = rc.replace("ROCE_", "")
                ic = f"imp_{year_str}"
                if ic in filtered.columns:
                    mask = filtered[ic].reindex(df.index).fillna(False).astype(bool)
                    styles.loc[mask, rc] = "background-color: #ffcccc; color: #cc0000"
            return styles

        # na_rep를 "자본잠식"으로, 자본잠식 셀은 NaN으로 처리됨
        imp_map = {}
        for rc in roce_cols:
            year_str = rc.replace("ROCE_", "")
            ic = f"imp_{year_str}"
            if ic in filtered.columns:
                imp_mask = filtered[ic].fillna(False).astype(bool)
                display_roce.loc[imp_mask, rc] = float("nan")
                imp_map[rc] = imp_mask

        # 커스텀 포맷: 자본잠식이면 "자본잠식", 아니면 숫자
        def format_roce_cell(val, rc, row_idx):
            if rc in imp_map and row_idx in imp_map[rc].index and imp_map[rc].loc[row_idx]:
                return "자본잠식"
            if pd.isna(val):
                return "-"
            return f"{val:.1f}"

        # 문자열 DataFrame으로 변환
        display_str = display_roce[["corp_code", "corp_name"]].copy()
        for rc in roce_cols:
            display_str[rc] = [
                format_roce_cell(display_roce.at[idx, rc], rc, idx)
                for idx in display_roce.index
            ]

        st.dataframe(
            display_str,
            use_container_width=True,
            height=600,
            hide_index=True,
        )
        st.download_button(
            "CSV 다운로드",
            filtered.to_csv(index=False).encode("utf-8-sig"),
            "roce_all.csv",
            "text/csv",
            key="dl_roce",
        )

# --- 부채비율 탭 ---
with tab_debt:
    st.subheader("전체 기업 부채비율 추이 (%)")
    debt_pivot = pivot_debt_ratio(metrics)
    if not debt_pivot.empty:
        col1, col2 = st.columns(2)
        with col1:
            search_d = st.text_input("기업명 검색 (부채비율)", key="debt_search")
        with col2:
            debt_cols = [c for c in debt_pivot.columns if c.startswith("부채비율_")]
            if debt_cols:
                latest_d = debt_cols[-1]
                max_debt = st.number_input("최대 부채비율 (%)", value=500.0, key="max_debt")

        filtered_d = debt_pivot.copy()
        if search_d:
            filtered_d = search_filter(filtered_d, "corp_name", search_d)
        if debt_cols:
            latest_debt = filtered_d[debt_cols].ffill(axis=1).iloc[:, -1]
            filtered_d = filtered_d[latest_debt <= max_debt]
            filtered_d = filtered_d.assign(_sort=latest_debt).sort_values("_sort", ascending=True).drop(columns="_sort")

        # 자본잠식 처리
        imp_map_d = {}
        for dc in debt_cols:
            year_str = dc.replace("부채비율_", "")
            ic = f"imp_{year_str}"
            if ic in filtered_d.columns:
                imp_map_d[dc] = filtered_d[ic].fillna(False).astype(bool)

        def format_debt_cell(val, dc, row_idx):
            if dc in imp_map_d and row_idx in imp_map_d[dc].index and imp_map_d[dc].loc[row_idx]:
                return "자본잠식"
            if pd.isna(val):
                return "-"
            return f"{val:.1f}"

        display_debt = filtered_d[["corp_code", "corp_name"]].copy()
        for dc in debt_cols:
            display_debt[dc] = [
                format_debt_cell(filtered_d.at[idx, dc], dc, idx)
                for idx in filtered_d.index
            ]

        st.dataframe(
            display_debt,
            use_container_width=True,
            height=600,
            hide_index=True,
        )
        st.download_button(
            "CSV 다운로드",
            filtered_d.to_csv(index=False).encode("utf-8-sig"),
            "debt_ratio_all.csv",
            "text/csv",
            key="dl_debt",
        )

# --- 개별 기업 분석 탭 ---
with tab_detail:
    st.subheader("개별 기업 상세 분석")
    corp_names = sorted(metrics["corp_name"].unique())

    # 기업 검색 (영문 약어 자동 변환)
    search_detail = st.text_input("기업명 검색", key="detail_search", placeholder="SK바이오팜 → 에스케이바이오팜 자동 변환")
    if search_detail:
        queries = expand_search(search_detail)
        filtered_names = [n for n in corp_names if any(q.lower() in n.lower() for q in queries)]
    else:
        filtered_names = corp_names

    selected = st.selectbox("기업 선택", filtered_names if filtered_names else ["검색 결과 없음"])

    if selected and selected != "검색 결과 없음":
        corp_data = metrics[metrics["corp_name"] == selected].sort_values("year")

        if corp_data.empty:
            st.warning("해당 기업의 데이터가 없습니다.")
        else:
            # ROCE + 부채비율 듀얼 차트
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(
                    x=corp_data["year"],
                    y=corp_data["roce"],
                    name="ROCE (%)",
                    marker_color="steelblue",
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=corp_data["year"],
                    y=corp_data["debt_ratio"],
                    name="부채비율 (%)",
                    mode="lines+markers",
                    line=dict(color="tomato", width=2),
                ),
                secondary_y=True,
            )
            fig.update_layout(title=f"{selected} - ROCE & 부채비율 추이", height=450)
            fig.update_yaxes(title_text="ROCE (%)", secondary_y=False)
            fig.update_yaxes(title_text="부채비율 (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # --- 자동 분석 요약 ---
            st.subheader("분석 요약")

            latest = corp_data.iloc[-1]
            first = corp_data.iloc[0]
            stock_code = str(latest.get("stock_code", "")).zfill(6) if pd.notna(latest.get("stock_code")) else None

            # 주가 및 시가총액 조회
            marcap = None
            current_price = None
            price_chg_1m = None
            price_chg_1y = None
            if stock_code and stock_code != "000000":
                try:
                    krx_list = fdr.StockListing("KRX")
                    krx_row = krx_list[krx_list["Code"] == stock_code]
                    if not krx_row.empty and "Marcap" in krx_row.columns:
                        marcap = krx_row["Marcap"].iloc[0]
                    price_df = fdr.DataReader(stock_code, "2024-01-01")
                    if not price_df.empty:
                        current_price = price_df["Close"].iloc[-1]
                        if len(price_df) >= 20:
                            price_chg_1m = (current_price / price_df["Close"].iloc[-20] - 1) * 100
                        if len(price_df) >= 250:
                            price_chg_1y = (current_price / price_df["Close"].iloc[-250] - 1) * 100
                except:
                    pass

            # 핵심 지표
            latest_roce = latest.get("roce")
            latest_debt = latest.get("debt_ratio")
            latest_nc = latest.get("net_cash")
            latest_equity = latest.get("total_equity")
            latest_ebit = latest.get("ebit")
            first_equity = first.get("total_equity")

            # 분석 포인트 생성
            highlights = []
            warnings = []

            # ROCE 분석
            if pd.notna(latest_roce):
                if latest_roce >= 20:
                    highlights.append(f"ROCE {latest_roce:.1f}%로 우수한 자본효율성")
                elif latest_roce >= 10:
                    highlights.append(f"ROCE {latest_roce:.1f}%로 양호한 수준")
                elif latest_roce < 0:
                    warnings.append(f"ROCE {latest_roce:.1f}%로 영업적자 상태")

            # 부채비율 분석
            if pd.notna(latest_debt):
                if latest_debt <= 50:
                    highlights.append(f"부채비율 {latest_debt:.0f}%로 재무구조 탄탄")
                elif latest_debt <= 100:
                    highlights.append(f"부채비율 {latest_debt:.0f}%로 적정 수준")
                elif latest_debt > 200:
                    warnings.append(f"부채비율 {latest_debt:.0f}%로 재무 부담")

            # 순현금 분석
            latest_stb = latest.get("short_term_borrowings")
            if pd.notna(latest_nc):
                if latest_nc > 0:
                    if pd.notna(latest_stb) and latest_stb == 0:
                        highlights.append(f"순현금 {latest_nc/1e8:,.0f}억원 (무차입 경영)")
                    else:
                        highlights.append(f"순현금 {latest_nc/1e8:,.0f}억원 보유")
                else:
                    warnings.append(f"순부채 {abs(latest_nc)/1e8:,.0f}억원")

            # 자기자본 성장
            if pd.notna(latest_equity) and pd.notna(first_equity) and first_equity > 0:
                equity_growth = latest_equity / first_equity
                years = int(latest["year"]) - int(first["year"])
                if years > 0 and equity_growth >= 2:
                    highlights.append(f"자기자본 {years}년간 {equity_growth:.1f}배 성장")

            # EBIT 성장 (최근 2년 비교)
            if len(corp_data) >= 2:
                prev = corp_data.iloc[-2]
                prev_ebit = prev.get("ebit")
                if pd.notna(latest_ebit) and pd.notna(prev_ebit) and prev_ebit > 0:
                    ebit_growth = (latest_ebit / prev_ebit - 1) * 100
                    if ebit_growth >= 50:
                        highlights.append(f"EBIT 전년 대비 {ebit_growth:.0f}% 성장")
                    elif ebit_growth <= -30:
                        warnings.append(f"EBIT 전년 대비 {abs(ebit_growth):.0f}% 감소")

            # 주가 모멘텀
            if price_chg_1m is not None and price_chg_1m >= 20:
                highlights.append(f"최근 1개월 주가 {price_chg_1m:+.0f}% 상승")

            # 밸류에이션 계산
            pbr = None
            ev_ebit = None
            if marcap and pd.notna(latest_equity) and latest_equity > 0:
                pbr = marcap / latest_equity
            if marcap and pd.notna(latest_ebit) and latest_ebit > 0 and pd.notna(latest_nc):
                ev = marcap - latest_nc
                ev_ebit = ev / latest_ebit

            # 화면 출력
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**핵심 포인트**")
                if highlights:
                    for h in highlights:
                        st.markdown(f"- ✅ {h}")
                if warnings:
                    for w in warnings:
                        st.markdown(f"- ⚠️ {w}")
                if not highlights and not warnings:
                    st.markdown("- 분석 데이터 부족")

            with col_right:
                st.markdown("**밸류에이션**")
                if current_price:
                    st.markdown(f"- 현재가: **{current_price:,.0f}원**")
                if marcap:
                    st.markdown(f"- 시가총액: **{marcap/1e8:,.0f}억원**")
                if pbr:
                    pbr_eval = "저평가" if pbr < 1 else ("적정" if pbr < 3 else "고평가")
                    st.markdown(f"- PBR: **{pbr:.2f}배** ({pbr_eval})")
                if ev_ebit:
                    ev_eval = "저평가" if ev_ebit < 10 else ("적정" if ev_ebit < 20 else "고평가")
                    st.markdown(f"- EV/EBIT: **{ev_ebit:.1f}배** ({ev_eval})")

            st.divider()

            # 상세 테이블 (천단위 쉼표)
            display_cols = ["year", "ebit", "total_assets", "total_liabilities", "total_equity",
                            "cash", "short_term_borrowings", "net_cash", "roce", "debt_ratio"]
            display_cols = [c for c in display_cols if c in corp_data.columns]
            money_cols = ["ebit", "total_assets", "total_liabilities", "total_equity",
                          "cash", "short_term_borrowings", "net_cash"]
            pct_cols = ["roce", "debt_ratio"]
            fmt = {}
            for c in money_cols:
                if c in display_cols:
                    fmt[c] = "{:,.0f}"
            for c in pct_cols:
                if c in display_cols:
                    fmt[c] = "{:,.1f}"

            rename_map = {
                "year": "연도", "ebit": "EBIT", "total_assets": "총자산",
                "total_liabilities": "총부채", "total_equity": "자기자본",
                "cash": "현금성자산", "short_term_borrowings": "단기차입금",
                "net_cash": "순현금", "roce": "ROCE(%)", "debt_ratio": "부채비율(%)",
            }
            renamed_fmt = {rename_map.get(k, k): v for k, v in fmt.items()}

            st.dataframe(
                corp_data[display_cols].rename(columns=rename_map).style.format(renamed_fmt, na_rep="-"),
                use_container_width=True,
                hide_index=True,
            )

            # 연도별 재무상태표(자산부채표) 팝업
            st.subheader("연도별 재무상태표 상세")
            if financials.empty:
                st.info("재무상태표 상세 데이터가 없습니다. 관리자 모드에서 데이터를 수집하면 확인할 수 있습니다.")
                selected_year = None
            else:
                available_years = sorted(corp_data["year"].unique())
                selected_year = st.selectbox("연도 선택", available_years, index=len(available_years)-1, key="bs_year")

            if selected_year:
                corp_code = corp_data["corp_code"].iloc[0]
                # 해당 기업·연도의 재무상태표 항목 추출
                bs_data = financials[
                    (financials["corp_code"] == corp_code) &
                    (financials["year"] == selected_year)
                ]
                # sj_div가 있으면 BS(재무상태표)만, 없으면 account_nm으로 필터
                if "sj_div" in bs_data.columns:
                    bs_data = bs_data[bs_data["sj_div"] == "BS"]
                else:
                    bs_keywords = ["자산", "부채", "자본", "현금", "차입", "유동", "비유동",
                                   "재고", "매출채권", "투자", "유형자산", "무형자산", "이익잉여금"]
                    mask = bs_data["account_nm"].apply(
                        lambda x: any(kw in str(x) for kw in bs_keywords) if pd.notna(x) else False
                    )
                    bs_data = bs_data[mask]

                if not bs_data.empty:
                    # ord 컬럼으로 재무제표 원래 순서 정렬
                    bs_all = bs_data.copy()
                    if "ord" in bs_all.columns:
                        bs_all["_ord_num"] = pd.to_numeric(bs_all["ord"], errors="coerce")
                        bs_all = bs_all.sort_values("_ord_num")

                    bs_all = bs_all[["account_nm", "thstrm_amount"]].copy()
                    bs_all.columns = ["계정과목", "금액"]
                    bs_all["금액"] = pd.to_numeric(
                        bs_all["금액"].astype(str).str.replace(",", ""), errors="coerce"
                    )
                    bs_all["계정_clean"] = bs_all["계정과목"].str.replace(" ", "")

                    # 자산총계 위치를 기준으로 자산 / 부채·자본 분리
                    # ord 순서상 자산총계까지 = 자산, 그 이후 = 부채·자본
                    asset_end_idx = None
                    for i, row_bs in enumerate(bs_all.itertuples()):
                        if "자산총계" in row_bs.계정_clean:
                            asset_end_idx = i
                            break

                    if asset_end_idx is not None:
                        # 자산총계까지(포함) = 자산, 나머지 = 부채·자본
                        assets = bs_all.iloc[:asset_end_idx + 1].reset_index(drop=True)
                        liab_eq = bs_all.iloc[asset_end_idx + 1:].reset_index(drop=True)
                    else:
                        # 자산총계를 못 찾으면 키워드 기반 분류
                        liab_eq_kw = ["부채", "자본", "차입", "매입채무", "미지급", "선수", "충당",
                                      "이익잉여금", "자본금", "자본잉여금", "기타포괄", "리스부채", "비지배지분"]
                        def classify(name):
                            for kw in liab_eq_kw:
                                if kw in name:
                                    return "liab_eq"
                            return "asset"
                        bs_all["_cls"] = bs_all["계정_clean"].apply(classify)
                        assets = bs_all[bs_all["_cls"] == "asset"].reset_index(drop=True)
                        liab_eq = bs_all[bs_all["_cls"] == "liab_eq"].reset_index(drop=True)

                    # 합계 항목 굵게 표시를 위한 스타일 함수
                    summary_keywords = ["총계", "소계", "합계", "소유주지분"]

                    def highlight_summary(row, col_name):
                        name = str(row.iloc[0]).replace(" ", "")
                        if any(kw in name for kw in summary_keywords):
                            return ["font-weight: bold; background-color: #f0f0f0"] * len(row)
                        return [""] * len(row)

                    assets_display = assets[["계정과목", "금액"]].copy()
                    assets_display.columns = ["자산 계정과목", "자산 금액"]
                    liab_eq_display = liab_eq[["계정과목", "금액"]].copy()
                    liab_eq_display.columns = ["부채·자본 계정과목", "부채·자본 금액"]

                    with st.expander(f"{selected} - {selected_year}년 재무상태표", expanded=True):
                        col_left, col_right = st.columns(2)
                        with col_left:
                            st.markdown("**자산**")
                            st.dataframe(
                                assets_display.style
                                    .format({"자산 금액": "{:,.0f}"}, na_rep="-")
                                    .apply(highlight_summary, col_name="자산 계정과목", axis=1),
                                use_container_width=True,
                                hide_index=True,
                            )
                        with col_right:
                            st.markdown("**부채 · 자본**")
                            st.dataframe(
                                liab_eq_display.style
                                    .format({"부채·자본 금액": "{:,.0f}"}, na_rep="-")
                                    .apply(highlight_summary, col_name="부채·자본 계정과목", axis=1),
                                use_container_width=True,
                                hide_index=True,
                            )
                else:
                    st.info(f"{selected_year}년 재무상태표 데이터가 없습니다.")

            # 주가 차트 + 후행 PER
            stock_code = corp_data["stock_code"].iloc[0] if "stock_code" in corp_data.columns else None
            if stock_code and pd.notna(stock_code) and stock_code != "":
                _scode = str(stock_code).zfill(6)
                with st.spinner("주가 데이터 조회 중..."):
                    price_df = fetch_stock_price(_scode, years)
                if not price_df.empty and "Close" in price_df.columns:
                    current_price = price_df["Close"].iloc[-1]

                    # 후행 PER 계산 (TTM: 최근 4개 분기 당기순이익 합산)
                    from analyzer import _extract_amount, _parse_number
                    _corp_code = corp_data["corp_code"].iloc[0]

                    _ni_names = ["지배기업의소유주에게귀속되는당기순이익", "지배기업소유주당기순이익",
                                 "당기순이익(손실)", "당기순이익", "당기순손익"]

                    def _get_ni_from_fs(fs_df):
                        """재무제표 DataFrame에서 당기순이익 추출."""
                        if fs_df is None or fs_df.empty:
                            return None
                        fs_df = fs_df.copy()
                        fs_df["_clean"] = fs_df["account_nm"].str.replace(" ", "")
                        for nm in _ni_names:
                            nc = nm.replace(" ", "")
                            exact = fs_df[fs_df["_clean"] == nc]
                            if not exact.empty:
                                v = _parse_number(exact.iloc[0].get("thstrm_amount"))
                                if v is not None:
                                    return v
                            starts = fs_df[fs_df["_clean"].str.startswith(nc, na=False)]
                            if not starts.empty:
                                v = _parse_number(starts.iloc[0].get("thstrm_amount"))
                                if v is not None:
                                    return v
                        return None

                    # OpenDart에서 누적 당기순이익 조회
                    # 11013=1Q, 11012=반기(1Q+2Q), 11014=3Q(1Q+2Q+3Q), 11011=연간(1Q+2Q+3Q+4Q)
                    # 개별 분기 = 해당 누적 - 직전 누적
                    # TTM = 최근 4개 개별 분기 합산
                    import datetime as _dtm
                    ttm_ni = None
                    ttm_label = ""
                    q_details = []  # (분기명, 금액) 리스트

                    try:
                        dart = get_dart_reader(api_key)
                        cur_y = _dtm.datetime.now().year

                        # 최근 2개년의 누적 당기순이익 수집
                        # {(연도, 보고서코드): 누적NI}
                        cum_data = {}
                        for y in [cur_y, cur_y - 1, cur_y - 2]:
                            for rcode, rlabel in [("11013","1Q"), ("11012","반기"), ("11014","3Q"), ("11011","연간")]:
                                try:
                                    fs = dart.finstate(_corp_code, y, reprt_code=rcode)
                                    ni = _get_ni_from_fs(fs)
                                    if ni is not None:
                                        cum_data[(y, rcode)] = ni
                                except Exception:
                                    pass

                        # 개별 분기 당기순이익 계산
                        # Q1 = 1Q누적, Q2 = 반기 - 1Q, Q3 = 3Q - 반기, Q4 = 연간 - 3Q
                        individual_quarters = []  # (연도, 분기, 금액) - 최신순

                        for y in [cur_y, cur_y - 1, cur_y - 2]:
                            q4 = None
                            if (y, "11011") in cum_data and (y, "11014") in cum_data:
                                q4 = cum_data[(y, "11011")] - cum_data[(y, "11014")]
                            q3 = None
                            if (y, "11014") in cum_data and (y, "11012") in cum_data:
                                q3 = cum_data[(y, "11014")] - cum_data[(y, "11012")]
                            q2 = None
                            if (y, "11012") in cum_data and (y, "11013") in cum_data:
                                q2 = cum_data[(y, "11012")] - cum_data[(y, "11013")]
                            q1 = None
                            if (y, "11013") in cum_data:
                                q1 = cum_data[(y, "11013")]

                            # 역순(Q4→Q1)으로 추가 (나중에 최신 4개 선택)
                            if q4 is not None:
                                individual_quarters.append((y, "Q4", q4))
                            if q3 is not None:
                                individual_quarters.append((y, "Q3", q3))
                            if q2 is not None:
                                individual_quarters.append((y, "Q2", q2))
                            if q1 is not None:
                                individual_quarters.append((y, "Q1", q1))

                        # 최신순 정렬 후 상위 4개 선택
                        q_order = {"Q4": 4, "Q3": 3, "Q2": 2, "Q1": 1}
                        individual_quarters.sort(key=lambda x: (x[0], q_order[x[1]]), reverse=True)

                        if len(individual_quarters) >= 4:
                            top4 = individual_quarters[:4]
                            ttm_ni = sum(q[2] for q in top4)
                            q_details = top4
                            q_strs = [f"{q[0]}년{q[1]}({q[2]:,.0f})" for q in reversed(top4)]
                            ttm_label = "TTM: " + " + ".join(q_strs)
                    except Exception:
                        pass

                    # fallback: 연간 사업보고서
                    if ttm_ni is None and not financials.empty:
                        latest_year_data = corp_data.sort_values("year", ascending=False).iloc[0]
                        _ly = int(latest_year_data["year"])
                        _fs = financials[
                            (financials["corp_code"] == _corp_code) &
                            (financials["year"] == _ly)
                        ]
                        ttm_ni = _extract_amount(_fs, _ni_names)
                        ttm_label = f"{_ly}년 연간 (분기 데이터 부족)"

                    # 발행주식수: KRX 상장정보에서 가져오기
                    trailing_per = None
                    try:
                        krx_list = fdr.StockListing("KRX")
                        krx_row = krx_list[krx_list["Code"] == _scode]
                        if krx_row.empty:
                            krx_row = krx_list[krx_list["Code"].str.strip() == _scode]
                        if not krx_row.empty and "Stocks" in krx_row.columns:
                            shares = int(krx_row["Stocks"].iloc[0])
                        elif not krx_row.empty and "SharesOutstanding" in krx_row.columns:
                            shares = int(krx_row["SharesOutstanding"].iloc[0])
                        else:
                            shares = None

                        if ttm_ni and shares and ttm_ni != 0:
                            eps = ttm_ni / shares
                            if eps > 0:
                                trailing_per = current_price / eps
                    except Exception:
                        pass

                    # PER 표시
                    per_text = f"후행 PER(TTM): **{trailing_per:.1f}배**" if trailing_per else "후행 PER: 산출 불가 (순이익 적자 또는 데이터 부족)"
                    ni_display = f"{ttm_ni:,.0f}" if ttm_ni else "-"
                    st.markdown(f"현재가: **{current_price:,.0f}원** | {per_text}")
                    st.caption(f"TTM 당기순이익: {ni_display}원 | {ttm_label}")

                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(
                        x=price_df.index, y=price_df["Close"],
                        mode="lines", name="종가",
                        line=dict(color="darkgreen"),
                    ))
                    fig_price.update_layout(title=f"{selected} 주가 추이", height=350)
                    st.plotly_chart(fig_price, use_container_width=True)

# --- 경기선행지수 탭 ---
with tab_leading:
    st.subheader("경기선행지수 모니터링")
    st.caption("OECD, 컨퍼런스보드, 한국 경기선행지수의 금융시장 관련 구성요소")

    # API 키 확인
    fred_key = os.getenv("FRED_API_KEY", "")
    bok_key = os.getenv("BOK_API_KEY", "")
    try:
        fred_key = st.secrets.get("FRED_API_KEY", fred_key)
        bok_key = st.secrets.get("BOK_API_KEY", bok_key)
    except Exception:
        pass

    # 환경변수로 전달 (leading_indicators 모듈에서 사용)
    if fred_key:
        os.environ["FRED_API_KEY"] = fred_key
    if bok_key:
        os.environ["BOK_API_KEY"] = bok_key

    if not fred_key and not bok_key:
        st.warning("FRED_API_KEY 또는 BOK_API_KEY가 설정되지 않았습니다. 환경변수 또는 Streamlit Secrets에 API 키를 설정해주세요.")
        st.info("""
**API 키 발급 안내**
- FRED API: https://fred.stlouisfed.org/docs/api/api_key.html (무료)
- 한국은행 ECOS API: https://ecos.bok.or.kr/api/ (무료)
        """)

    # 3개 섹션으로 구분
    st.markdown("---")

    # 1. OECD 경기선행지수
    st.markdown("### OECD 경기선행지수 (금융시장 구성요소)")
    st.caption("주가지수, 단기금리, 장단기 금리차")

    if fred_key:
        with st.spinner("OECD 지표 조회 중..."):
            oecd_indicators = fetch_oecd_indicators()

        if oecd_indicators:
            oecd_latest = get_latest_values(oecd_indicators)
            if not oecd_latest.empty:
                # 변화 방향 표시
                def format_change(row):
                    if pd.isna(row["변화"]):
                        return "-"
                    arrow = "▲" if row["변화"] > 0 else ("▼" if row["변화"] < 0 else "─")
                    color = "red" if row["변화"] > 0 else ("blue" if row["변화"] < 0 else "gray")
                    return f"{arrow} {row['변화']:+.2f}"

                oecd_display = oecd_latest.copy()
                oecd_display["변화"] = oecd_display.apply(format_change, axis=1)
                oecd_display["변화율(%)"] = oecd_display["변화율(%)"].apply(
                    lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"
                )
                oecd_display["최신값"] = oecd_display["최신값"].apply(lambda x: f"{x:,.2f}")
                oecd_display["전월값"] = oecd_display["전월값"].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else "-"
                )

                st.dataframe(
                    oecd_display[["지표", "최신값", "기준일", "변화", "변화율(%)"]],
                    use_container_width=True,
                    hide_index=True,
                )

            # 차트 - 주가지수와 금리 지표 분리 (스케일 차이)
            oecd_history = get_indicator_history(oecd_indicators, months=36)
            if not oecd_history.empty:
                # 주가지수와 금리 지표 분리
                price_cols = [c for c in oecd_history.columns if "S&P" in c or "주가" in c]
                rate_cols = [c for c in oecd_history.columns if c not in price_cols]

                # 주가지수 차트
                if price_cols:
                    fig_oecd_price = go.Figure()
                    for col in price_cols:
                        fig_oecd_price.add_trace(go.Scatter(
                            x=oecd_history.index, y=oecd_history[col],
                            mode="lines", name=col,
                            line=dict(color="steelblue"),
                        ))
                    fig_oecd_price.update_layout(
                        title="OECD - 주가지수 (S&P 500)",
                        height=300,
                        yaxis_title="지수",
                    )
                    st.plotly_chart(fig_oecd_price, use_container_width=True)

                # 금리 지표 차트
                if rate_cols:
                    fig_oecd_rate = go.Figure()
                    colors = ["tomato", "green"]
                    for i, col in enumerate(rate_cols):
                        fig_oecd_rate.add_trace(go.Scatter(
                            x=oecd_history.index, y=oecd_history[col],
                            mode="lines", name=col,
                            line=dict(color=colors[i % len(colors)]),
                        ))
                    fig_oecd_rate.update_layout(
                        title="OECD - 금리 지표",
                        height=300,
                        yaxis_title="금리 (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    # 장단기 금리차 0선 표시
                    fig_oecd_rate.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    st.plotly_chart(fig_oecd_rate, use_container_width=True)
        else:
            st.info("OECD 지표 데이터를 불러올 수 없습니다.")
    else:
        st.info("FRED API 키가 필요합니다.")

    st.markdown("---")

    # 2. 컨퍼런스보드 경기선행지수
    st.markdown("### 컨퍼런스보드 경기선행지수 (금융시장 구성요소)")
    st.caption("S&P 500, 실질 M2 통화공급, 장단기 금리차")

    if fred_key:
        with st.spinner("컨퍼런스보드 지표 조회 중..."):
            cb_indicators = fetch_conference_board_indicators()

        if cb_indicators:
            cb_latest = get_latest_values(cb_indicators)
            if not cb_latest.empty:
                cb_display = cb_latest.copy()
                cb_display["변화"] = cb_display.apply(
                    lambda row: f"{'▲' if row['변화'] > 0 else ('▼' if row['변화'] < 0 else '─')} {row['변화']:+.2f}" if pd.notna(row["변화"]) else "-",
                    axis=1
                )
                cb_display["변화율(%)"] = cb_display["변화율(%)"].apply(
                    lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"
                )
                cb_display["최신값"] = cb_display["최신값"].apply(lambda x: f"{x:,.2f}")
                cb_display["전월값"] = cb_display["전월값"].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) else "-"
                )

                st.dataframe(
                    cb_display[["지표", "최신값", "기준일", "변화", "변화율(%)"]],
                    use_container_width=True,
                    hide_index=True,
                )

            # 차트 - 주가/M2와 금리 지표 분리 (스케일 차이)
            cb_history = get_indicator_history(cb_indicators, months=36)
            if not cb_history.empty:
                # 대형 스케일과 금리 지표 분리
                large_cols = [c for c in cb_history.columns if "S&P" in c or "M2" in c]
                rate_cols = [c for c in cb_history.columns if "금리" in c]

                # S&P 500 & 실질 M2 차트 (듀얼 축)
                if large_cols:
                    fig_cb_large = make_subplots(specs=[[{"secondary_y": True}]])
                    colors = {"S&P 500": "purple", "실질 M2 통화공급": "orange"}
                    for col in large_cols:
                        secondary = "M2" in col
                        fig_cb_large.add_trace(
                            go.Scatter(
                                x=cb_history.index, y=cb_history[col],
                                mode="lines", name=col,
                                line=dict(color=colors.get(col, "gray")),
                            ),
                            secondary_y=secondary,
                        )
                    fig_cb_large.update_layout(
                        title="컨퍼런스보드 - S&P 500 & 실질 M2",
                        height=300,
                    )
                    fig_cb_large.update_yaxes(title_text="S&P 500", secondary_y=False)
                    fig_cb_large.update_yaxes(title_text="실질 M2", secondary_y=True)
                    st.plotly_chart(fig_cb_large, use_container_width=True)

                # 금리 지표 차트
                if rate_cols:
                    fig_cb_rate = go.Figure()
                    for col in rate_cols:
                        fig_cb_rate.add_trace(go.Scatter(
                            x=cb_history.index, y=cb_history[col],
                            mode="lines", name=col,
                            line=dict(color="teal"),
                        ))
                    fig_cb_rate.update_layout(
                        title="컨퍼런스보드 - 장단기 금리차",
                        height=300,
                        yaxis_title="금리차 (%)",
                    )
                    fig_cb_rate.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    st.plotly_chart(fig_cb_rate, use_container_width=True)
        else:
            st.info("컨퍼런스보드 지표 데이터를 불러올 수 없습니다.")
    else:
        st.info("FRED API 키가 필요합니다.")

    st.markdown("---")

    # 3. 한국 경기선행지수
    st.markdown("### 한국 경기선행지수 (금융시장 구성요소)")
    st.caption("경제심리지수, 국제원자재가격, KOSPI, 장단기 금리차")

    with st.spinner("한국 경기선행지수 조회 중..."):
        kr_indicators = fetch_korea_indicators()

    if kr_indicators:
        kr_latest = get_latest_values(kr_indicators)
        if not kr_latest.empty:
            kr_display = kr_latest.copy()
            kr_display["변화"] = kr_display.apply(
                lambda row: f"{'▲' if row['변화'] > 0 else ('▼' if row['변화'] < 0 else '─')} {row['변화']:+.2f}" if pd.notna(row["변화"]) else "-",
                axis=1
            )
            kr_display["변화율(%)"] = kr_display["변화율(%)"].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"
            )
            kr_display["최신값"] = kr_display["최신값"].apply(lambda x: f"{x:,.2f}")
            kr_display["전월값"] = kr_display["전월값"].apply(
                lambda x: f"{x:,.2f}" if pd.notna(x) else "-"
            )

            st.dataframe(
                kr_display[["지표", "최신값", "기준일", "변화", "변화율(%)"]],
                use_container_width=True,
                hide_index=True,
            )

        # 차트
        kr_history = get_indicator_history(kr_indicators, months=36)
        if not kr_history.empty:
            # KOSPI는 스케일이 다르므로 별도 축 사용
            fig_kr = make_subplots(specs=[[{"secondary_y": True}]])

            colors = ["navy", "crimson", "darkgreen", "darkorange"]
            for i, col in enumerate(kr_history.columns):
                secondary = col == "KOSPI"
                fig_kr.add_trace(
                    go.Scatter(
                        x=kr_history.index, y=kr_history[col],
                        mode="lines", name=col,
                        line=dict(color=colors[i % len(colors)]),
                    ),
                    secondary_y=secondary,
                )

            fig_kr.update_layout(
                title="한국 경기선행지수 구성요소 추이 (3년)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig_kr.update_yaxes(title_text="지수 / 금리차", secondary_y=False)
            fig_kr.update_yaxes(title_text="KOSPI", secondary_y=True)
            st.plotly_chart(fig_kr, use_container_width=True)
    else:
        if not bok_key:
            st.info("한국은행 ECOS API 키가 필요합니다. (KOSPI는 API 키 없이 조회 가능)")
        else:
            st.info("한국 경기선행지수 데이터를 불러올 수 없습니다.")

    # 해석 가이드
    st.markdown("---")
    with st.expander("경기선행지수 해석 가이드"):
        st.markdown("""
**장단기 금리차 (Yield Spread)**
- 장기금리(10년) - 단기금리(3개월 또는 3년)
- **양수**: 정상적인 수익률 곡선, 경기 확장 신호
- **음수(역전)**: 경기 침체 선행 신호 (보통 12~18개월 전 예고)

**경제심리지수 (ESI)**
- 기준선: 100
- **100 초과**: 경기 낙관, 소비·투자 증가 예상
- **100 미만**: 경기 비관, 경제활동 위축 예상

**실질 M2 통화공급**
- 인플레이션 조정된 화폐 공급량
- **증가**: 유동성 확대, 자산가격 상승 지지
- **감소**: 긴축적 금융환경, 경기 둔화 가능성

**국제원자재가격지수**
- **상승**: 인플레이션 압력, 기업 비용 증가
- **하락**: 디플레이션 압력, 경기 둔화 신호

**주가지수 (S&P 500 / KOSPI)**
- 경기를 6~9개월 선행
- **상승**: 기업 실적 개선 기대, 경기 확장 예상
- **하락**: 경기 둔화 예상
        """)

# --- AI Regime 전략 탭 ---
with tab_regime:
    st.subheader("🤖 AI Regime 전략 — 구조적 레짐 전환 분석")
    st.caption("2028 AI Crisis 시나리오(CitriniResearch) 기반 KOSPI200 포지션 맵")

    regime_sub1, regime_sub2, regime_sub3, regime_sub4 = st.tabs([
        "📋 시나리오 개요", "🏭 산업 분류", "🗺️ 포지션 맵", "📡 실시간 모니터"
    ])
    _sector_df = build_sector_dataframe()

    # ── 시나리오 개요 ──
    with regime_sub1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("S&P 500 하락폭", "-38%", delta="-38%", delta_color="inverse")
        with col2:
            st.metric("실업률 시나리오", "10.2%", delta="+0.3%p 서프라이즈", delta_color="inverse")
        with col3:
            st.metric("노동소득 비중 변화", "56% → 46%", delta="-10%p (4년)", delta_color="inverse")
        with col4:
            st.metric("한국 포지셔닝", "AI 인프라 Convex", delta="반도체/전력 수혜")

        st.divider()
        st.plotly_chart(plot_scenario_timeline(), use_container_width=True)
        st.divider()

        _col_l, _col_r = st.columns([1, 1])
        with _col_l:
            st.markdown("### Intelligence Displacement Spiral")
            st.plotly_chart(plot_displacement_spiral(), use_container_width=True)
        with _col_r:
            st.markdown("### 핵심 전달 메커니즘")
            st.markdown("""
**1단계: AI 능력 도약** (2025 Q4~)
- 에이전틱 코딩 → SaaS 내재화 가능
- CIO: "직접 만들면 안 되나?"

**2단계: 고용 → 소비 연쇄** (2026~)
- 화이트칼라 해고 → 하향취업
- 상위 10% 소득자 = 소비의 50%+
- 2% 고용 감소 → 3~4% 재량소비 감소

**3단계: 금융 전이** (2027~)
- SaaS ARR 해체 → 사모 신용 디폴트
- 중개 마진 붕괴 (DoorDash, 카드, 보험)
- 프라임 모기지 위기 조짐

**4단계: 피드백 루프 가속**
- 기존 통화정책 무력화
- AI가 $180K PM 업무를 $200/월에 수행 → 금리 인하로 해결 불가
            """)

        st.divider()
        st.markdown("### 한국 시장 특수성")
        _kr1, _kr2 = st.columns(2)
        with _kr1:
            st.markdown("""
**🟢 긍정 요인 (AI 인프라 Convex)**
- 반도체: 삼성전자·SK하이닉스 (HBM/GPU)
- 전력기기: HD현대일렉트릭, LS일렉트릭
- 시나리오 내 "Taiwan and Korea outperformed massively"
- 데이터센터 건설 수요 → 전력·냉각·네트워크
            """)
        with _kr2:
            st.markdown("""
**🔴 부정 요인 (대체 위험)**
- IT 서비스/플랫폼: 카카오·네이버 중개 모델
- 금융 중개: 카드·보험 수수료 구조
- 부동산: 판교·강남 = 한국의 SF/시애틀
- 경기소비재: 백화점·면세 고소득 의존
            """)

    # ── 산업 분류 ──
    with regime_sub2:
        st.plotly_chart(plot_sensitivity_bar(_sector_df), use_container_width=True)
        st.divider()

        _pos_filter = st.multiselect(
            "포지션 필터", ["LONG", "SHORT", "NEUTRAL"],
            default=["LONG", "SHORT", "NEUTRAL"], key="ai_pos_filter",
        )
        _filtered = _sector_df[_sector_df["포지션"].isin(_pos_filter)]

        for _sec in _filtered["섹터"].unique():
            _si = KOSPI200_SECTOR_MAP[_sec]
            _pe = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "🟡"}[_si["position"]]
            with st.expander(f"{_pe} **{_sec}** — 민감도 {_si['sensitivity']} | {_si['position']} | {_si['regime_role']}"):
                st.markdown(f"**근거:** {_si['rationale']}")
                st.dataframe(
                    pd.DataFrame([{"종목코드": k, "종목명": v} for k, v in _si["tickers"].items()]),
                    use_container_width=True, hide_index=True,
                )

        st.divider()
        st.subheader("전체 종목 분류표")
        _disp = _filtered[["포지션", "레짐역할", "섹터", "종목코드", "종목명", "AI대체민감도"]].copy()
        _disp = _disp.sort_values(["포지션", "AI대체민감도"], ascending=[True, False])

        def _hl_pos(row):
            _c = {"LONG": "background-color: rgba(0,200,83,0.15)",
                  "SHORT": "background-color: rgba(255,23,68,0.15)",
                  "NEUTRAL": "background-color: rgba(255,214,0,0.15)"}
            return [_c.get(row["포지션"], "")] * len(row)

        st.dataframe(_disp.style.apply(_hl_pos, axis=1), use_container_width=True, height=600, hide_index=True)
        st.download_button("CSV 다운로드", _disp.to_csv(index=False).encode("utf-8-sig"),
                           "ai_regime_classification.csv", "text/csv", key="dl_regime")

    # ── 포지션 맵 ──
    with regime_sub3:
        st.plotly_chart(plot_position_treemap(_sector_df), use_container_width=True)
        st.divider()
        st.subheader("포지션 요약")
        _cl, _cs, _cn = st.columns(3)
        _long_s = [s for s, v in KOSPI200_SECTOR_MAP.items() if v["position"] == "LONG"]
        _short_s = [s for s, v in KOSPI200_SECTOR_MAP.items() if v["position"] == "SHORT"]
        _neut_s = [s for s, v in KOSPI200_SECTOR_MAP.items() if v["position"] == "NEUTRAL"]
        with _cl:
            st.markdown("### 🟢 LONG (매수)")
            for _s in _long_s:
                _i = KOSPI200_SECTOR_MAP[_s]
                st.markdown(f"**{_s}** (민감도 {_i['sensitivity']})")
                st.caption(", ".join(_i["tickers"].values()))
        with _cs:
            st.markdown("### 🔴 SHORT (매도/회피)")
            for _s in _short_s:
                _i = KOSPI200_SECTOR_MAP[_s]
                st.markdown(f"**{_s}** (민감도 {_i['sensitivity']})")
                st.caption(", ".join(_i["tickers"].values()))
        with _cn:
            st.markdown("### 🟡 NEUTRAL (선별)")
            for _s in _neut_s:
                _i = KOSPI200_SECTOR_MAP[_s]
                st.markdown(f"**{_s}** (민감도 {_i['sensitivity']})")
                st.caption(", ".join(_i["tickers"].values()))

        st.divider()
        st.markdown("""### 전략 가이드라인
| 구분 | 전략 | 핵심 논리 |
|------|------|-----------|
| **LONG** | AI 인프라 구축 수혜 | GPU/HBM, 전력기기, 데이터센터 → 컴퓨트 수요 구조적 증가 |
| **SHORT** | 노동 대체 민감 섹터 회피 | SaaS, 중개/플랫폼, 카드, 보험 → 인건비 기반 비즈니스 모델 해체 |
| **NEUTRAL** | 개별 종목 선별 | 자동차(자율주행 수혜 vs 소비 위축), 소재(IDC 수요 vs 경기둔화) |
| **방어** | 필수소비·헬스케어 | 경기 방어적 특성, 다만 에이전트 가격 최적화로 마진 압박 가능 |
        """)

    # ── 실시간 모니터 ──
    with regime_sub4:
        st.caption("KOSPI200 대표 종목의 실시간 가격 기반 레짐 모멘텀 분석")
        _lookback = st.selectbox("분석 기간", [20, 40, 60, 120], index=2, key="regime_lookback",
                                 format_func=lambda x: f"{x}거래일 (~{x//20}개월)")
        with st.spinner("시장 데이터 조회 중..."):
            _all_codes = _sector_df["종목코드"].unique().tolist()
            _prices = fetch_sector_prices(_all_codes, days=_lookback)

        if not _prices.empty:
            _momentum = compute_regime_momentum(_prices, _sector_df)
            if not _momentum.empty:
                _sfig = plot_regime_scatter(_momentum)
                if _sfig:
                    st.plotly_chart(_sfig, use_container_width=True)

                st.divider()
                st.subheader("섹터별 모멘텀 대시보드")
                _sort_col = st.selectbox("정렬 기준",
                    ["AI대체민감도", "1개월수익률(%)", "단기모멘텀(%)", "장기모멘텀(%)"], key="regime_sort")
                _sort_asc = st.checkbox("오름차순", value=False, key="regime_sort_asc")
                _msorted = _momentum.sort_values(_sort_col, ascending=_sort_asc)

                def _cm(val):
                    if pd.isna(val): return ""
                    return "color: #00C853" if val > 0 else ("color: #FF1744" if val < 0 else "")

                def _cp(val):
                    return {"LONG": "color: #00C853; font-weight: bold",
                            "SHORT": "color: #FF1744; font-weight: bold",
                            "NEUTRAL": "color: #FFD600; font-weight: bold"}.get(val, "")

                _styled = _msorted.style \
                    .map(_cm, subset=["단기모멘텀(%)", "장기모멘텀(%)", "1개월수익률(%)"]) \
                    .map(_cp, subset=["포지션"]) \
                    .format({"단기모멘텀(%)": "{:+.1f}", "장기모멘텀(%)": "{:+.1f}",
                             "1개월수익률(%)": "{:+.1f}"}, na_rep="-")
                st.dataframe(_styled, use_container_width=True, height=500, hide_index=True)

                st.divider()
                st.subheader("포지션별 평균 성과")
                _pa = _momentum.groupby("포지션").agg({
                    "1개월수익률(%)": "mean", "단기모멘텀(%)": "mean",
                    "장기모멘텀(%)": "mean", "AI대체민감도": "mean", "섹터": "count",
                }).rename(columns={"섹터": "섹터수"}).reset_index()
                _cfig = go.Figure()
                _cmap = {"LONG": "#00C853", "SHORT": "#FF1744", "NEUTRAL": "#FFD600"}
                for _, _r in _pa.iterrows():
                    _cfig.add_trace(go.Bar(
                        x=[_r["포지션"]], y=[_r["1개월수익률(%)"]],
                        name=_r["포지션"], marker_color=_cmap.get(_r["포지션"], "#8B949E"),
                        text=f'{_r["1개월수익률(%)"]:.1f}%', textposition="outside",
                    ))
                _cfig.update_layout(
                    title="포지션별 평균 1개월 수익률", yaxis_title="수익률 (%)", height=350,
                    plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", font=dict(color="#E6EDF3"),
                    yaxis=dict(gridcolor="#30363D", zeroline=True, zerolinecolor="#8B949E"),
                    showlegend=False,
                )
                st.plotly_chart(_cfig, use_container_width=True)

                # LONG-SHORT 스프레드
                _lr = _momentum[_momentum["포지션"] == "LONG"]["1개월수익률(%)"].mean()
                _sr = _momentum[_momentum["포지션"] == "SHORT"]["1개월수익률(%)"].mean()
                _sp = _lr - _sr if pd.notna(_lr) and pd.notna(_sr) else None
                if _sp is not None:
                    st.markdown(f"""
                    <div style="text-align:center; padding:1.5rem; background:#161B22; border-radius:8px; border:1px solid #30363D;">
                        <div style="color:#8B949E; font-size:0.9rem;">LONG-SHORT 스프레드 (1개월)</div>
                        <div style="font-size:2rem; font-weight:bold; color:{'#00C853' if _sp > 0 else '#FF1744'};">{_sp:+.1f}%p</div>
                        <div style="color:#8B949E; font-size:0.8rem;">LONG 평균: {_lr:+.1f}% | SHORT 평균: {_sr:+.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("시장 데이터를 조회하지 못했습니다. 잠시 후 다시 시도해주세요.")

        st.divider()
        with st.expander("⚠️ 면책 고지"):
            st.markdown("""
본 분석은 **CitriniResearch의 가상 시나리오**에 기반한 교육적·연구적 목적의 레짐 분석 도구입니다.
- 이 시나리오는 **예측이 아닌 시뮬레이션**입니다
- 실제 투자 결정은 개인의 판단과 책임 하에 이루어져야 합니다
- AI 대체 민감도 점수는 정성적 평가이며, 정량 모델의 결과가 아닙니다
- 과거 수익률은 미래 성과를 보장하지 않습니다
            """)
