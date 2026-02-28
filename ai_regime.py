# -*- coding: utf-8 -*-
"""
AI Regime Strategy 모듈
- 2028 AI Crisis 시나리오 기반 KOSPI200 매핑
- AI 대체 민감도별 산업 분류
- 투자 가능 포지션 맵
"""
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import FinanceDataReader as fdr
from datetime import datetime, timedelta

# ============================================================
# 1. AI 대체 민감도 산업 분류 체계 (KOSPI200 기준)
# ============================================================
KOSPI200_SECTOR_MAP = {
    # ── LONG: AI 인프라 수혜 ──
    "반도체": {
        "position": "LONG", "sensitivity": 5, "regime_role": "수혜",
        "color": "#00C853",
        "rationale": "GPU/HBM/파운드리 → AI 컴퓨트 수요의 직접 수혜. TSM 95%+ 가동률, NVDA 매출 신기록 지속 시나리오",
        "tickers": {
            "005930": "삼성전자", "000660": "SK하이닉스",
            "042700": "한미반도체", "403870": "HPSP",
        },
    },
    "반도체장비·소재": {
        "position": "LONG", "sensitivity": 5, "regime_role": "수혜",
        "color": "#00E676",
        "rationale": "AI 반도체 설비투자(CapEx) 확대 → 장비·소재 수요 동반 증가",
        "tickers": {
            "058470": "리노공업", "036930": "주성엔지니어링",
            "357780": "솔브레인", "272210": "한화에어로스페이스",
        },
    },
    "전력·에너지인프라": {
        "position": "LONG", "sensitivity": 8, "regime_role": "수혜",
        "color": "#76FF03",
        "rationale": "데이터센터 전력 수요 폭증 → 터빈·변압기·전선 2040년까지 완판 시나리오 (GE Vernova 사례)",
        "tickers": {
            "267260": "HD현대일렉트릭", "010120": "LS일렉트릭",
            "003530": "한화투자증권",
            "009540": "HD한국조선해양",
        },
    },
    "데이터센터·통신인프라": {
        "position": "LONG", "sensitivity": 15, "regime_role": "수혜",
        "color": "#69F0AE",
        "rationale": "AI 추론/학습 인프라 확대 → IDC, 네트워크장비, 광통신 수요 증가",
        "tickers": {
            "030200": "KT", "017670": "SK텔레콤",
            "032640": "LG유플러스", "053800": "안랩",
        },
    },
    "방위산업": {
        "position": "LONG", "sensitivity": 10, "regime_role": "방어",
        "color": "#B2FF59",
        "rationale": "지정학적 리스크 + AI 군사 응용 → 비경기적 수요. 경기침체기에도 정부 지출 유지",
        "tickers": {
            "012450": "한화에어로스페이스", "047810": "한국항공우주",
            "079550": "LIG넥스원", "005880": "대한해운",
        },
    },
    # ── SHORT: AI 대체 고위험 ──
    "소프트웨어·IT서비스": {
        "position": "SHORT", "sensitivity": 90, "regime_role": "피해",
        "color": "#FF1744",
        "rationale": "ServiceNow/Zendesk 사례 → SaaS ARR 해체. 에이전트 코딩으로 내부 개발 대체. SI/컨설팅 수요 붕괴",
        "tickers": {
            "035720": "카카오", "035420": "NAVER",
            "036570": "엔씨소프트", "259960": "크래프톤",
        },
    },
    "금융중개·카드": {
        "position": "SHORT", "sensitivity": 80, "regime_role": "피해",
        "color": "#FF5252",
        "rationale": "에이전트 커머스 → 카드 수수료(인터체인지) 우회. AmEx/MA 사례. 스테이블코인 결제 확산",
        "tickers": {
            "029780": "삼성카드", "071050": "한국금융지주",
            "105560": "KB금융", "055550": "신한지주",
        },
    },
    "부동산·건설": {
        "position": "SHORT", "sensitivity": 75, "regime_role": "피해",
        "color": "#FF8A80",
        "rationale": "화이트칼라 실직 → 소득 가정 붕괴 → 프라임 모기지 리스크. SF/시애틀 -11% 시나리오의 한국 강남/판교 매핑",
        "tickers": {
            "000720": "현대건설", "047040": "대우건설",
            "006360": "GS건설", "034730": "SK",
        },
    },
    "플랫폼·중개서비스": {
        "position": "SHORT", "sensitivity": 85, "regime_role": "피해",
        "color": "#FF5722",
        "rationale": "DoorDash 사례 → 습관적 중개(habitual intermediation) 파괴. AI 에이전트가 최저 수수료·최적가 자동 탐색",
        "tickers": {
            "035720": "카카오",
            "357230": "에스디바이오센서",
            "069080": "웹젠",
        },
    },
    "보험": {
        "position": "SHORT", "sensitivity": 70, "regime_role": "피해",
        "color": "#E57373",
        "rationale": "보험갱신 관성(passive renewal) 해체. 에이전트가 연 1회 자동 재비교 → 15~20% 프리미엄 소멸",
        "tickers": {
            "000810": "삼성화재", "032830": "삼성생명",
            "082640": "동양생명", "088350": "한화생명",
        },
    },
    # ── NEUTRAL/차별화 ──
    "자동차": {
        "position": "NEUTRAL", "sensitivity": 40, "regime_role": "차별화",
        "color": "#FFD600",
        "rationale": "자율주행 AI 수혜 vs 소비자 소득 감소 충격. 프리미엄 브랜드 방어, 대중차 타격 가능성",
        "tickers": {
            "005380": "현대자동차", "000270": "기아",
            "012330": "현대모비스", "018880": "한온시스템",
        },
    },
    "필수소비재": {
        "position": "NEUTRAL", "sensitivity": 25, "regime_role": "방어",
        "color": "#FFC107",
        "rationale": "경기 방어적 특성. 소비 위축에도 식품·생활용품 수요는 유지. 다만 에이전트의 가격 최적화로 마진 압박",
        "tickers": {
            "004370": "농심", "097950": "CJ제일제당",
            "051900": "LG생활건강", "090430": "아모레퍼시픽",
        },
    },
    "헬스케어·제약": {
        "position": "NEUTRAL", "sensitivity": 30, "regime_role": "방어",
        "color": "#FFAB00",
        "rationale": "AI 신약개발 수혜 vs 임상·규제 불확실성. 고령화 메가트렌드 방어적. 바이오는 R&D 효율화 수혜",
        "tickers": {
            "068270": "셀트리온", "207940": "삼성바이오로직스",
            "128940": "한미약품", "326030": "SK바이오팜",
        },
    },
    "경기소비재": {
        "position": "SHORT", "sensitivity": 65, "regime_role": "피해",
        "color": "#FF9100",
        "rationale": "상위 10% 소득자 소비 50%+ 차지 → 화이트칼라 실직 시 가장 큰 소비 타격. 2% 고용 감소 → 3~4% 재량소비 감소",
        "tickers": {
            "004170": "신세계", "023530": "롯데쇼핑",
            "069960": "현대백화점", "139480": "이마트",
        },
    },
    "철강·화학·소재": {
        "position": "NEUTRAL", "sensitivity": 35, "regime_role": "차별화",
        "color": "#FFE082",
        "rationale": "AI 인프라 건설 수요(데이터센터 철강·구리) vs 전반적 경기 둔화. 소재별 차별화",
        "tickers": {
            "005490": "POSCO홀딩스", "051910": "LG화학",
            "006400": "삼성SDI", "011170": "롯데케미칼",
        },
    },
    "운송·물류": {
        "position": "NEUTRAL", "sensitivity": 50, "regime_role": "차별화",
        "color": "#FFD54F",
        "rationale": "자율주행·드론배송 수혜 vs 기존 운송업 대체. 해운은 글로벌 교역 의존",
        "tickers": {
            "028670": "팬오션", "011200": "HMM",
            "000120": "CJ대한통운",
        },
    },
}


# ============================================================
# 2. 데이터 함수
# ============================================================
def build_sector_dataframe():
    """섹터 분류 데이터를 DataFrame으로 변환"""
    rows = []
    for sector, info in KOSPI200_SECTOR_MAP.items():
        for code, name in info["tickers"].items():
            rows.append({
                "섹터": sector,
                "종목코드": code,
                "종목명": name,
                "포지션": info["position"],
                "AI대체민감도": info["sensitivity"],
                "레짐역할": info["regime_role"],
                "색상": info["color"],
                "근거": info["rationale"],
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def fetch_sector_prices(codes: list, days: int = 60) -> pd.DataFrame:
    """섹터 대표 종목 가격 데이터 조회"""
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.8))
    frames = {}
    for code in codes:
        try:
            df = fdr.DataReader(code, start.strftime("%Y-%m-%d"))
            if df is not None and not df.empty and "Close" in df.columns:
                frames[code] = df["Close"].tail(days)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


def compute_regime_momentum(prices_df, sector_df, short_window=5, long_window=20):
    """각 섹터별 모멘텀 계산"""
    if prices_df.empty:
        return pd.DataFrame()
    results = []
    for sector, grp in sector_df.groupby("섹터"):
        codes = [c for c in grp["종목코드"].values if c in prices_df.columns]
        if not codes:
            continue
        returns = prices_df[codes].pct_change().mean(axis=1)
        if len(returns) < long_window + 1:
            continue
        mom_short = returns.tail(short_window).mean() * 252 * 100
        mom_long = returns.tail(long_window).mean() * 252 * 100
        chg_1m = returns.tail(20).sum() * 100 if len(returns) >= 20 else None
        info = KOSPI200_SECTOR_MAP[sector]
        results.append({
            "섹터": sector,
            "포지션": info["position"],
            "AI대체민감도": info["sensitivity"],
            "레짐역할": info["regime_role"],
            "단기모멘텀(%)": round(mom_short, 1),
            "장기모멘텀(%)": round(mom_long, 1),
            "1개월수익률(%)": round(chg_1m, 1) if chg_1m is not None else None,
            "종목수": len(codes),
        })
    return pd.DataFrame(results)


# ============================================================
# 3. 시각화 함수
# ============================================================
def plot_sensitivity_bar(sector_df):
    """AI 대체 민감도 수평 막대 차트"""
    agg = sector_df.groupby("섹터").agg({
        "AI대체민감도": "first", "포지션": "first",
        "레짐역할": "first", "색상": "first",
    }).reset_index().sort_values("AI대체민감도", ascending=True)

    color_map = {"LONG": "#00C853", "SHORT": "#FF1744", "NEUTRAL": "#FFD600"}
    agg["bar_color"] = agg["포지션"].map(color_map)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=agg["섹터"], x=agg["AI대체민감도"], orientation="h",
        marker_color=agg["bar_color"],
        text=agg.apply(lambda r: f'{r["AI대체민감도"]}  [{r["포지션"]}]', axis=1),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>민감도: %{x}<br>역할: %{customdata}<extra></extra>",
        customdata=agg["레짐역할"],
    ))
    fig.update_layout(
        title="KOSPI200 섹터별 AI 대체 민감도 스코어",
        xaxis_title="AI Displacement Sensitivity (0=안전 → 100=고위험)",
        yaxis_title="", height=max(400, len(agg) * 35), margin=dict(l=200),
        plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", font=dict(color="#E6EDF3"),
        xaxis=dict(range=[0, 110], gridcolor="#30363D"), yaxis=dict(gridcolor="#30363D"),
    )
    fig.add_vrect(x0=0, x1=30, fillcolor="#00C853", opacity=0.05, line_width=0)
    fig.add_vrect(x0=30, x1=60, fillcolor="#FFD600", opacity=0.05, line_width=0)
    fig.add_vrect(x0=60, x1=100, fillcolor="#FF1744", opacity=0.05, line_width=0)
    fig.add_annotation(x=15, y=-0.5, text="LONG Zone", showarrow=False,
                       font=dict(color="#00C853", size=10), yref="paper")
    fig.add_annotation(x=45, y=-0.5, text="NEUTRAL", showarrow=False,
                       font=dict(color="#FFD600", size=10), yref="paper")
    fig.add_annotation(x=80, y=-0.5, text="SHORT Zone", showarrow=False,
                       font=dict(color="#FF1744", size=10), yref="paper")
    return fig


def plot_position_treemap(sector_df):
    """포지션 맵 (Treemap)"""
    agg = sector_df.groupby(["포지션", "레짐역할", "섹터"]).agg({
        "AI대체민감도": "first", "종목명": lambda x: ", ".join(x), "색상": "first",
    }).reset_index()
    agg["size_val"] = agg.apply(
        lambda r: (100 - r["AI대체민감도"]) if r["포지션"] == "LONG" else r["AI대체민감도"], axis=1
    ).clip(lower=10)
    color_scale = {"LONG": "#00C853", "SHORT": "#FF1744", "NEUTRAL": "#FFD600"}
    fig = px.treemap(
        agg, path=["포지션", "레짐역할", "섹터"], values="size_val",
        color="포지션", color_discrete_map=color_scale,
        hover_data={"AI대체민감도": True, "종목명": True, "size_val": False},
        title="AI Regime 포지션 맵",
    )
    fig.update_layout(
        height=600, plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        font=dict(color="#E6EDF3", size=13), margin=dict(t=50, l=10, r=10, b=10),
    )
    fig.update_traces(
        textinfo="label+text",
        texttemplate="<b>%{label}</b><br>민감도: %{customdata[0]}",
        hovertemplate="<b>%{label}</b><br>민감도: %{customdata[0]}<br>종목: %{customdata[1]}<extra></extra>",
    )
    return fig


def plot_regime_scatter(momentum_df):
    """레짐 전략 산점도: X=AI대체민감도, Y=1개월수익률"""
    if momentum_df.empty:
        return None
    color_map = {"LONG": "#00C853", "SHORT": "#FF1744", "NEUTRAL": "#FFD600"}
    momentum_df = momentum_df.copy()
    momentum_df["color"] = momentum_df["포지션"].map(color_map)
    fig = go.Figure()
    for pos in ["LONG", "SHORT", "NEUTRAL"]:
        sub = momentum_df[momentum_df["포지션"] == pos]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["AI대체민감도"], y=sub["1개월수익률(%)"],
            mode="markers+text",
            marker=dict(size=sub["종목수"] * 5 + 10, color=color_map[pos], opacity=0.7,
                        line=dict(width=1, color="#E6EDF3")),
            text=sub["섹터"], textposition="top center",
            textfont=dict(size=9, color="#E6EDF3"), name=pos,
            hovertemplate="<b>%{text}</b><br>민감도: %{x}<br>수익률: %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        title="AI 대체 민감도 vs 최근 시장 성과",
        xaxis_title="AI Displacement Sensitivity", yaxis_title="1개월 수익률 (%)",
        height=500, plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        font=dict(color="#E6EDF3"),
        xaxis=dict(gridcolor="#30363D"),
        yaxis=dict(gridcolor="#30363D", zeroline=True, zerolinecolor="#8B949E"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.add_vline(x=50, line_dash="dot", line_color="#8B949E", opacity=0.5)
    fig.add_hline(y=0, line_dash="dot", line_color="#8B949E", opacity=0.5)
    fig.add_annotation(x=25, y=1.05, text="✅ LONG & 상승", showarrow=False,
                       font=dict(color="#00C853", size=10), xref="x", yref="paper")
    fig.add_annotation(x=75, y=1.05, text="⚠️ SHORT & 상승 (경고)", showarrow=False,
                       font=dict(color="#FF9100", size=10), xref="x", yref="paper")
    fig.add_annotation(x=75, y=-0.05, text="🎯 SHORT & 하락 (확인)", showarrow=False,
                       font=dict(color="#FF1744", size=10), xref="x", yref="paper")
    return fig


def plot_displacement_spiral():
    """Intelligence Displacement Spiral 흐름도"""
    fig = go.Figure()
    labels = [
        "AI 능력 향상", "기업 AI 투자 확대", "화이트칼라 해고",
        "소비 위축", "기업 매출 감소", "마진 방어 → 추가 AI 투자",
    ]
    n = len(labels)
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    r = 2
    xs = [r * math.cos(a) for a in angles]
    ys = [r * math.sin(a) for a in angles]
    for i in range(n):
        fig.add_annotation(
            x=xs[(i + 1) % n], y=ys[(i + 1) % n], ax=xs[i], ay=ys[i],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor="#FF5252",
        )
    colors = ["#FF5252", "#FF7043", "#EF5350", "#E53935", "#D32F2F", "#C62828"]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=50, color=colors, line=dict(width=2, color="#E6EDF3")),
        text=[f"<b>{i+1}. {l}</b>" for i, l in enumerate(labels)],
        textposition=["top center" if ys[i] < 0 else "bottom center" for i in range(n)],
        textfont=dict(size=11, color="#E6EDF3"), hoverinfo="text",
    ))
    fig.add_annotation(x=0, y=0, text="<b>Intelligence<br>Displacement<br>Spiral</b>",
                       showarrow=False, font=dict(size=14, color="#FF1744"))
    fig.update_layout(
        height=450, plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        font=dict(color="#E6EDF3"),
        xaxis=dict(visible=False, range=[-3.5, 3.5]),
        yaxis=dict(visible=False, range=[-3.5, 3.5], scaleanchor="x"),
        showlegend=False, margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def plot_scenario_timeline():
    """시나리오 타임라인"""
    events = [
        ("2025 Q4", "에이전틱 코딩 도약", "AI 코딩 도구 단계 도약"),
        ("2026 Q1", "화이트칼라 해고 시작", "마진 확대, 실적 호조"),
        ("2026 Q3", "ServiceNow 충격", "SaaS ARR 해체 시작"),
        ("2026 Q4", "S&P 8000 → 피크", "JOLTS 5.5M 하회"),
        ("2027 Q1", "에이전틱 커머스 확산", "중개 마진 붕괴"),
        ("2027 Q2", "경기침체 진입", "2분기 연속 역성장"),
        ("2027 Q3", "초기실업 487K", "사모 신용 디폴트"),
        ("2027 Q4", "Zendesk 디폴트", "시스템 리스크 전이"),
        ("2028 Q2", "모기지 위기 조짐", "SF -11%, 실업 10.2%"),
    ]
    fig = go.Figure()
    for i, (period, event, detail) in enumerate(events):
        color = "#00C853" if i < 2 else ("#FFD600" if i < 4 else ("#FF5252" if i < 7 else "#B71C1C"))
        fig.add_trace(go.Scatter(
            x=[i], y=[0], mode="markers+text",
            marker=dict(size=20, color=color, symbol="diamond"),
            text=f"<b>{period}</b><br>{event}", textposition="top center",
            textfont=dict(size=9, color="#E6EDF3"),
            hovertemplate=f"<b>{period}: {event}</b><br>{detail}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        title="2028 AI Crisis 시나리오 타임라인", height=250,
        plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", font=dict(color="#E6EDF3"),
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-0.5, 1.5]),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.add_shape(type="line", x0=0, x1=len(events) - 1, y0=0, y1=0,
                  line=dict(color="#8B949E", width=2, dash="dot"))
    return fig
