# -*- coding: utf-8 -*-
"""
Taiwan Stock Tactical Monitor v2.0 - Optimized for Jason
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Jason's Tactical Monitor v2.0", layout="wide")

# 每 5 分鐘自動刷新一次（300,000 ms）
st_autorefresh(interval=300_000, key="autorefresh")

# =========================
# 1. 核心參數
# =========================
WATCHLIST = {
    "8028.TW":  "昇陽半導體",
    "8086.TWO": "宏捷科",
    "3680.TWO": "家登",
    "6213.TW":  "聯茂",
    "2330.TW":  "台積電",
    "8069.TWO": "元太",
}

DEFAULT_POSITIONS = {
    "8069.TWO": {"shares": 262000, "cost": 0.0},
}

# =========================
# 2. 分析函式
# =========================
@st.cache_data(ttl=300)
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        if df.empty or len(df) < 30:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

def technical_analysis(df):
    df = df.copy()
    close = df['Close'].squeeze()
    for ma in [5, 10, 20, 60]:
        df[f'MA{ma}'] = close.rolling(window=ma).mean()
    df['BB_Mid'] = close.rolling(20).mean()
    df['BB_Std'] = close.rolling(20).std()
    df['BB_Up']  = df['BB_Mid'] + df['BB_Std'] * 2
    df['BB_Low'] = df['BB_Mid'] - df['BB_Std'] * 2
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI']    = 100 - 100 / (1 + gain / loss)
    e1 = close.ewm(span=12, adjust=False).mean()
    e2 = close.ewm(span=26, adjust=False).mean()
    df['MACD']   = e1 - e2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist']   = df['MACD'] - df['Signal']
    return df

def scalar(val):
    if hasattr(val, 'item'):
        return float(val.item())
    return float(val)

def get_signal(df):
    if len(df) < 2:
        return {"score": 0, "status": "無資料", "reason": "",
                "support": 0, "resistance": 0}
    curr, prev = df.iloc[-1], df.iloc[-2]
    score, msg = 0, []
    if scalar(curr['Close']) > scalar(curr['MA20']): score += 2; msg.append("站上月線")
    if scalar(curr['MA5'])   > scalar(curr['MA20']): score += 1; msg.append("短均多頭排列")
    if scalar(curr['Hist'])  > scalar(prev['Hist']):  score += 1; msg.append("MACD動能增強")
    if scalar(curr['RSI'])   > 75:                    score -= 1; msg.append("RSI過熱")
    elif scalar(curr['RSI']) < 30:                    score += 2; msg.append("超賣反彈")
    support    = scalar(df['Low'].tail(20).min())
    resistance = scalar(df['High'].tail(20).max())
    status = "觀望"
    if score >= 3:    status = "強多"
    elif score >= 1:  status = "偏多"
    elif score <= -2: status = "弱空"
    return {"score": score, "status": status, "reason": " | ".join(msg),
            "support": round(support, 2), "resistance": round(resistance, 2)}

# =========================
# 3. UI
# =========================
st.title("🛡️ Jason 台股最強戰術儀表板")
st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with st.sidebar:
    # 從 URL query params 讀取已儲存的額外股票
    # 格式：?extra=2454.TW|聯發科,0050.TW|元大台灣50
    if "extra_tickers" not in st.session_state:
        raw_extra = st.query_params.get("extra", "")
        extra = {}
        if raw_extra:
            for item in raw_extra.split(","):
                parts = item.split("|")
                if len(parts) == 2:
                    extra[parts[0]] = parts[1]
                elif len(parts) == 1 and parts[0]:
                    extra[parts[0]] = parts[0]
        st.session_state.extra_tickers = extra

    def save_extra_to_url():
        val = ",".join(f"{t}|{n}" for t, n in st.session_state.extra_tickers.items())
        st.query_params["extra"] = val if val else ""

    def save_selected_to_url(tickers):
        st.query_params["selected"] = ",".join(tickers)

    def get_saved_selected(full_wl):
        saved = st.query_params.get("selected", "")
        lst = saved.split(",") if saved else list(full_wl.keys())
        return [t for t in lst if t in full_wl] or list(full_wl.keys())

    # 動態新增股票
    st.header("➕ 新增股票")
    col1, col2 = st.columns([2, 1])
    with col1:
        new_ticker = st.text_input("代碼", placeholder="e.g. 2454.TW", label_visibility="collapsed")
    with col2:
        add_btn = st.button("新增", use_container_width=True)

    if add_btn and new_ticker:
        t = new_ticker.strip().upper()
        if t not in WATCHLIST and t not in st.session_state.extra_tickers:
            with st.spinner(f"驗證 {t}..."):
                test = yf.download(t, period="3d", progress=False)
                if isinstance(test.columns, pd.MultiIndex):
                    test = test.droplevel(1, axis=1)
            if not test.empty:
                try:
                    info = yf.Ticker(t).info
                    name = info.get("shortName") or info.get("longName") or t
                except Exception:
                    name = t
                st.session_state.extra_tickers[t] = name
                save_extra_to_url()
                # 自動加入已選清單並儲存
                new_full = {**WATCHLIST, **st.session_state.extra_tickers}
                current_selected = get_saved_selected(new_full)
                if t not in current_selected:
                    current_selected.append(t)
                save_selected_to_url(current_selected)
                fetch_data.clear()
                st.rerun()
            else:
                st.error(f"找不到 {t}，請確認代碼格式")
        else:
            st.warning("此股票已在清單中")

    # 顯示已新增的股票，可刪除
    if st.session_state.extra_tickers:
        for t, n in list(st.session_state.extra_tickers.items()):
            c1, c2 = st.columns([3, 1])
            c1.caption(f"{n} ({t})")
            if c2.button("✕", key=f"del_{t}"):
                del st.session_state.extra_tickers[t]
                save_extra_to_url()
                # 從已選清單移除並儲存
                new_full = {**WATCHLIST, **st.session_state.extra_tickers}
                current_selected = get_saved_selected(new_full)
                save_selected_to_url(current_selected)
                fetch_data.clear()
                st.rerun()

    st.divider()

    # 合併完整清單
    full_watchlist = {**WATCHLIST, **st.session_state.extra_tickers}

    # 從 URL 讀取已儲存的監控清單
    saved_list = get_saved_selected(full_watchlist)

    # 股票選擇
    st.header("📋 股票選擇")
    selected_tickers = st.multiselect(
        "選擇要監控的股票",
        options=list(full_watchlist.keys()),
        default=saved_list,
        format_func=lambda x: f"{full_watchlist[x]} ({x})"
    )
    if not selected_tickers:
        selected_tickers = list(full_watchlist.keys())

    save_list_btn = st.button("💾 儲存監控清單", use_container_width=True)
    if save_list_btn:
        st.query_params["selected"] = ",".join(selected_tickers)
        st.success("✅ 監控清單已儲存！")

    st.divider()

    # 持股水位
    st.header("💼 持股水位設定")
    positions = {}
    for ticker in selected_tickers:
        name = full_watchlist[ticker]
        with st.expander(f"{name} ({ticker})"):
            sh = st.number_input("股數", value=int(DEFAULT_POSITIONS.get(ticker, {}).get("shares", 0)), key=f"sh_{ticker}", step=1000)
            ct = st.number_input("成本", value=float(DEFAULT_POSITIONS.get(ticker, {}).get("cost", 0.0)), key=f"ct_{ticker}")
        positions[ticker] = {"shares": sh, "cost": ct}

    st.divider()
    st.caption("⏱️ 每 5 分鐘自動更新")
    manual_refresh = st.button("🔄 立即更新", type="primary", use_container_width=True)

# 手動更新時清除快取
if manual_refresh:
    fetch_data.clear()

# 自動載入（含進度條）
active_watchlist = {t: full_watchlist[t] for t in selected_tickers}
summary_data, failed = [], []
progress = st.progress(0, text="載入資料中...")
for i, (ticker, name) in enumerate(active_watchlist.items()):
    progress.progress((i + 1) / len(WATCHLIST), text=f"載入 {name}...")
    raw = fetch_data(ticker)
    if raw.empty:
        failed.append(f"{name}({ticker})")
        continue
    df  = technical_analysis(raw)
    sig = get_signal(df)
    curr_p = scalar(df.iloc[-1]['Close'])
    pnl = (curr_p - positions[ticker]['cost']) * positions[ticker]['shares'] \
          if positions[ticker]['cost'] > 0 else 0
    summary_data.append({
        "代碼": ticker, "名稱": name,
        "現價": round(curr_p, 2), "評分": sig['score'],
        "趨勢": sig['status'], "支撐": sig['support'],
        "壓力": sig['resistance'], "預估損益": int(pnl),
        "戰術理由": sig['reason']
    })
progress.empty()
if failed:
    st.warning(f"資料載入失敗：{', '.join(failed)}")

if not summary_data:
    st.error("所有股票資料載入失敗，請稍後再試。")
    st.stop()

df_sum = pd.DataFrame(summary_data)
try:
    styled = df_sum.style.background_gradient(subset=['評分'], cmap='RdYlGn')
    st.dataframe(styled, use_container_width=True)
except Exception:
    st.dataframe(df_sum, use_container_width=True)

# 儲存按鈕
now_str = datetime.now().strftime("%Y-%m-%d_%H%M")
csv = df_sum.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    label="💾 儲存監控表格（CSV）",
    data=csv,
    file_name=f"jason_monitor_{now_str}.csv",
    mime="text/csv",
)

valid_tickers = [r["代碼"] for r in summary_data]
selected = st.selectbox("🔍 選擇詳細分析對象", options=valid_tickers,
                        format_func=lambda x: f"{x} {active_watchlist[x]}")

if selected:
    raw = fetch_data(selected)
    if not raw.empty:
        detail = technical_analysis(raw).tail(120)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=detail.index, open=detail['Open'],
            high=detail['High'], low=detail['Low'], close=detail['Close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=detail.index, y=detail['MA20'],
            line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=detail.index, y=detail['BB_Up'],
            line=dict(dash='dash', color='gray'), name="布林上軌"), row=1, col=1)
        fig.add_trace(go.Scatter(x=detail.index, y=detail['BB_Low'],
            line=dict(dash='dash', color='gray'), name="布林下軌"), row=1, col=1)
        fig.add_trace(go.Bar(x=detail.index, y=detail['Hist'], name="MACD Hist"), row=2, col=1)
        fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
