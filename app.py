# -*- coding: utf-8 -*-
"""
Taiwan Stock Tactical Monitor v2.0 - Optimized for Jason
Emily (Gemini 3) 強化版
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="Jason's Tactical Monitor v2.0", layout="wide")

# =========================
# 1. 核心參數與持股配置
# =========================
WATCHLIST = {
    "8028.TW":  "昇陽半導體",
    "8086.TW":  "宏捷科",
    "3680.TW":  "家登",
    "6213.TW":  "聯茂",
    "2330.TW":  "台積電",
    "8069.TWO": "元太",
}

DEFAULT_POSITIONS = {
    "8069.TWO": {"shares": 262000, "cost": 0.0},
}

# =========================
# 2. 數據分析引擎
# =========================
@st.cache_data(ttl=300)
def fetch_data(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 30:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

def technical_analysis(df):
    df = df.copy()
    for ma in [5, 10, 20, 60]:
        df[f'MA{ma}'] = df['Close'].rolling(window=ma).mean()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Up']  = df['BB_Mid'] + df['BB_Std'] * 2
    df['BB_Low'] = df['BB_Mid'] - df['BB_Std'] * 2
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI']    = 100 - 100 / (1 + gain / loss)
    exp1         = df['Close'].ewm(span=12, adjust=False).mean()
    exp2         = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']   = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist']   = df['MACD'] - df['Signal']
    return df

def get_tactical_signal(df):
    if len(df) < 2:
        return {"score": 0, "status": "無資料", "reason": "",
                "support": 0, "resistance": 0, "stop_loss": 0}
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    score, msg = 0, []
    if curr['Close'] > curr['MA20']:   score += 2; msg.append("站上月線")
    if curr['MA5']   > curr['MA20']:   score += 1; msg.append("短均多頭排列")
    if curr['Hist']  > prev['Hist']:   score += 1; msg.append("MACD動能增強")
    if curr['RSI']   > 75:             score -= 1; msg.append("RSI過熱回檔風險")
    elif curr['RSI'] < 30:             score += 2; msg.append("超賣區具反彈契機")
    support    = float(df['Low'].tail(20).min())
    resistance = float(df['High'].tail(20).max())
    status = "觀望"
    if score >= 3:   status = "強多"
    elif score >= 1: status = "偏多"
    elif score <= -2: status = "弱空"
    return {"score": score, "status": status, "reason": " | ".join(msg),
            "support": round(support, 2), "resistance": round(resistance, 2),
            "stop_loss": round(support * 0.97, 2)}

# =========================
# 3. UI
# =========================
st.title("🛡️ Jason 台股最強戰術儀表板")
st.caption(f"最後更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with st.sidebar:
    st.header("💼 持股水位設定")
    positions = {}
    for ticker, name in WATCHLIST.items():
        st.subheader(f"{name} ({ticker})")
        sh = st.number_input("股數", value=int(DEFAULT_POSITIONS.get(ticker, {}).get("shares", 0)), key=f"sh_{ticker}")
        ct = st.number_input("成本", value=float(DEFAULT_POSITIONS.get(ticker, {}).get("cost", 0.0)), key=f"ct_{ticker}")
        positions[ticker] = {"shares": sh, "cost": ct}

# 核心總覽
summary_data = []
failed = []

with st.spinner("載入股票資料中..."):
    for ticker, name in WATCHLIST.items():
        raw_df = fetch_data(ticker)
        if raw_df.empty:
            failed.append(f"{name}({ticker})")
            continue
        df  = technical_analysis(raw_df)
        sig = get_tactical_signal(df)
        curr_p = float(df.iloc[-1]['Close'])
        pnl = (curr_p - positions[ticker]['cost']) * positions[ticker]['shares'] \
              if positions[ticker]['cost'] > 0 else 0
        summary_data.append({
            "代碼": ticker, "名稱": name,
            "現價": round(curr_p, 2),
            "評分": sig['score'], "趨勢": sig['status'],
            "支撐": sig['support'], "壓力": sig['resistance'],
            "預估損益": int(pnl), "戰術理由": sig['reason']
        })

if failed:
    st.warning(f"以下股票資料載入失敗：{', '.join(failed)}")

if summary_data:
    df_sum = pd.DataFrame(summary_data)
    try:
        styled = df_sum.style.background_gradient(subset=['評分'], cmap='RdYlGn')
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df_sum, use_container_width=True)
else:
    st.error("所有股票資料載入失敗，請稍後重新整理。")
    st.stop()

# 詳細圖表
valid_tickers = [row["代碼"] for row in summary_data]
selected_stock = st.selectbox(
    "🔍 選擇詳細分析對象",
    options=valid_tickers,
    format_func=lambda x: f"{x} {WATCHLIST[x]}"
)

if selected_stock:
    raw = fetch_data(selected_stock)
    if raw.empty:
        st.warning("該股票資料暫時無法取得。")
    else:
        detail_df = technical_analysis(raw).tail(120)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(
            x=detail_df.index, open=detail_df['Open'], high=detail_df['High'],
            low=detail_df['Low'], close=detail_df['Close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=detail_df.index, y=detail_df['MA20'],
            line=dict(color='orange', width=1.5), name="20MA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=detail_df.index, y=detail_df['BB_Up'],
            line=dict(dash='dash', color='gray'), name="布林上軌"), row=1, col=1)
        fig.add_trace(go.Scatter(x=detail_df.index, y=detail_df['BB_Low'],
            line=dict(dash='dash', color='gray'), name="布林下軌"), row=1, col=1)
        fig.add_trace(go.Bar(x=detail_df.index, y=detail_df['Hist'],
            name="MACD柱狀體"), row=2, col=1)
        fig.update_layout(height=600, xaxis_rangeslider_visible=False,
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.success("Emily 的戰術建議：對於 8069 元太，請密切關注其在電子紙零售應用的動能，若回測月線不破即是 Jason 佈局良機。")
