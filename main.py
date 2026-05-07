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

# 台股中文名稱對照表
TW_NAME_MAP = {
    # 大型權值股
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科",
    "2308.TW": "台達電", "2382.TW": "廣達", "2412.TW": "中華電",
    "2303.TW": "聯電",   "3711.TW": "日月光投控", "2881.TW": "富邦金",
    "2882.TW": "國泰金", "2891.TW": "中信金", "2886.TW": "兆豐金",
    "2884.TW": "玉山金", "2885.TW": "元大金", "2892.TW": "第一金",
    "2880.TW": "華南金", "2883.TW": "開發金", "5880.TW": "合庫金",
    "2887.TW": "台新金", "2890.TW": "永豐金",
    # 科技股
    "2379.TW": "瑞昱",   "3034.TW": "聯詠",   "2408.TW": "南亞科",
    "3008.TW": "大立光", "2357.TW": "華碩",   "2353.TW": "宏碁",
    "2376.TW": "技嘉",   "2352.TW": "佳世達", "3045.TW": "台灣大",
    "4904.TW": "遠傳",   "2395.TW": "研華",   "2327.TW": "國巨",
    "2337.TW": "旺宏",   "3023.TW": "信邦",   "6669.TW": "緯穎",
    "2301.TW": "光寶科", "2311.TW": "日月光",
    # 上櫃科技
    "8028.TW":  "昇陽半導體", "8086.TWO": "宏捷科", "3680.TWO": "家登",
    "6213.TW":  "聯茂",       "8069.TWO": "元太",
    "3529.TWO": "力旺",       "6770.TWO": "力積電",
    # ETF（TW）
    "0050.TW":  "元大台灣50",     "0056.TW":  "元大高股息",
    "00878.TW": "國泰永續高股息", "00881.TW": "國泰台灣5G+",
    "00692.TW": "富邦公司治理",   "00850.TW": "元大臺灣ESG永續",
    "006208.TW":"富邦台50",       "00646.TW": "元大S&P500",
    "00893.TW": "國泰智能電動車", "00896.TW": "中信綠能及電動車",
    "00900.TW": "富邦特選高股息", "00907.TW": "永豐優息存股",
    "00915.TW": "凱基優選高股息30","00919.TW": "群益台灣精選高息",
    "00929.TW": "復華台灣科技優息","00934.TW": "中信成長高股息",
    "00939.TW": "統一台灣高息動能","00940.TW": "元大台灣價值高息",
    "00941.TW": "台新臺灣MSCI永續", "00944.TW":"群益科技高息成長",
    "00945.TW": "玉山AI智能理財",  "00946.TW": "台新北美科技",
    "00947.TW": "台新證券投信",
    # ETF（TWO）
    "00935.TW": "野村台灣新科技50","009816.TW":"凱基優選",
    "00985A.TW":"野村台灣ESG",     "00635U.TW":"元大S&P黃金",
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

def generate_analysis(df, sig):
    curr = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else curr
    prev5 = df.iloc[-5] if len(df) >= 5 else df.iloc[0]

    price   = scalar(curr['Close'])
    ma20    = scalar(curr['MA20'])
    ma5     = scalar(curr['MA5'])
    rsi     = scalar(curr['RSI'])
    hist    = scalar(curr['Hist'])
    macd    = scalar(curr['MACD'])
    sig_ln  = scalar(curr['Signal'])
    bb_up   = scalar(curr['BB_Up'])
    bb_low  = scalar(curr['BB_Low'])
    ph      = scalar(prev['Hist'])
    chg5    = (price - scalar(prev5['Close'])) / scalar(prev5['Close']) * 100

    # 目前的現況
    s = []
    s.append(f"現價 **{price:.1f}**，" + ("站於月線之上" if price > ma20 else f"跌破月線（{ma20:.1f}）"))
    if rsi >= 75:   s.append(f"RSI {rsi:.0f} 進入超買區")
    elif rsi >= 60: s.append(f"RSI {rsi:.0f} 偏強")
    elif rsi <= 30: s.append(f"RSI {rsi:.0f} 超賣反彈機會")
    elif rsi <= 45: s.append(f"RSI {rsi:.0f} 偏弱")
    else:           s.append(f"RSI {rsi:.0f} 中性")
    if hist > 0 and hist > ph:    s.append("MACD 柱擴張、動能向上")
    elif hist > 0 and hist <= ph: s.append("MACD 正值但動能略縮")
    elif hist < 0 and hist < ph:  s.append("MACD 負向擴張，空頭動能強")
    else:                         s.append("MACD 負值但收縮中")
    if price > bb_up:   s.append("突破布林上軌")
    elif price < bb_low: s.append("跌破布林下軌（超賣）")
    s.append(f"近 5 日 {chg5:+.1f}%")
    situation = "；".join(s) + "。"

    # 未來的趨勢
    score = sig['score']
    t = []
    if score >= 3:   t.append("技術面強多，趨勢向上")
    elif score >= 1: t.append("技術面偏多，但尚未全面確立")
    elif score == 0: t.append("技術面中性，方向未明")
    elif score >= -1: t.append("技術面偏弱，留意下行風險")
    else:            t.append("技術面疲弱，空頭訊號明顯")
    t.append("短均仍位於月線上方" if ma5 > ma20 else "短均已跌破月線")
    t.append(f"近期支撐 **{sig['support']:.1f}**，壓力 **{sig['resistance']:.1f}**")
    if macd > sig_ln and hist > 0:   t.append("MACD 黃金交叉維持")
    elif macd < sig_ln and hist < 0: t.append("MACD 死亡交叉持續")
    trend = "；".join(t) + "。"

    # 操作的建議
    sup, res = sig['support'], sig['resistance']
    if score >= 3:
        rec = f"📈 **強多格局**：可考慮逢回（支撐 {sup:.1f} 附近）分批布局，目標壓力 {res:.1f}，停損設月線下方。"
    elif score >= 1:
        rec = f"📊 **偏多觀察**：技術偏多但力道有限，建議輕倉試水，等待 RSI 回落至 50 附近或 MACD 確認再加碼。支撐 {sup:.1f}。"
    elif score == 0:
        rec = f"⏸️ **觀望為主**：訊號中性，建議持幣等待方向確認，注意 {sup:.1f} 支撐是否守住。"
    elif score >= -1:
        rec = f"⚠️ **偏空留意**：技術偏弱，持股留意 {sup:.1f} 支撐，跌破可考慮減碼。"
    else:
        rec = f"🔴 **弱空格局**：技術疲弱，建議觀望或減碼，反彈至 {res:.1f} 附近可考慮出清。"

    return situation, trend, rec

def confidence_score(df, sig):
    curr = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else curr
    rsi  = scalar(curr['RSI'])
    hist = scalar(curr['Hist'])
    ph   = scalar(prev['Hist'])
    ma5  = scalar(curr['MA5'])
    ma10 = scalar(curr['MA10'])
    ma20 = scalar(curr['MA20'])

    pts = 5.0
    pts += sig['score'] * 0.7              # 訊號評分：-1.4 ~ +2.8
    if 45 <= rsi <= 65:   pts += 1.0       # RSI 健康區間，訊號可靠
    elif rsi > 75 or rsi < 30: pts -= 0.5  # 極端區間，均值回歸風險
    if hist > 0 and hist > ph:   pts += 0.8  # MACD 正向擴張
    elif hist < 0 and hist < ph: pts -= 0.8  # MACD 負向擴張
    if ma5 > ma10 > ma20:   pts += 0.7    # 完整多頭排列
    elif ma5 < ma10 < ma20: pts -= 0.7    # 完整空頭排列

    return max(1, min(10, round(pts)))

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
    new_ticker = st.text_input("代碼", placeholder="e.g. 00878.TW", label_visibility="collapsed")
    new_name   = st.text_input("名稱（選填，留空自動抓）", placeholder="e.g. 國泰永續高股息", label_visibility="collapsed")
    add_btn = st.button("新增", use_container_width=True)

    if add_btn and new_ticker:
        t = new_ticker.strip().upper()
        if t not in WATCHLIST and t not in st.session_state.extra_tickers:
            with st.spinner(f"驗證 {t}..."):
                test = yf.download(t, period="3d", progress=False)
                if isinstance(test.columns, pd.MultiIndex):
                    test = test.droplevel(1, axis=1)
            if not test.empty:
                if new_name.strip():
                    name = new_name.strip()
                elif t in TW_NAME_MAP:
                    name = TW_NAME_MAP[t]
                else:
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

    st.divider()

    # 合併完整清單
    full_watchlist = {**WATCHLIST, **st.session_state.extra_tickers}

    # 從 URL 讀取已儲存的監控清單
    saved_list = get_saved_selected(full_watchlist)

    # 編輯模式 toggle
    if "editing_watchlist" not in st.session_state:
        st.session_state.editing_watchlist = False

    # 股票選擇標題列
    h1, h2 = st.columns([3, 1])
    h1.header("📋 股票選擇")
    if h2.button("✏️ 編輯" if not st.session_state.editing_watchlist else "✕ 收起",
                 use_container_width=True):
        st.session_state.editing_watchlist = not st.session_state.editing_watchlist
        st.rerun()

    if st.session_state.editing_watchlist:
        # 編輯模式：顯示 multiselect
        selected_tickers = st.multiselect(
            "選擇要監控的股票",
            options=list(full_watchlist.keys()),
            default=saved_list,
            format_func=lambda x: f"{full_watchlist[x]} ({x})"
        )
        if not selected_tickers:
            selected_tickers = list(full_watchlist.keys())

        if st.button("💾 儲存監控清單", type="primary", use_container_width=True):
            st.query_params["selected"] = ",".join(selected_tickers)
            st.session_state.editing_watchlist = False
            st.rerun()
    else:
        # 收起模式：顯示精簡清單
        selected_tickers = saved_list
        for t in selected_tickers:
            st.caption(f"• {full_watchlist.get(t, t)} ({t})")

    # 自訂股票管理（可移除）
    if st.session_state.extra_tickers:
        with st.expander("🗑️ 移除自訂股票"):
            for t, n in list(st.session_state.extra_tickers.items()):
                c1, c2 = st.columns([3, 1])
                c1.caption(f"{n} ({t})")
                if c2.button("✕", key=f"del_{t}"):
                    del st.session_state.extra_tickers[t]
                    save_extra_to_url()
                    fetch_data.clear()
                    st.rerun()

    st.divider()

    # 持股水位 — 從 URL 讀取已儲存的水位
    # 格式：?positions=8069.TWO:262000:45.2,2330.TW:1000:500
    saved_pos = {}
    raw_pos = st.query_params.get("positions", "")
    if raw_pos:
        for item in raw_pos.split(","):
            parts = item.split(":")
            if len(parts) == 3:
                saved_pos[parts[0]] = {"shares": int(parts[1]), "cost": float(parts[2])}

    st.header("💼 持股水位設定")
    positions = {}
    for ticker in selected_tickers:
        name = full_watchlist[ticker]
        default_sh = saved_pos.get(ticker, DEFAULT_POSITIONS.get(ticker, {})).get("shares", 0)
        default_ct = saved_pos.get(ticker, DEFAULT_POSITIONS.get(ticker, {})).get("cost", 0.0)
        with st.expander(f"{name} ({ticker})"):
            sh = st.number_input("股數", value=int(default_sh), key=f"sh_{ticker}", step=1000)
            ct = st.number_input("成本", value=float(default_ct), key=f"ct_{ticker}")
        positions[ticker] = {"shares": sh, "cost": ct}

    if st.button("💾 儲存持股水位", use_container_width=True):
        val = ",".join(f"{t}:{v['shares']}:{v['cost']}" for t, v in positions.items())
        st.query_params["positions"] = val
        st.success("✅ 持股水位已儲存！")

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
    progress.progress((i + 1) / len(active_watchlist), text=f"載入 {name}...")
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
        full_detail = technical_analysis(raw)
        sig_detail  = get_signal(full_detail)
        detail = full_detail.tail(120)
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

        situation, trend, recommendation = generate_analysis(full_detail, sig_detail)
        conf = confidence_score(full_detail, sig_detail)

        st.subheader("📋 技術分析摘要")

        # 信心指數
        if conf >= 8:
            conf_color, conf_label = "#22c55e", "信心強"
        elif conf >= 6:
            conf_color, conf_label = "#84cc16", "信心偏高"
        elif conf >= 4:
            conf_color, conf_label = "#f97316", "信心一般"
        else:
            conf_color, conf_label = "#ef4444", "信心偏低"

        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown(f"""
            <div style='text-align:center; padding:20px; border-radius:14px;
                        background:{conf_color}18; border:2px solid {conf_color}; margin-bottom:8px'>
                <div style='font-size:14px; color:#888; margin-bottom:4px'>信心指數</div>
                <div style='font-size:56px; font-weight:bold; color:{conf_color}; line-height:1'>
                    {conf}<span style='font-size:24px; color:#aaa'> / 10</span>
                </div>
                <div style='font-size:16px; color:{conf_color}; margin-top:6px'>{conf_label}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(conf / 10)

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🔍 目前的現況**")
            st.info(situation)
        with c2:
            st.markdown("**📈 未來的趨勢**")
            if sig_detail['score'] >= 1:
                st.success(trend)
            elif sig_detail['score'] <= -1:
                st.error(trend)
            else:
                st.warning(trend)
        with c3:
            st.markdown("**💡 操作的建議**")
            st.info(recommendation)
