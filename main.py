# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(page_title="Jason Monitor v2", layout="wide")

st.title("🛡️ Jason 台股最強戰術儀表板")
st.success("✅ App 正常運行中")

with st.sidebar:
    st.header("控制面板")
    st.write("側邊欄正常顯示")
    load_btn = st.button("🔄 載入資料", type="primary", use_container_width=True)

if load_btn:
    with st.spinner("測試資料載入中..."):
        import yfinance as yf
        try:
            df = yf.download("2330.TW", period="5d", interval="1d",
                             auto_adjust=True, progress=False)
            if df.empty:
                st.error("yfinance 無法取得資料（可能被防火牆擋住）")
            else:
                st.write(df.tail())
                st.success(f"台積電現價：{float(df['Close'].iloc[-1]):.1f}")
        except Exception as e:
            st.error(f"錯誤：{e}")
else:
    st.info("請點擊左側「🔄 載入資料」按鈕測試")
