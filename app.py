import streamlit as st
from dotenv import load_dotenv
import os
import json
from io import BytesIO
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas.tseries.offsets import BDay
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ---- Load .env ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in your .env file.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="FinDocGPT", layout="wide")
st.title("📊 FinDocGPT – Forecast + Doc Q&A")

tab1, tab2 = st.tabs(["🔮 Forecast", "📑 Doc Q&A"])

# -----------------------------------------------
# 🔮 Forecast Tab
# -----------------------------------------------
with tab1:
    st.header("Stock Price Forecast (Holt-Winters)")

    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    start_date = st.date_input("Start Date", value=date(2020, 1, 1))
    n_days = st.slider("Forecast Horizon (Days)", 30, 252, 90)
    use_seasonality = st.checkbox("Use Seasonality (252 days)", value=True)
    threshold = st.slider("Buy/Sell Threshold (%)", 0.0, 20.0, 2.0) / 100

    if st.button("Run Forecast"):
        df = yf.download(ticker, start=start_date.isoformat(), auto_adjust=True)
        if df.empty:
            st.error("No data found. Check the ticker symbol.")
        else:
            series = df["Close"]
            series.index = pd.to_datetime(series.index)
            if use_seasonality and len(series) < 260:
                use_seasonality = False

            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add" if use_seasonality else None,
                seasonal_periods=252 if use_seasonality else None,
                damped_trend=True,
                initialization_method="estimated"
            )
            fit = model.fit()
            future_index = pd.bdate_range(start=series.index[-1] + BDay(1), periods=n_days)
            forecast = pd.Series(fit.forecast(n_days), index=future_index)
            resid_std = np.std(series - fit.fittedvalues)
            upper = forecast + 1.96 * resid_std
            lower = forecast - 1.96 * resid_std

            st.line_chart(pd.concat([series.rename("Historical"), forecast.rename("Forecast")], axis=1))

            # Recommendation
            last_price = series.iloc[-1]
            pred_price = forecast.iloc[-1]
            exp_return = (pred_price - last_price) / last_price
            if exp_return > threshold:
                signal = "🟢 BUY"
            elif exp_return < -threshold:
                signal = "🔴 SELL"
            else:
                signal = "🟡 HOLD"

            st.metric("Last Price", f"${last_price:.2f}")
            st.metric(f"Forecast ({forecast.index[-1].date()})", f"${pred_price:.2f}")
            st.metric("Recommendation", f"{signal} ({exp_return:.2%})")

# -----------------------------------------------
# 📑 Document Q&A Tab
# -----------------------------------------------
with tab2:
    st.header("Answer Questions from Financial Documents")

    try:
        with open("qa_data.json", "r", encoding="utf-8") as f:
            qa_data = json.load(f)
    except Exception:
        st.warning("Could not load qa_data.json")
        qa_data = []

    options = [q["question"] for q in qa_data]
    selected_question = st.selectbox("Choose a question from qa_data.json:", ["—"] + options)
    custom_question = st.text_area("Or write your own question:")

    final_question = custom_question if selected_question == "—" else selected_question
    pdf = st.file_uploader("Upload a financial PDF (e.g. 10-K)", type="pdf")
    top_k = st.slider("Top matching pages", 1, 10, 4)

    if st.button("Answer from PDF"):
        if not final_question:
            st.warning("Enter or select a question.")
            st.stop()
        if not pdf:
            st.warning("Please upload a PDF.")
            st.stop()

        reader = PdfReader(BytesIO(pdf.read()))
        pages = [{"page": i, "text": p.extract_text() or ""} for i, p in enumerate(reader.pages)]

        corpus = [p["text"] for p in pages]
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(corpus)
        q_vec = vectorizer.transform([final_question])
        sims = cosine_similarity(q_vec, X).flatten()
        top_pages = sorted(zip(pages, sims), key=lambda x: -x[1])[:top_k]
        snippets = "\n\n".join([f"[Page {p['page']}]\n{p['text'][:1500]}" for p, _ in top_pages])

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant. Answer the question using the context only."},
                {"role": "user", "content": f"Question: {final_question}\n\nContext:\n{snippets}"}
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content
        st.subheader("Answer")
        st.write(answer)

        with st.expander("Sources"):
            for p, _ in top_pages:
                st.markdown(f"**Page {p['page']}**")
                st.write(p["text"][:1000] or "_No text extracted_")
