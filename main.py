import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💻 Computer Price Predictor",
    page_icon="💻",
    layout="wide",
)

# ─── CSS Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fc; }
    .stMetric { background-color: #000000; border-radius: 12px; padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 16px; padding: 28px;
        text-align: center; box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }
    .section-title { color: #4A4A8A; font-weight: 700; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Load & Prepare Data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Computer_price.csv")
    df = df[df["brand"] != "Unknown"].copy()
    return df

@st.cache_resource
def train_models(df):
    le_brand = LabelEncoder()
    le_cpu = LabelEncoder()

    df2 = df.copy()
    df2["brand_enc"] = le_brand.fit_transform(df2["brand"])
    df2["cpu_enc"]   = le_cpu.fit_transform(df2["cpu"])

    features = ["brand_enc", "cpu_enc", "ram_gb", "storage_gb", "screen_size_inches", "purchase_year"]
    X = df2[features]
    y = df2["price_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest":       RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting":   GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, random_state=42),
        "Linear Regression":   LinearRegression(),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "mae":   mean_absolute_error(y_test, preds),
            "rmse":  np.sqrt(mean_squared_error(y_test, preds)),
            "r2":    r2_score(y_test, preds),
            "y_test": y_test,
            "preds":  preds,
        }

    return results, le_brand, le_cpu, features

df = load_data()
model_results, le_brand, le_cpu, features = train_models(df)

# ─── Header ─────────────────────────────────────────────────────────────────
st.title("💻 Computer Price Prediction App")
st.markdown("A machine-learning powered tool that estimates computer prices based on specs.")
st.write('team members:')
st.write('Leu sophal')
st.write('Ly Laisrun')
st.write('Horn Samborokisin')
st.write('Horn Hengveasna')
st.write('Moeung Devid')
st.divider()

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    algo = st.selectbox("Algorithm", list(model_results.keys()))
    st.divider()
    st.header("🔍 Predict a Price")
    st.caption("Enter computer specifications below:")

    brand  = st.selectbox("Brand",   sorted(df["brand"].unique()))
    cpu    = st.selectbox("CPU",     sorted(df["cpu"].unique()))
    ram    = st.select_slider("RAM (GB)", options=[4,8,16,32,64], value=16)
    storage= st.select_slider("Storage (GB)", options=[128,256,512,1024,2048], value=512)
    screen = st.slider("Screen Size (inches)", 13.0, 24.0, 15.6, step=0.1)
    year   = st.slider("Purchase Year", 2019, 2026, 2024)

    predict_btn = st.button("🚀 Predict Price", use_container_width=True, type="primary")

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "📈 Model Performance", "💰 Prediction"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – Dataset Overview
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Avg Price (USD)", f"${df['price_usd'].mean():,.0f}")
    col3.metric("Min Price", f"${df['price_usd'].min():,.0f}")
    col4.metric("Max Price", f"${df['price_usd'].max():,.0f}")

    st.markdown("### 📋 Sample Data")
    st.dataframe(df.drop(columns=["timestamp"]).head(10), use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Price Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df["price_usd"], bins=15, color="#667eea", edgecolor="white", linewidth=0.8)
        ax.set_xlabel("Price (USD)"); ax.set_ylabel("Count")
        ax.set_title("Price Distribution"); ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("#### Avg Price by Brand")
        brand_avg = df.groupby("brand")["price_usd"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.barh(brand_avg.index, brand_avg.values, color=plt.cm.viridis(np.linspace(0.2,0.8,len(brand_avg))))
        ax.set_xlabel("Avg Price (USD)")
        ax.set_title("Average Price by Brand")
        ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, brand_avg.values):
            ax.text(val + 10, bar.get_y() + bar.get_height()/2, f"${val:,.0f}", va="center", fontsize=8)
        st.pyplot(fig); plt.close()

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### RAM vs Price")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(df["ram_gb"], df["price_usd"], alpha=0.7, color="#764ba2", edgecolor="white", s=70)
        ax.set_xlabel("RAM (GB)"); ax.set_ylabel("Price (USD)")
        ax.set_title("RAM vs Price"); ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    with col_d:
        st.markdown("#### Storage vs Price")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(df["storage_gb"], df["price_usd"], alpha=0.7, color="#43b89c", edgecolor="white", s=70)
        ax.set_xlabel("Storage (GB)"); ax.set_ylabel("Price (USD)")
        ax.set_title("Storage vs Price"); ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – Model Performance
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🏆 Model Comparison")

    metrics_data = []
    for name, r in model_results.items():
        metrics_data.append({"Model": name, "MAE ($)": f"{r['mae']:.2f}", "RMSE ($)": f"{r['rmse']:.2f}", "R² Score": f"{r['r2']:.4f}"})
    st.dataframe(pd.DataFrame(metrics_data).set_index("Model"), use_container_width=True)

    res = model_results[algo]
    st.markdown(f"### 📉 {algo} — Actual vs Predicted")

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE",  f"${res['mae']:.2f}")
    col2.metric("RMSE", f"${res['rmse']:.2f}")
    col3.metric("R²",   f"{res['r2']:.4f}")

    col_e, col_f = st.columns(2)

    with col_e:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(res["y_test"], res["preds"], color="#667eea", alpha=0.8, edgecolor="white", s=80)
        mn, mx = min(res["y_test"].min(), res["preds"].min()), max(res["y_test"].max(), res["preds"].max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect Fit")
        ax.set_xlabel("Actual Price"); ax.set_ylabel("Predicted Price")
        ax.set_title("Actual vs Predicted"); ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()

    with col_f:
        if algo != "Linear Regression":
            st.markdown("#### Feature Importance")
            importances = res["model"].feature_importances_
            feat_df = pd.Series(importances, index=features).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feat_df)))
            feat_df.plot(kind="barh", ax=ax, color=colors)
            ax.set_title("Feature Importance"); ax.set_xlabel("Importance Score")
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig); plt.close()
        else:
            st.markdown("#### Residuals Distribution")
            residuals = res["y_test"].values - res["preds"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(residuals, bins=12, color="#f093fb", edgecolor="white")
            ax.axvline(0, color="red", linestyle="--")
            ax.set_xlabel("Residual (USD)"); ax.set_ylabel("Count")
            ax.set_title("Residual Distribution"); ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 💰 Price Prediction")
    st.info("👈 Configure the specs in the sidebar, then click **Predict Price**.")

    # Show input summary
    st.markdown("#### 🖥️ Selected Configuration")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    cfg_col1.markdown(f"**Brand:** {brand}  \n**CPU:** {cpu}")
    cfg_col2.markdown(f"**RAM:** {ram} GB  \n**Storage:** {storage} GB")
    cfg_col3.markdown(f"**Screen:** {screen}\"  \n**Year:** {year}")

    st.divider()

    if predict_btn:
        try:
            brand_enc  = le_brand.transform([brand])[0]
            cpu_enc    = le_cpu.transform([cpu])[0]
        except ValueError as e:
            st.error(f"Encoding error: {e}")
            st.stop()

        X_input = pd.DataFrame([[brand_enc, cpu_enc, ram, storage, screen, year]], columns=features)

        # Predict with all models
        predictions = {name: r["model"].predict(X_input)[0] for name, r in model_results.items()}
        chosen_pred = predictions[algo]

        st.markdown(
            f"""<div class='prediction-box'>
                <h2 style='margin:0; font-size:18px; opacity:0.9;'>Estimated Price ({algo})</h2>
                <h1 style='margin:8px 0 0 0; font-size:48px;'>${chosen_pred:,.2f}</h1>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("")

        st.markdown("#### 🤖 All Model Predictions")
        pred_df = pd.DataFrame(
            [(name, f"${p:,.2f}") for name, p in predictions.items()],
            columns=["Model", "Predicted Price"]
        )
        st.dataframe(pred_df.set_index("Model"), use_container_width=True)

        # Confidence range (±1 RMSE of chosen model)
        rmse = model_results[algo]["rmse"]
        lo, hi = max(0, chosen_pred - rmse), chosen_pred + rmse
        st.markdown(f"**Confidence Range (±1 RMSE):** ${lo:,.2f} – ${hi:,.2f}")

        # Compare against dataset averages
        st.markdown("#### 📊 How does this compare?")
        overall_avg = df["price_usd"].mean()
        brand_avg_price = df[df["brand"] == brand]["price_usd"].mean() if brand in df["brand"].values else overall_avg

        comp_col1, comp_col2, comp_col3 = st.columns(3)
        comp_col1.metric("Your Prediction",  f"${chosen_pred:,.0f}")
        comp_col2.metric("Dataset Avg",       f"${overall_avg:,.0f}", delta=f"{chosen_pred - overall_avg:+,.0f}")
        comp_col3.metric(f"{brand} Avg",      f"${brand_avg_price:,.0f}", delta=f"{chosen_pred - brand_avg_price:+,.0f}")
    else:
        st.markdown(
            "<div style='text-align:center; color:#aaa; padding:60px 0; font-size:18px;'>"
            "🎯 Configure specs in the sidebar and click <b>Predict Price</b></div>",
            unsafe_allow_html=True,
        )
    
