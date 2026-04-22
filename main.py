import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="💻 Computer Price Predictor",
    page_icon="💻",
    layout="wide",
)

# ════════════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    [data-testid="stMetric"] {
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid rgba(128,128,128,0.2);
        background: rgba(102,126,234,0.12);
    }
    [data-testid="stMetricLabel"]  { font-size: 13px !important; font-weight: 600 !important; opacity: 0.75; }
    [data-testid="stMetricValue"]  { font-size: 26px !important; font-weight: 700 !important; }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 16px; padding: 36px;
        text-align: center; box-shadow: 0 6px 24px rgba(102,126,234,0.35);
        margin: 16px 0;
    }
    .divider-label {
        font-size: 20px; font-weight: 700;
        letter-spacing: 0.5px; margin: 8px 0 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# DATA & MODEL
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("Computer_price.csv")
    df = df[df["brand"] != "Unknown"].copy()
    return df

@st.cache_resource
def train_models(df):
    le_brand = LabelEncoder()
    le_cpu   = LabelEncoder()
    df2 = df.copy()
    df2["brand_enc"] = le_brand.fit_transform(df2["brand"])
    df2["cpu_enc"]   = le_cpu.fit_transform(df2["cpu"])
    features = ["brand_enc", "cpu_enc", "ram_gb", "storage_gb", "screen_size_inches", "purchase_year"]
    X, y = df2[features], df2["price_usd"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, random_state=42),
        "Linear Regression": LinearRegression(),
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
            "y_test": y_test, "preds": preds,
        }
    return results, le_brand, le_cpu, features

df = load_data()
model_results, le_brand, le_cpu, features = train_models(df)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TITLE / HEADER / SUBHEADER
# ════════════════════════════════════════════════════════════════════════════
st.title("💻 Computer Price Prediction App")
st.header("ML-Powered Price Estimator for Laptops & Desktops")
st.subheader("Enter your computer specifications in the sidebar to get an instant price prediction.")
st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SIDEBAR (collect information)
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    st.header("🤖 Model")
    algo = st.selectbox("Algorithm", list(model_results.keys()))
    st.markdown("---")
    st.header("🖥️ Computer Specs")
    st.caption("Fill in the specifications of the computer you want to price:")
    brand   = st.selectbox("Brand",            sorted(df["brand"].unique()))
    cpu     = st.selectbox("CPU",              sorted(df["cpu"].unique()))
    ram     = st.select_slider("RAM (GB)",     options=[4, 8, 16, 32, 64], value=16)
    storage = st.select_slider("Storage (GB)", options=[128, 256, 512, 1024, 2048], value=512)
    screen  = st.slider("Screen Size (inches)", 13.0, 24.0, 15.6, step=0.1)
    year    = st.slider("Purchase Year", 2019, 2026, 2024)
    st.markdown("---")

    predict_btn = st.button("🚀 Predict Price", use_container_width=True, type="primary")

    if predict_btn:
        try:
            b_enc = le_brand.transform([brand])[0]
            c_enc = le_cpu.transform([cpu])[0]
            X_in  = pd.DataFrame([[b_enc, c_enc, ram, storage, screen, year]], columns=features)
            preds_all = {n: r["model"].predict(X_in)[0] for n, r in model_results.items()}
            st.session_state["preds_all"]   = preds_all
            st.session_state["pred_inputs"] = (brand, cpu, ram, storage, screen, year)
            st.session_state["predicted"]   = True
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Mini result in sidebar
    if st.session_state.get("predicted") and "preds_all" in st.session_state:
        chosen_sidebar = st.session_state["preds_all"][algo]
        st.markdown(
            f"""<div style='background:linear-gradient(135deg,#667eea,#764ba2);
                color:white;border-radius:12px;padding:16px;text-align:center;margin-top:4px;'>
                <div style='font-size:11px;opacity:0.8;margin-bottom:4px;'>Latest Prediction</div>
                <div style='font-size:30px;font-weight:700;'>${chosen_sidebar:,.0f}</div>
                <div style='font-size:11px;opacity:0.65;'>{algo}</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREDICTION RESULT
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider-label">💰 Prediction Result</div>', unsafe_allow_html=True)

if not st.session_state.get("predicted") or "preds_all" not in st.session_state:
    st.info("👈 Select your computer specifications in the sidebar and click **🚀 Predict Price** to see the result here.")
else:
    saved_brand, saved_cpu, saved_ram, saved_storage, saved_screen, saved_year = st.session_state["pred_inputs"]
    predictions = st.session_state["preds_all"]
    chosen_pred = predictions[algo]

    # Spec summary
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Brand",   saved_brand)
    s2.metric("CPU",     saved_cpu)
    s3.metric("RAM",     f"{saved_ram} GB")
    s4.metric("Storage", f"{saved_storage} GB")
    s5.metric("Screen",  f'{saved_screen}"')
    s6.metric("Year",    saved_year)

    # Main price box
    rmse_val = model_results[algo]["rmse"]
    st.markdown(
        f"""<div class='prediction-box'>
            <p style='margin:0;font-size:14px;opacity:0.85;letter-spacing:1px;text-transform:uppercase;'>
                Estimated Price &nbsp;·&nbsp; {algo}
            </p>
            <h1 style='margin:10px 0 4px 0;font-size:64px;letter-spacing:-1px;'>${chosen_pred:,.2f}</h1>
            <p style='margin:0;font-size:13px;opacity:0.7;'>
                Confidence range (±1 RMSE):&nbsp;
                ${max(0, chosen_pred - rmse_val):,.0f} – ${chosen_pred + rmse_val:,.0f}
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🤖 All Model Predictions")
        pred_df = pd.DataFrame(
            [(n, f"${p:,.2f}", "✅" if n == algo else "") for n, p in predictions.items()],
            columns=["Model", "Predicted Price", "Selected"]
        )
        st.dataframe(pred_df.set_index("Model"), use_container_width=True)

    with col_right:
        st.subheader("📊 How Does It Compare?")
        overall_avg   = df["price_usd"].mean()
        brand_avg_val = df[df["brand"] == saved_brand]["price_usd"].mean() if saved_brand in df["brand"].values else overall_avg
        c1, c2, c3 = st.columns(3)
        c1.metric("Your Prediction",    f"${chosen_pred:,.0f}")
        c2.metric("Dataset Average",    f"${overall_avg:,.0f}",   delta=f"{chosen_pred - overall_avg:+,.0f}")
        c3.metric(f"{saved_brand} Avg", f"${brand_avg_val:,.0f}", delta=f"{chosen_pred - brand_avg_val:+,.0f}")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EDA
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider-label">📊 Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
st.subheader("Dataset Overview")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Records",   len(df))
k2.metric("Avg Price (USD)", f"${df['price_usd'].mean():,.0f}")
k3.metric("Min Price",       f"${df['price_usd'].min():,.0f}")
k4.metric("Max Price",       f"${df['price_usd'].max():,.0f}")

with st.expander("🗂️ View Full Dataset", expanded=False):
    st.dataframe(df.drop(columns=["timestamp"]), use_container_width=True)

st.subheader("Visualizations")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Price Distribution**")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["price_usd"], bins=15, color="#667eea", edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Price (USD)"); ax.set_ylabel("Count")
    ax.set_title("Distribution of Computer Prices")
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig); plt.close()

with col_b:
    st.markdown("**Average Price by Brand**")
    brand_avg = df.groupby("brand")["price_usd"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(brand_avg.index, brand_avg.values,
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, len(brand_avg))))
    ax.set_xlabel("Avg Price (USD)"); ax.set_title("Average Price by Brand")
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, brand_avg.values):
        ax.text(val + 10, bar.get_y() + bar.get_height()/2, f"${val:,.0f}", va="center", fontsize=8)
    st.pyplot(fig); plt.close()

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("**RAM vs Price**")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(df["ram_gb"], df["price_usd"], alpha=0.75, color="#764ba2", edgecolor="white", s=70)
    ax.set_xlabel("RAM (GB)"); ax.set_ylabel("Price (USD)")
    ax.set_title("RAM vs Price"); ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig); plt.close()

with col_d:
    st.markdown("**Storage vs Price**")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(df["storage_gb"], df["price_usd"], alpha=0.75, color="#43b89c", edgecolor="white", s=70)
    ax.set_xlabel("Storage (GB)"); ax.set_ylabel("Price (USD)")
    ax.set_title("Storage vs Price"); ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig); plt.close()

st.subheader("Model Performance")
res = model_results[algo]
m1, m2, m3 = st.columns(3)
m1.metric("MAE",      f"${res['mae']:.2f}")
m2.metric("RMSE",     f"${res['rmse']:.2f}")
m3.metric("R² Score", f"{res['r2']:.4f}")

col_e, col_f = st.columns(2)
with col_e:
    st.markdown(f"**Actual vs Predicted — {algo}**")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(res["y_test"], res["preds"], color="#667eea", alpha=0.8, edgecolor="white", s=80)
    mn = min(res["y_test"].min(), res["preds"].min())
    mx = max(res["y_test"].max(), res["preds"].max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect Fit")
    ax.set_xlabel("Actual Price"); ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig); plt.close()

with col_f:
    if algo != "Linear Regression":
        st.markdown("**Feature Importance**")
        importances = res["model"].feature_importances_
        feat_df = pd.Series(importances, index=features).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        feat_df.plot(kind="barh", ax=ax, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(feat_df))))
        ax.set_title("Feature Importance"); ax.set_xlabel("Importance Score")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()
    else:
        st.markdown("**Residuals Distribution**")
        residuals = res["y_test"].values - res["preds"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(residuals, bins=12, color="#f093fb", edgecolor="white")
        ax.axvline(0, color="red", linestyle="--")
        ax.set_xlabel("Residual (USD)"); ax.set_ylabel("Count")
        ax.set_title("Residual Distribution"); ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close()

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — USER FEEDBACK
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider-label">💬 User Feedback</div>', unsafe_allow_html=True)
st.subheader("Help us improve the prediction model")
st.caption("Your feedback helps us understand how accurate the predictions are and where we can improve.")

if "feedbacks" not in st.session_state:
    st.session_state["feedbacks"] = []

with st.form("feedback_form", clear_on_submit=True):
    fb_col1, fb_col2 = st.columns(2)

    with fb_col1:
        fb_name   = st.text_input("Your Name (optional)", placeholder="e.g. John Doe")
        fb_rating = st.select_slider(
            "How accurate was the prediction?",
            options=["⭐ Very Poor", "⭐⭐ Poor", "⭐⭐⭐ Average", "⭐⭐⭐⭐ Good", "⭐⭐⭐⭐⭐ Excellent"],
            value="⭐⭐⭐ Average",
        )
        fb_useful = st.radio("Was the app useful to you?", ["Yes", "No", "Somewhat"], horizontal=True)

    with fb_col2:
        fb_actual = st.number_input(
            "Actual price you found (USD) — optional",
            min_value=0, max_value=99999, value=0, step=50,
            help="If you know the real price, enter it so we can track accuracy."
        )
        fb_comment = st.text_area(
            "Comments or suggestions",
            placeholder="e.g. The model predicted well but I expected a higher price for Apple M4...",
            height=120,
        )

    submitted = st.form_submit_button("📨 Submit Feedback", use_container_width=True, type="primary")
    if submitted:
        entry = {
            "Name":    fb_name or "Anonymous",
            "Rating":  fb_rating,
            "Useful":  fb_useful,
            "Actual $": f"${fb_actual:,}" if fb_actual > 0 else "—",
            "Comment": fb_comment or "—",
        }
        st.session_state["feedbacks"].append(entry)
        st.success("✅ Thank you for your feedback! It has been recorded.")

if st.session_state["feedbacks"]:
    st.markdown("#### 📋 Submitted Feedback")
    fb_df = pd.DataFrame(st.session_state["feedbacks"])
    fb_df.index = range(1, len(fb_df) + 1)
    st.dataframe(fb_df, use_container_width=True)

    ratings = [f["Rating"] for f in st.session_state["feedbacks"]]
    rating_counts = pd.Series(ratings).value_counts().sort_index()
    st.markdown("**Rating Summary**")
    r_cols = st.columns(len(rating_counts))
    for col, (label, count) in zip(r_cols, rating_counts.items()):
        col.metric(label, count)