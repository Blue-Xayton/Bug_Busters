# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import base64
import scipy.stats as stats

st.set_page_config(
    page_title="Electricity Demand Forecast & Bias Audit",
    layout="wide",
)

# --- Helper functions -------------------------------------------------------
def encode_region_series(series):
    """Return encoded series and mapping dict."""
    if series.dtype.name == "category" or series.dtype == object:
        cats = series.astype("category")
        codes = cats.cat.codes
        mapping = dict(enumerate(cats.cat.categories))
        return codes, mapping
    else:
        # assume numeric already
        uniques = sorted(series.unique())
        mapping = {i: v for i, v in enumerate(uniques)}
        return series, mapping

def summary_metrics_by_group(df, group_col, true_col="Demand_MW", pred_col="prediction"):
    groups = []
    for g, sub in df.groupby(group_col):
        mae = mean_absolute_error(sub[true_col], sub[pred_col])
        rmse = mean_squared_error(sub[true_col], sub[pred_col], squared=False)
        mean_err = (sub[true_col] - sub[pred_col]).mean()
        var_err = (sub[true_col] - sub[pred_col]).var()
        groups.append({
            group_col: g,
            "count": len(sub),
            "MAE": mae,
            "RMSE": rmse,
            "Mean Error": mean_err,
            "Error Variance": var_err
        })
    return pd.DataFrame(groups)

def download_link(df: pd.DataFrame, name="audit_results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}">⬇️ Download audit CSV</a>'
    return href

# --- Top header --------------------------------------------------------------
st.markdown("<h1 style='margin-bottom:0.15rem;'>⚡ Electricity Demand Forecast & Bias Audit</h1>", unsafe_allow_html=True)
st.markdown("<div style='color: #5b5b5b; margin-bottom:1.2rem;'>Train ARIMAX forecasts, inspect group errors, and run a bias audit by Region & Day of Week.</div>", unsafe_allow_html=True)
st.write("---")

# --- Sidebar controls -------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    retrain_on_upload = st.checkbox("Retrain model on uploaded file (recommended)", value=True)
    arima_p = st.number_input("AR order (p)", min_value=0, max_value=10, value=1)
    arima_d = st.number_input("I order (d)", min_value=0, max_value=2, value=0)
    arima_q = st.number_input("MA order (q)", min_value=0, max_value=10, value=4)
    steps = st.number_input("Forecast steps", min_value=1, max_value=168, value=24)
    st.markdown("---")
    st.markdown("Tip: Provide a dataset with `dateTime`, `Demand_MW`, `Region`, `dayOfWeek`, `Temperature` columns.")

# --- File upload ------------------------------------------------------------
uploaded = st.file_uploader("Upload dataset (CSV or XLSX)", type=["csv", "xlsx"])
sample_df = None
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.subheader("Preview data")
    st.dataframe(df.head())

    # Basic checks
    required_cols = {"dateTime", "Demand_MW", "Region", "dayOfWeek", "Temperature"}
    if not required_cols.issubset(set(df.columns)):
        st.warning(f"Missing required columns. Found: {list(df.columns)}\nRequired: {sorted(list(required_cols))}")
    else:
        # Prepare dataframe
        df["dateTime"] = pd.to_datetime(df["dateTime"])
        df = df.sort_values("dateTime").reset_index(drop=True)

        # Ensure dayOfWeek numeric 0-6
        if df["dayOfWeek"].dtype == object:
            try:
                df["dayOfWeek"] = df["dayOfWeek"].astype(int)
            except:
                # attempt parse weekday names (Mon, Tue...)
                mapping_week = {name: i for i, name in enumerate(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])}
                df["dayOfWeek"] = df["dayOfWeek"].map(lambda x: mapping_week.get(str(x), np.nan))

        # Encode region & capture mapping
        region_codes, region_map = encode_region_series(df["Region"])
        df["Region_code"] = region_codes

        st.info("Region mapping (code -> original):")
        st.json(region_map)

        # store sample
        sample_df = df.copy()
else:
    st.info("Upload a dataset to begin. If you want to use your local sample, place it in sample_data and uncomment code in the script.")
    st.stop()

# --- Model training / load --------------------------------------------------
model = None
if retrain_on_upload and sample_df is not None:
    st.header("Model training")
    if st.button("Train ARIMAX on uploaded dataset"):
        with st.spinner("Training ARIMAX..."):
            try:
                y = sample_df["Demand_MW"]
                exog = sample_df[["Temperature", "Region_code", "dayOfWeek"]]
                model = ARIMA(y, order=(int(arima_p), int(arima_d), int(arima_q)), exog=exog).fit()
                st.success("Model trained.")
                st.code(str(model.summary().tables[0]), language="text")
                st.session_state["fitted_model"] = model
            except Exception as e:
                st.error(f"Training failed: {e}")
else:
    # if user chose not to retrain, check if fitted model in session
    model = st.session_state.get("fitted_model", None)
    if model is None:
        st.warning("No model available. Please train the model.")
        st.stop()

# --- Forecast UI ------------------------------------------------------------
st.header("Forecast")
col1, col2, col3, col4 = st.columns([2,2,2,1])
with col1:
    future_temp = st.number_input("Future Temperature (single value or average)", value=float(sample_df["Temperature"].median()))
with col2:
    # show region selection using mapping
    inv_map = {v:k for k,v in region_map.items()}  # original -> code (if needed)
    region_options = list(region_map.values())
    sel_region = st.selectbox("Region (original label)", options=region_options)
    sel_region_code = [k for k,v in region_map.items() if v == sel_region][0]
with col3:
    sel_day = st.selectbox("Day of Week (0=Mon)", options=sorted(sample_df["dayOfWeek"].unique()))
with col4:
    run_forecast = st.button("Run Forecast")

if run_forecast:
    if model is None:
        st.error("No trained model found.")
    else:
        future_exog = pd.DataFrame({
            "Temperature": [float(future_temp)] * int(steps),
            "Region_code": [sel_region_code] * int(steps),
            "dayOfWeek": [int(sel_day)] * int(steps)
        })
        try:
            forecast = model.forecast(steps=int(steps), exog=future_exog)
            # Build forecast time index (hourly as default)
            last_ts = sample_df["dateTime"].iloc[-1]
            freq = pd.infer_freq(sample_df["dateTime"])
            if freq is None:
                # fallback to hourly
                freq = "H"
            future_index = pd.date_range(last_ts, periods=int(steps)+1, freq=freq)[1:]
            fc_df = pd.DataFrame({"dateTime": future_index, "forecast": forecast})
            st.subheader("Forecast preview")
            st.dataframe(fc_df.head(20))

            # plot historical + forecast
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(sample_df["dateTime"], sample_df["Demand_MW"], label="Historical", linewidth=1)
            ax.plot(fc_df["dateTime"], fc_df["forecast"], label="Forecast", linestyle="--", linewidth=2)
            ax.set_title("Historical Demand + Forecast")
            ax.set_xlabel("Time")
            ax.set_ylabel("Demand (MW)")
            ax.legend()
            st.pyplot(fig)

            st.session_state["last_forecast_df"] = fc_df
        except Exception as e:
            st.error(f"Forecast failed: {e}")

# --- Bias Audit -------------------------------------------------------------
st.header("Bias / Fairness Audit")
st.write("We compute error-based fairness metrics across groups. These are interpretable for forecasting tasks.")

# recompute predictions on training set for audit
try:
    fitted_model = st.session_state["fitted_model"]
except KeyError:
    fitted_model = model

sample_df["prediction"] = fitted_model.predict(start=0, end=len(sample_df)-1, exog=sample_df[["Temperature", "Region_code", "dayOfWeek"]])
sample_df["error"] = sample_df["Demand_MW"] - sample_df["prediction"]

# generate metrics for Region and Day of week
metrics_region = summary_metrics_by_group(sample_df, "Region_code", "Demand_MW", "prediction")
# map back readable region label
metrics_region["Region_label"] = metrics_region["Region_code"].map(region_map)

metrics_dow = summary_metrics_by_group(sample_df, "dayOfWeek", "Demand_MW", "prediction")
metrics_dow = metrics_dow.sort_values("dayOfWeek")

st.subheader("Region-level metrics")
st.dataframe(metrics_region[["Region_label","count","MAE","RMSE","Mean Error","Error Variance"]].sort_values("MAE"))

# bar chart MAE by region
fig1, ax1 = plt.subplots()
ax1.bar(metrics_region["Region_label"], metrics_region["MAE"])
ax1.set_title("MAE by Region")
ax1.set_ylabel("MAE")
ax1.set_xticklabels(metrics_region["Region_label"], rotation=45, ha="right")
st.pyplot(fig1)

st.subheader("Day-of-week metrics")
st.dataframe(metrics_dow[["dayOfWeek","count","MAE","RMSE","Mean Error","Error Variance"]])

fig2, ax2 = plt.subplots()
ax2.plot(metrics_dow["dayOfWeek"], metrics_dow["MAE"], marker="o")
ax2.set_title("MAE by Day of Week")
ax2.set_xlabel("Day of week (0=Mon)")
ax2.set_ylabel("MAE")
st.pyplot(fig2)

# simple statistical test: compare top vs bottom region MAE
top_region = metrics_region.sort_values("MAE", ascending=False).iloc[0]["Region_code"]
bottom_region = metrics_region.sort_values("MAE", ascending=True).iloc[0]["Region_code"]

top_errors = sample_df[sample_df["Region_code"] == top_region]["error"]
bottom_errors = sample_df[sample_df["Region_code"] == bottom_region]["error"]

tstat, pval = stats.ttest_ind(top_errors.dropna(), bottom_errors.dropna(), equal_var=False)
st.markdown("### Quick statistical check")
st.write(f"Comparing error distributions: **Top MAE region** vs **Bottom MAE region** — t-stat = {tstat:.3f}, p = {pval:.3e}")
if pval < 0.05:
    st.error("The difference in errors between regions is statistically significant (p < 0.05). This suggests disparate performance.")
else:
    st.success("No statistically significant difference detected between the two extreme regions (p >= 0.05).")

# Export audit table
audit_export = metrics_region[["Region_label","count","MAE","RMSE","Mean Error","Error Variance"]].rename(columns={"Region_label":"Region"})
st.markdown(download_link(audit_export, name="region_audit.csv"), unsafe_allow_html=True)

# --- Ethics / Narrative -----------------------------------------------------
st.header("Ethics & Recommendations (auto-generated)")
mean_gap_region = metrics_region["MAE"].max() - metrics_region["MAE"].min()
st.write(f"- **Region MAE gap:** {mean_gap_region:.2f} MW. Larger gaps indicate unequal performance across regions.")
st.write("- Possible real-world harms: under-prediction in a region could lead to under-supply and outages; over-prediction may cause unnecessary cost or over-provisioning.")
st.write("- Short recommendations:")
st.write("  1. Evaluate class- or region-specific features and consider adding local covariates (local weather stations).")
st.write("  2. Consider reweighting examples from underperforming regions or separate local models per region.")
st.write("  3. Run post-processing corrections (calibration) for groups with systematic mean error.")
st.write("  4. Track performance over time and involve domain stakeholders for acceptability thresholds.")

st.write("---")
st.info("If you want I can now add interactive bias mitigation choices (reweighing, group calibration) and show Before/After comparisons.")
