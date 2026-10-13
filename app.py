from pathlib import Path
from datetime import datetime, timezone
import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="India Landslide Watch", page_icon="⛰️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif}
.stApp {background:radial-gradient(circle at 80% 0%,rgba(35,100,93,.18),transparent 30%),#0b1220}
.block-container{max-width:1500px;padding-top:1.2rem;padding-bottom:2.5rem}
[data-testid="stSidebar"]{background:#101a2b;border-right:1px solid #26384b}
.hero{padding:1.5rem 1.7rem;border:1px solid #26384b;border-radius:20px;background:linear-gradient(120deg,#183644,#111b2c);margin-bottom:1rem}
.hero h1{font-family:'Manrope',sans-serif;letter-spacing:-1px;font-size:clamp(2rem,4vw,3rem);margin:0}
.hero p{color:#bdcbd9;margin:.4rem 0 0}
.eyebrow{color:#78d6c6;text-transform:uppercase;letter-spacing:2px;font-size:.72rem;font-weight:700}
.section{font-family:'Manrope',sans-serif;font-weight:800;font-size:1.35rem;margin:1.25rem 0 .3rem}
.muted{color:#9caec2;font-size:.9rem}
div[data-testid="stMetric"]{background:#121e30;border:1px solid #26384b;padding:1rem;border-radius:15px}
hr{border-color:#26384b}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
 <div class="eyebrow">Geospatial monitoring workspace</div>
 <h1>⛰️ India Landslide Watch</h1>
 <p>Multi-location susceptibility screening, forecast rainfall monitoring and transparent evidence.</p>
</div>
""", unsafe_allow_html=True)

st.error(
    "SAFETY LIMITATION: This is a research decision-support dashboard, not an authorized public warning system. "
    "Do not use its outputs alone to decide evacuation, sheltering, road closure or other life-safety actions. "
    "Follow alerts and instructions from IMD, NDMA/SDMA, district administration and emergency services."
)

CSV_NAME = "INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv"
ROOT = Path(__file__).parent
CANDIDATES = [ROOT / CSV_NAME, ROOT / "data" / CSV_NAME]

LABEL_CANDIDATES = ["label", "target", "class", "landslide_label"]
EXCLUDE = {
    "source_record_id", "sample_type", "label_interpretation", "spatial_block_id",
    "predictor_source_elevation", "predictor_source_slope_aspect", "predictor_source_landcover",
    "predictor_source_ndvi", "event_date", "date", "longitude", "latitude"
}

def parse_labels(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    ss = s.astype("string").str.strip().str.lower()
    mapping = {"1":1,"true":1,"landslide":1,"event":1,"positive":1,
               "0":0,"false":0,"background":0,"non-landslide":0,"negative":0}
    mapped = ss.map(mapping)
    return mapped.where(mapped.notna(), pd.to_numeric(ss, errors="coerce"))

def train(df):
    label = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label is None:
        raise ValueError("CSV needs a binary 'label' column.")
    yy = parse_labels(df[label])
    mask = yy.notna() & yy.isin([0, 1])
    d = df.loc[mask].copy()
    y = yy.loc[mask].astype(int)
    if y.nunique() != 2:
        raise ValueError("Both class 0 (sampled background) and class 1 (recorded event) are required.")
    features = []
    for c in d.columns:
        if c == label or c in EXCLUDE:
            continue
        if c == "landcover_class":
            if d[c].notna().any(): features.append(c)
        elif pd.to_numeric(d[c], errors="coerce").notna().any():
            features.append(c)
    if not features:
        raise ValueError("No usable predictor columns found.")
    X = d[features].copy()
    cats = [c for c in features if c == "landcover_class" or X[c].dtype == "object" or str(X[c].dtype).startswith("string")]
    nums = [c for c in features if c not in cats]
    for c in nums:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf,-np.inf], np.nan)
    for c in cats:
        X[c] = X[c].astype("string").fillna("Unknown").astype(str)
    transformers = []
    if nums:
        transformers.append(("num", SimpleImputer(strategy="median", keep_empty_features=True), nums))
    if cats:
        transformers.append(("cat", Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
                                               ("ohe",OneHotEncoder(handle_unknown="ignore"))]), cats))
    pipe = Pipeline([("prep",ColumnTransformer(transformers)),
                     ("rf",RandomForestClassifier(n_estimators=350,min_samples_leaf=2,
                         class_weight="balanced",random_state=20261014,n_jobs=-1))])
    pipe.fit(X, y)
    return pipe, features, nums, cats, label, d, y

@st.cache_data(show_spinner=False)
def read_local(path):
    return pd.read_csv(path, low_memory=False)

@st.cache_data(ttl=600, show_spinner=False)
def get_forecast_cached(lat, lon, days, refresh_token):
    # refresh_token is changed by the refresh button, forcing a fresh request.
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude":float(lat),"longitude":float(lon),
              "daily":"precipitation_sum,precipitation_probability_max",
              "forecast_days":int(days),"timezone":"auto"}
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def get_forecast(lat, lon, days, token):
    return get_forecast_cached(round(float(lat), 4), round(float(lon), 4), int(days), int(token))

with st.sidebar:
    st.markdown("## Data & monitoring")
    upload = st.file_uploader("Training dataset CSV", type=["csv"])
    site_file = st.file_uploader("Optional multi-location CSV", type=["csv"], key="sites")
    st.caption("Multi-location CSV requires columns: site, latitude, longitude. Optional predictor columns can be included.")
    days = st.selectbox("Forecast horizon", [1, 3, 7], index=1)
    if "refresh_token" not in st.session_state: st.session_state.refresh_token = 0
    if st.button("↻ Refresh forecast data", use_container_width=True):
        st.session_state.refresh_token += 1
        st.cache_data.clear()
        st.rerun()
    st.caption("Forecasts are model-generated forecasts, not live rain-gauge observations.")

# Load dataset
try:
    if upload is not None:
        df = pd.read_csv(upload, low_memory=False)
        source = "Uploaded CSV"
    else:
        p = next((x for x in CANDIDATES if x.exists()), None)
        if p is None:
            st.info(f"Upload `{CSV_NAME}` in the sidebar or commit it beside app.py in your GitHub repository.")
            st.stop()
        df, source = read_local(str(p)), p.name
    model, features, nums, cats, label_col, train_df, y = train(df)
except Exception as e:
    st.error(f"Dataset/model error: {e}")
    st.stop()

top = st.columns(4)
top[0].metric("Training rows", f"{len(train_df):,}")
top[1].metric("Recorded-event labels", f"{int((y==1).sum()):,}")
top[2].metric("Sampled-background labels", f"{int((y==0).sum()):,}")
top[3].metric("Predictors", f"{len(features)}")
st.caption(f"Dataset: {source} · New model fitted from this CSV at startup. It does not load the exact prior research model.")

st.markdown('<div class="section">1 · Monitor multiple locations</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Enter sites one by one or upload a CSV. Coordinate-specific predictor values are needed for meaningful site-specific susceptibility scores.</div>', unsafe_allow_html=True)

default_sites = pd.DataFrame([
    {"site":"Nilgiris sample site","latitude":11.40,"longitude":76.70},
    {"site":"Wayanad sample site","latitude":11.68,"longitude":76.13},
    {"site":"Darjeeling sample site","latitude":27.04,"longitude":88.26},
])
if site_file is not None:
    try:
        sites = pd.read_csv(site_file)
    except Exception as e:
        st.error(f"Could not read location CSV: {e}")
        st.stop()
else:
    sites = st.data_editor(default_sites, num_rows="dynamic", use_container_width=True, key="sites_editor")

required = {"latitude","longitude"}
if not required.issubset(set(sites.columns)):
    st.error("Location table must include latitude and longitude columns.")
    st.stop()
sites = sites.copy()
sites["latitude"] = pd.to_numeric(sites["latitude"], errors="coerce")
sites["longitude"] = pd.to_numeric(sites["longitude"], errors="coerce")
sites = sites.dropna(subset=["latitude","longitude"])
sites = sites[sites.latitude.between(6,37.5) & sites.longitude.between(68,97.5)]
if "site" not in sites.columns: sites["site"] = [f"Site {i+1}" for i in range(len(sites))]
if sites.empty:
    st.warning("Add at least one valid Indian coordinate.")
    st.stop()

# Predictor values for susceptibility. A site CSV may supply columns with matching predictor names.
def median_or_mode(col):
    if col in cats:
        m = train_df[col].dropna().astype(str).mode()
        return m.iloc[0] if len(m) else "Unknown"
    s = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
    return float(s.median()) if len(s) else 0.0

def site_predictor_row(site_row):
    vals = {}
    for c in features:
        if c in site_row.index and pd.notna(site_row[c]):
            vals[c] = site_row[c]
        else:
            vals[c] = median_or_mode(c)
    return vals

prediction_rows = []
for _, sr in sites.iterrows():
    row = site_predictor_row(sr)
    x = pd.DataFrame([row], columns=features)
    for c in nums: x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf,-np.inf],np.nan)
    for c in cats: x[c] = x[c].astype("string").fillna("Unknown").astype(str)
    try:
        p_event = float(model.predict_proba(x)[0, list(model.classes_).index(1)])
    except Exception:
        p_event = np.nan
    prediction_rows.append({"site":str(sr["site"]), "latitude":float(sr.latitude),
                            "longitude":float(sr.longitude), "model_score":p_event,
                            "predictor_source":"site CSV values" if all(c in sr.index and pd.notna(sr.get(c)) for c in features) else "training medians/modes for missing predictors"})
pred_df = pd.DataFrame(prediction_rows)
pred_df["screening_band"] = pd.cut(pred_df.model_score, [-0.001,0.4,0.7,1.001],
                                    labels=["Lower relative score","Moderate relative score","Higher relative score"])
st.dataframe(pred_df, use_container_width=True, hide_index=True)
fig_map = px.scatter_mapbox(pred_df, lat="latitude", lon="longitude", color="model_score",
    size=np.clip(pred_df["model_score"].fillna(0.01)*30, 6, 22), hover_name="site",
    hover_data={"model_score":":.3f","screening_band":True,"latitude":":.4f","longitude":":.4f"},
    zoom=3.1, height=480, mapbox_style="open-street-map",
    color_continuous_scale=["#4fbf93","#f2bd4b","#ee6b67"],
    title="Monitoring sites — model screening score")
fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=55,b=0))
st.plotly_chart(fig_map, use_container_width=True)
st.caption("If predictor values are missing from the site CSV, the app uses training medians/modes. Such scores are not genuinely site-specific until predictors are supplied from verified spatial data.")

st.markdown('<div class="section">2 · Rainfall forecast for every location</div>', unsafe_allow_html=True)
st.caption("The forecast is requested separately for each coordinate. Use Refresh forecast data to force a new request. Forecasts can remain unchanged if the provider's forecast has not changed.")
rain_rows = []
progress = st.progress(0, text="Requesting forecasts…")
for i, (_, sr) in enumerate(sites.iterrows(), start=1):
    try:
        fc = get_forecast(sr.latitude, sr.longitude, days, st.session_state.refresh_token)
        daily = fc.get("daily", {})
        dates = daily.get("time", [])
        vals = daily.get("precipitation_sum", [])
        probs = daily.get("precipitation_probability_max", [None]*len(dates))
        for j, date in enumerate(dates):
            rain_rows.append({
                "site":str(sr["site"]), "latitude":float(sr.latitude), "longitude":float(sr.longitude),
                "date":date, "rainfall_mm":float(vals[j] or 0),
                "precipitation_probability_pct":probs[j],
                "forecast_timezone":fc.get("timezone","unknown"),
                "fetched_utc":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            })
    except Exception as e:
        rain_rows.append({"site":str(sr["site"]), "latitude":float(sr.latitude),
                          "longitude":float(sr.longitude), "date":"", "rainfall_mm":np.nan,
                          "precipitation_probability_pct":np.nan, "forecast_timezone":"unavailable",
                          "fetched_utc":str(e)})
    progress.progress(i/len(sites), text=f"Forecasts requested: {i}/{len(sites)}")
progress.empty()

rain_df = pd.DataFrame(rain_rows)
if rain_df.empty or rain_df["rainfall_mm"].isna().all():
    st.warning("Rainfall provider returned no usable forecasts. Check the connection and refresh.")
else:
    summary = rain_df.groupby("site", as_index=False).agg(
        forecast_total_mm=("rainfall_mm","sum"),
        max_daily_mm=("rainfall_mm","max")
    )
    merged = pred_df.merge(summary, on="site", how="left")
    # Illustrative screening flags only, not official warning categories.
    def flag(row):
        if pd.isna(row.forecast_total_mm): return "Unavailable"
        if row.forecast_total_mm >= 100 or row.max_daily_mm >= 50: return "High rainfall screen"
        if row.forecast_total_mm >= 50 or row.max_daily_mm >= 25: return "Raised rainfall screen"
        return "No flag from provisional rule"
    merged["rainfall_screen"] = merged.apply(flag, axis=1)
    st.dataframe(merged[["site","forecast_total_mm","max_daily_mm","rainfall_screen"]], use_container_width=True, hide_index=True)
    plot_df = rain_df.dropna(subset=["rainfall_mm"]).copy()
    if len(plot_df):
        f = px.line(plot_df, x="date", y="rainfall_mm", color="site", markers=True,
                    template="plotly_dark", title="Forecast rainfall by site (mm/day)")
        f.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=55,b=5))
        st.plotly_chart(f, use_container_width=True)
    with st.expander("Detailed daily forecast data"):
        st.dataframe(rain_df, use_container_width=True, hide_index=True)
    csv_bytes = rain_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download rainfall results CSV", csv_bytes, "india_landslide_rainfall_forecasts.csv", "text/csv")

st.markdown('<div class="section">3 · What this can and cannot tell you</div>', unsafe_allow_html=True)
st.info(
    "This dashboard combines a susceptibility screening score with a separate forecast-rainfall view. "
    "It does not calculate an independently validated probability of landslide occurrence, and its rainfall rules "
    "are not official thresholds. Your earlier ROC-AUC (~0.928) measured discrimination between recorded events and "
    "sampled backgrounds under a research sampling design; it is not an evacuation threshold."
)
st.markdown("""
**For life-safety use, do not remove these safeguards.** A deployable public warning service requires formal validation with
independent dated events and non-event periods, calibrated warning thresholds, measured lead time, missed-event and
false-alarm analysis, reliable event-time rainfall observations/forecasts, redundant data feeds, uptime monitoring,
and an agreed alert protocol with the relevant disaster-management authority. This dashboard must not issue evacuation orders.
""")
st.caption(f"Last dashboard render: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Rainfall provider: Open-Meteo forecast API")
