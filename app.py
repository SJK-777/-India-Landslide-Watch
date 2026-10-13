from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="India Landslide Watch", page_icon="⛰️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(circle at 85% 0%,rgba(40,120,110,.16),transparent 28%),#0b1220}
.block-container{max-width:1280px;padding-top:1.4rem;padding-bottom:2.5rem}
[data-testid="stSidebar"]{background:#101a2b;border-right:1px solid #26384b}
.hero{padding:1.5rem 1.7rem;border:1px solid #26384b;border-radius:20px;background:linear-gradient(120deg,#183644,#111b2c);margin-bottom:1.1rem}
.hero h1{font-family:'Manrope',sans-serif;letter-spacing:-1px;font-size:clamp(2rem,4vw,3rem);margin:0}
.hero p{color:#bdcbd9;margin:.4rem 0 0}
.eyebrow{color:#78d6c6;text-transform:uppercase;letter-spacing:2px;font-size:.72rem;font-weight:700}
.section{font-family:'Manrope',sans-serif;font-weight:800;font-size:1.3rem;margin:1.3rem 0 .3rem}
.muted{color:#9caec2;font-size:.9rem}
div[data-testid="stMetric"]{background:#121e30;border:1px solid #26384b;padding:1rem;border-radius:15px}
div.stButton>button{border-radius:11px;font-weight:700;min-height:2.7rem}
hr{border-color:#26384b}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
 <div class="eyebrow">Landslide monitoring · Location assessment</div>
 <h1>⛰️ India Landslide Watch</h1>
 <p>Choose a location, enter environmental conditions, and review a clear susceptibility and rainfall summary.</p>
</div>
""", unsafe_allow_html=True)

st.warning("Decision-support only. This dashboard is not an authorized evacuation or public-warning system. Follow official IMD and district disaster-management instructions.")

ROOT = Path(__file__).parent
CSV_NAME = "INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv"
CSV_PATHS = [ROOT / CSV_NAME, ROOT / "data" / CSV_NAME]
LABEL_CANDIDATES = ["label", "target", "class", "landslide_label"]
EXCLUDE = {
    "source_record_id","sample_type","label_interpretation","spatial_block_id",
    "predictor_source_elevation","predictor_source_slope_aspect",
    "predictor_source_landcover","predictor_source_ndvi","event_date","date",
    "longitude","latitude"
}

def parse_labels(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    ss = s.astype("string").str.strip().str.lower()
    mapping = {"1":1,"true":1,"landslide":1,"event":1,"positive":1,
               "0":0,"false":0,"background":0,"non-landslide":0,"negative":0}
    v = ss.map(mapping)
    return v.where(v.notna(), pd.to_numeric(ss, errors="coerce"))

@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path, low_memory=False)

def fit_model(df):
    label = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label is None:
        raise ValueError("Training CSV must contain a binary 'label' column.")
    y0 = parse_labels(df[label])
    mask = y0.notna() & y0.isin([0,1])
    d = df.loc[mask].copy()
    y = y0.loc[mask].astype(int)
    if y.nunique() != 2:
        raise ValueError("The CSV must include both label 1 (recorded event) and label 0 (sampled background).")
    features = []
    for c in d.columns:
        if c == label or c in EXCLUDE:
            continue
        if c == "landcover_class":
            if d[c].notna().any(): features.append(c)
        elif pd.to_numeric(d[c], errors="coerce").notna().any():
            features.append(c)
    if not features:
        raise ValueError("No environmental predictor columns were found.")
    X = d[features].copy()
    cats = [c for c in features if c == "landcover_class" or X[c].dtype == "object" or str(X[c].dtype).startswith("string")]
    nums = [c for c in features if c not in cats]
    for c in nums:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf,-np.inf],np.nan)
    for c in cats:
        X[c] = X[c].astype("string").fillna("Unknown").astype(str)
    transformers = []
    if nums:
        transformers.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), nums))
    if cats:
        transformers.append(("categorical", Pipeline([
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore"))
        ]), cats))
    pipe = Pipeline([
        ("prep",ColumnTransformer(transformers)),
        ("model",RandomForestClassifier(n_estimators=350,min_samples_leaf=2,class_weight="balanced",
                                        random_state=20261014,n_jobs=-1))
    ])
    pipe.fit(X,y)
    return pipe,features,nums,cats,d,y

@st.cache_data(ttl=600, show_spinner=False)
def get_forecast(lat,lon,days,refresh_id):
    r = requests.get("https://api.open-meteo.com/v1/forecast",params={
        "latitude":float(lat),"longitude":float(lon),
        "daily":"precipitation_sum,precipitation_probability_max",
        "forecast_days":int(days),"timezone":"auto"
    },timeout=25)
    r.raise_for_status()
    return r.json()

with st.sidebar:
    st.markdown("## Location & forecast")
    lat = st.number_input("Latitude", min_value=6.0,max_value=37.5,value=11.40,step=0.01,format="%.4f")
    lon = st.number_input("Longitude", min_value=68.0,max_value=97.5,value=76.70,step=0.01,format="%.4f")
    horizon = st.selectbox("Forecast period", [1,3,7], index=1)
    if "refresh_id" not in st.session_state: st.session_state.refresh_id = 0
    if st.button("Refresh rainfall forecast", use_container_width=True):
        st.session_state.refresh_id += 1
        st.cache_data.clear()
        st.rerun()

# Data is loaded in the background only; no dataset previews or upload controls shown in the main UI.
try:
    p = next((x for x in CSV_PATHS if x.exists()), None)
    if p is None:
        st.error(f"Training data file not found. Add `{CSV_NAME}` beside app.py in your GitHub repository.")
        st.stop()
    df = load_csv(str(p))
    with st.spinner("Loading the model…"):
        model, features, nums, cats, train_df, y = fit_model(df)
except Exception as e:
    st.error(f"Could not prepare the prediction model: {e}")
    st.stop()

st.markdown('<div class="section">1 · Enter site conditions</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Enter values measured or extracted for this location. Defaults are dataset medians/modes, not local measurements.</div>', unsafe_allow_html=True)

def default_value(c):
    if c in cats:
        m = train_df[c].dropna().astype(str).mode()
        return m.iloc[0] if len(m) else "Unknown"
    v = pd.to_numeric(train_df[c],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
    return float(v.median()) if len(v) else 0.0

with st.container(border=True):
    cols = st.columns(3)
    site = {}
    for i,c in enumerate(features):
        with cols[i % 3]:
            if c in cats:
                options = sorted(train_df[c].dropna().astype(str).unique().tolist())
                default = str(default_value(c))
                site[c] = st.selectbox(c,options,index=options.index(default) if default in options else 0,key=f"site_{c}") if options else "Unknown"
            else:
                site[c] = st.number_input(c,value=float(default_value(c)),format="%.4f",key=f"site_{c}")

if st.button("Calculate susceptibility", type="primary", use_container_width=True):
    st.session_state.site_values = site
    st.session_state.site_location = (lat,lon)

site = st.session_state.get("site_values",site)
X = pd.DataFrame([{c:site[c] for c in features}],columns=features)
for c in nums: X[c] = pd.to_numeric(X[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
for c in cats: X[c] = X[c].astype("string").fillna("Unknown").astype(str)
try:
    prob = float(model.predict_proba(X)[0,list(model.classes_).index(1)])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

if prob >= .70:
    band, color, meaning = "Higher relative susceptibility", "🔴", "Prioritise local inspection and consult official information."
elif prob >= .40:
    band, color, meaning = "Moderate relative susceptibility", "🟠", "Review local conditions and official forecasts."
else:
    band, color, meaning = "Lower relative susceptibility", "🟢", "A lower score does not mean zero risk."
st.markdown('<div class="section">2 · Prediction result</div>', unsafe_allow_html=True)
a,b,c = st.columns(3)
a.metric("Event-class model score",f"{prob:.1%}")
b.metric("Screening category",band)
c.metric("Assessment location",f"{lat:.3f}, {lon:.3f}")
st.progress(prob,text="Relative model score (not a calibrated event probability)")
st.info(f"{color} **{band}.** {meaning}")

st.markdown('<div class="section">3 · Rainfall forecast</div>', unsafe_allow_html=True)
try:
    fc = get_forecast(lat,lon,horizon,st.session_state.refresh_id)
    daily = fc.get("daily",{})
    rain_df = pd.DataFrame({
        "Date":daily.get("time",[]),
        "Forecast rainfall (mm/day)":daily.get("precipitation_sum",[]),
        "Rain probability (%)":daily.get("precipitation_probability_max",[])
    })
    rain_df["Forecast rainfall (mm/day)"] = pd.to_numeric(rain_df["Forecast rainfall (mm/day)"],errors="coerce")
    total = float(rain_df["Forecast rainfall (mm/day)"].fillna(0).sum())
    maxday = float(rain_df["Forecast rainfall (mm/day)"].fillna(0).max()) if len(rain_df) else 0
    q1,q2,q3 = st.columns(3)
    q1.metric(f"Total · {horizon} day(s)",f"{total:.1f} mm")
    q2.metric("Maximum in one day",f"{maxday:.1f} mm")
    q3.metric("Forecast timezone",str(fc.get("timezone","local")))
    fig = px.bar(rain_df,x="Date",y="Forecast rainfall (mm/day)",template="plotly_dark",
                 color_discrete_sequence=["#65d6c4"],title="Forecast rainfall")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=55,b=5))
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(rain_df,use_container_width=True,hide_index=True)
    st.caption(f"Forecast retrieved from Open-Meteo · rendered {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    if total >= 100 or maxday >= 50:
        st.warning("High rainfall screening flag under an unvalidated illustrative rule. Check official warnings.")
    elif total >= 50 or maxday >= 25:
        st.warning("Raised rainfall screening flag under an unvalidated illustrative rule. Check official warnings.")
    else:
        st.success("No flag under the illustrative rainfall rule. This does not establish that the site is safe.")
except Exception as e:
    st.warning(f"Forecast unavailable: {e}. Check connection and use the refresh button.")

st.markdown('<div class="section">How to interpret this result</div>', unsafe_allow_html=True)
st.write(
    "The score compares the supplied environmental predictors with patterns learned from historical recorded events and sampled background points. "
    "Background points are not confirmed non-landslide locations, and the score is not a calibrated probability of a future landslide. "
    "The rainfall forecast is shown separately; it is not an input to this classifier."
)
st.write(
    "For real life-safety decisions, use official IMD and disaster-management alerts. An operational warning system needs independent validation, "
    "event-aligned rainfall, false-alarm and missed-event analysis, measured lead time, and approval from responsible authorities."
)

