# app.py
# --------------------------------------------
# Smart City – Traffic Volume Forecasting Demo
# Streamlit dashboard (simple version)
# --------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Page config
st.set_page_config(
    page_title="Smart Traffic Forecasting",
    layout="centered"
)

st.title("🚦 Smart City – Traffic Volume Forecasting")
st.write(
    "This app trains a simple machine learning model on real traffic data "
    "and lets you predict traffic volume for different conditions."
)

# 2. Load & preprocess data (cached)
@st.cache_data
def load_data():
    df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

    df["date_time"] = pd.to_datetime(df["date_time"])

    # Time features
    df["hour"] = df["date_time"].dt.hour
    df["weekday"] = df["date_time"].dt.weekday
    df["month"] = df["date_time"].dt.month
    df["year"] = df["date_time"].dt.year

    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_holiday"] = (df["holiday"] != "None").astype(int)

    # One-hot for main weather
    weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather")
    df = pd.concat([df, weather_dummies], axis=1)

    return df


@st.cache_resource
def train_model(df):
    base_features = [
        "temp", "rain_1h", "snow_1h", "clouds_all",
        "hour", "weekday", "month", "year",
        "is_rush_hour", "is_weekend", "is_holiday",
    ]
    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    features = base_features + weather_cols
    target = "traffic_volume"

    model_df = df[features + [target]].dropna()
    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=120,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate once, return metrics
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    return model, scaler, features, weather_cols, metrics


with st.spinner("Loading data and training model (first time only)..."):
    df = load_data()
    model, scaler, FEATURES, WEATHER_COLS, metrics = train_model(df)

st.success("Model ready!")

# Show metrics
st.subheader("📊 Model Performance")
st.write(f"**MAE:** {metrics['mae']:.2f} vehicles/hour")
st.write(f"**RMSE:** {metrics['rmse']:.2f} vehicles/hour")
st.write(f"**R²:** {metrics['r2']:.4f}")

st.write("---")

# Sidebar inputs
st.sidebar.header("Input Conditions")

temp = st.sidebar.slider("Temperature (°C)", -20.0, 40.0, 10.0, 0.5)
rain_1h = st.sidebar.slider("Rain in last 1h (mm)", 0.0, 20.0, 0.0, 0.1)
snow_1h = st.sidebar.slider("Snow in last 1h (mm)", 0.0, 20.0, 0.0, 0.1)
clouds_all = st.sidebar.slider("Cloud cover (%)", 0, 100, 40, 5)
hour = st.sidebar.slider("Hour of day (0–23)", 0, 23, 8, 1)
weekday = st.sidebar.selectbox(
    "Weekday (0=Mon ... 6=Sun)",
    options=list(range(7)),
    index=2
)
month = st.sidebar.slider("Month", 1, 12, 3, 1)
year = st.sidebar.slider("Year", 2012, 2018, 2013, 1)
is_holiday = st.sidebar.selectbox("Holiday?", options=["No", "Yes"])
is_holiday_flag = 1 if is_holiday == "Yes" else 0

weather_options = ["None"]
if WEATHER_COLS:
    weather_options = [c.replace("weather_", "") for c in WEATHER_COLS]
weather_main = st.sidebar.selectbox("Weather condition", options=weather_options)

# Build input row
input_row = {}

input_row["temp"] = temp
input_row["rain_1h"] = rain_1h
input_row["snow_1h"] = snow_1h
input_row["clouds_all"] = clouds_all
input_row["hour"] = hour
input_row["weekday"] = weekday
input_row["month"] = month
input_row["year"] = year
input_row["is_rush_hour"] = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
input_row["is_weekend"] = 1 if weekday in [5, 6] else 0
input_row["is_holiday"] = is_holiday_flag

for col in WEATHER_COLS:
    input_row[col] = 0
if WEATHER_COLS and weather_main != "None":
    chosen_col = "weather_" + weather_main
    if chosen_col in WEATHER_COLS:
        input_row[chosen_col] = 1

input_df = pd.DataFrame([[input_row[col] for col in FEATURES]], columns=FEATURES)

if st.button("Predict Traffic Volume"):
    X_scaled = scaler.transform(input_df)
    pred = model.predict(X_scaled)[0]

    if pred < 3000:
        congestion = "LOW"
    elif pred < 6000:
        congestion = "MEDIUM"
    else:
        congestion = "HIGH"

    st.subheader("🔮 Prediction")
    st.write(f"**Predicted Traffic Volume:** {pred:.0f} vehicles/hour")
    st.write(f"**Estimated Congestion Level:** `{congestion}`")

    st.write("### Input Summary")
    st.json({
        "temp_C": temp,
        "rain_1h_mm": rain_1h,
        "snow_1h_mm": snow_1h,
        "clouds_all_%": clouds_all,
        "hour": hour,
        "weekday(0=Mon)": weekday,
        "month": month,
        "year": year,
        "is_holiday": is_holiday,
        "weather_main": weather_main
    })
else:
    st.info("Set conditions in the sidebar and click **Predict Traffic Volume**.")
