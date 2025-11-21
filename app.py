import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# SMART CITY – SMART TRAFFIC STREAMLIT APP
# Everything (data + training + dashboard) in one file
# ============================================

st.set_page_config(page_title="Smart Traffic Forecasting", page_icon="🚦")

st.title("🚦 Smart City – Traffic Volume Forecasting")
st.write("""
This app uses a Machine Learning model (Random Forest) to predict **traffic volume (vehicles/hour)** 
based on weather and time features, simulating a Smart City traffic management scenario.
""")

# -----------------------------
# 1. Load & preprocess data
# -----------------------------
@st.cache_data
def load_and_prepare_data():
    data_url = "https://raw.githubusercontent.com/selva86/datasets/master/Metro_Interstate_Traffic_Volume.csv"
    df = pd.read_csv(data_url)

    # Datetime
    df["date_time"] = pd.to_datetime(df["date_time"])

    # Time features
    df["hour"] = df["date_time"].dt.hour
    df["weekday"] = df["date_time"].dt.weekday
    df["month"] = df["date_time"].dt.month
    df["year"] = df["date_time"].dt.year

    # Flags
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_holiday"] = (df.get("holiday", "None") != "None").astype(int)

    # Weather one-hot
    if "weather_main" in df.columns:
        weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather")
        df = pd.concat([df, weather_dummies], axis=1)
    else:
        weather_dummies = pd.DataFrame()

    base_features = [
        "temp", "rain_1h", "snow_1h", "clouds_all",
        "hour", "weekday", "month", "year",
        "is_rush_hour", "is_weekend", "is_holiday"
    ]
    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    features = base_features + weather_cols

    # Drop NaNs
    model_df = df[features + ["traffic_volume"]].dropna()
    X = model_df[features]
    y = model_df["traffic_volume"]

    # Weather labels for UI
    weather_labels = [c.replace("weather_", "") for c in weather_cols]
    if not weather_labels:
        weather_labels = ["None"]

    return df, X, y, features, weather_cols, weather_labels

# -----------------------------
# 2. Train model (cached)
# -----------------------------
@st.cache_resource
def train_model():
    df, X, y, features, weather_cols, weather_labels = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)

    y_pred = rf.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

    return rf, scaler, features, weather_cols, weather_labels, metrics

rf, scaler, features, weather_cols, weather_labels, metrics = train_model()

# -----------------------------
# 3. Show model metrics
# -----------------------------
st.subheader("📊 Model Performance (Validation Set)")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{metrics['MAE']:.0f}")
col2.metric("RMSE", f"{metrics['RMSE']:.0f}")
col3.metric("R²", f"{metrics['R2']:.3f}")

st.caption("MAE/RMSE are in vehicles/hour.")

# -----------------------------
# 4. Sidebar controls (inputs)
# -----------------------------
st.sidebar.header("Input Conditions")

temp = st.sidebar.slider("Temperature (°C)", -20.0, 40.0, 10.0, 0.5)
rain_1h = st.sidebar.slider("Rain in last 1h (mm)", 0.0, 20.0, 0.0, 0.1)
snow_1h = st.sidebar.slider("Snow in last 1h (mm)", 0.0, 20.0, 0.0, 0.1)
clouds_all = st.sidebar.slider("Cloud Cover (%)", 0, 100, 40, 5)

hour = st.sidebar.slider("Hour of Day (0–23)", 0, 23, 8, 1)
weekday = st.sidebar.slider("Weekday (0=Mon ... 6=Sun)", 0, 6, 2, 1)
month = st.sidebar.slider("Month (1–12)", 1, 12, 3, 1)
year = st.sidebar.slider("Year", 2012, 2018, 2013, 1)

holiday = st.sidebar.radio("Is it a holiday?", ["No", "Yes"])
is_holiday = 1 if holiday == "Yes" else 0

weather_main = st.sidebar.selectbox("Weather Condition", weather_labels)

st.sidebar.markdown("---")
st.sidebar.write("Adjust the sliders and dropdown to change conditions, then click **Predict** on the main page.")

# -----------------------------
# 5. Build input row
# -----------------------------
def build_input_dataframe():
    row = {
        "temp": temp,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "clouds_all": clouds_all,
        "hour": hour,
        "weekday": weekday,
        "month": month,
        "year": year,
        "is_rush_hour": 1 if hour in [7, 8, 9, 16, 17, 18] else 0,
        "is_weekend": 1 if weekday in [5, 6] else 0,
        "is_holiday": is_holiday,
    }

    for col in weather_cols:
        row[col] = 0

    if weather_main != "None":
        chosen_col = "weather_" + weather_main
        if chosen_col in weather_cols:
            row[chosen_col] = 1

    input_df = pd.DataFrame([[row[col] for col in features]], columns=features)
    return input_df, row

input_df, raw_row = build_input_dataframe()

# -----------------------------
# 6. Prediction button
# -----------------------------
st.subheader("🔮 Predict Traffic Volume")

if st.button("Predict"):
    X_scaled = scaler.transform(input_df)
    pred = rf.predict(X_scaled)[0]

    if pred < 3000:
        congestion = "LOW"
        color = "🟢"
    elif pred < 6000:
        congestion = "MEDIUM"
        color = "🟡"
    else:
        congestion = "HIGH"
        color = "🔴"

    st.markdown(f"**Predicted Traffic Volume:** `{int(pred):,}` vehicles/hour")
    st.markdown(f"**Estimated Congestion Level:** {color} **{congestion}**")

    with st.expander("See input details"):
        st.json(raw_row)

    chart_df = pd.DataFrame({
        "Category": ["Predicted Volume"],
        "Vehicles/hour": [pred]
    }).set_index("Category")
    st.bar_chart(chart_df)
else:
    st.info("Set your conditions in the sidebar, then click **Predict**.")
