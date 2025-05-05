import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor, plot_importance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import time

# --- Page config
st.set_page_config(page_title="📈 Smart Time Series & Inventory Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Title
st.title("📦 Smart Time Series Forecasting & Inventory Optimization 🚀")
st.markdown("Welcome! Select a product, choose your models, and forecast into the future! 🌟")

# --- Sidebar
with st.sidebar:
    st.header("🔍 Product & Forecast Setup")

    # Load Data
    file_path = "walmart.csv"
    df = pd.read_csv(file_path)

    # Date Generation
    start_date = pd.to_datetime("2019-01-01")
    df["Date"] = start_date + pd.to_timedelta(np.random.randint(0, 1460, size=len(df)), unit="D")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Feature Engineering
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week

    # Inputs
    product_list = df["Product_ID"].unique().tolist()
    product_id = st.selectbox("🎯 Select Product ID", product_list)
    forecast_models = st.multiselect("📈 Choose Forecast Models", ["ARIMA", "Prophet", "XGBoost", "LSTM"], default=["ARIMA"])
    forecast_days = st.slider("📅 Forecast Horizon (Days)", 10, 90, 30)
    granularity = st.radio("🕒 Forecast Granularity", ["Daily", "Weekly"])
    date_range = st.date_input("📆 Select Training Date Range", [df["Date"].min(), df["Date"].max()])

    # Hyperparameters
    st.subheader("⚙️ Model Hyperparameters")
    with st.expander("🔧 ARIMA"):
        arima_order = (st.slider("p", 0, 5, 5), st.slider("d", 0, 2, 1), st.slider("q", 0, 5, 0))
    with st.expander("🔧 Prophet"):
        prophet_seasonality = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
    with st.expander("🔧 XGBoost"):
        xgb_max_depth = st.slider("Max Depth", 1, 10, 3)
    with st.expander("🔧 LSTM"):
        lstm_units = st.slider("Units", 10, 100, 50)
        lstm_epochs = st.slider("Epochs", 1, 50, 5)

# --- Filter Data
df_product = df[(df["Product_ID"] == product_id) &
                (df["Date"] >= pd.to_datetime(date_range[0])) &
                (df["Date"] <= pd.to_datetime(date_range[1]))]

if df_product.empty:
    st.error("❌ No data for this Product ID and Date Range!")
    st.stop()

# --- Data Aggregation
df_daily = df_product.groupby("Date")["Purchase"].sum().reset_index()
if granularity == "Weekly":
    df_daily = df_daily.set_index("Date").resample("W").sum().fillna(0)
else:
    df_daily = df_daily.set_index("Date").resample("D").sum().fillna(0)

# --- Train-Test Split
train_size = int(len(df_daily) * 0.8)
train, test = df_daily[:train_size], df_daily[train_size:]

# --- Forecasting
results = {}
progress_bar = st.progress(0)
status_text = st.empty()

for i, model_name in enumerate(forecast_models):
    status_text.text(f"Training {model_name} model...")
    time.sleep(0.5)

    if model_name == "ARIMA":
        model = ARIMA(train["Purchase"], order=arima_order)
        result = model.fit()
        forecast = result.forecast(steps=len(test))
        results["ARIMA"] = forecast

    elif model_name == "Prophet":
        prophet_df = train.reset_index().rename(columns={"Date": "ds", "Purchase": "y"})
        model = Prophet(seasonality_mode=prophet_seasonality)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=len(test), freq='D' if granularity == "Daily" else 'W')
        forecast_df = model.predict(future)
        forecast = forecast_df.iloc[-len(test):]["yhat"].values
        results["Prophet"] = forecast

    elif model_name == "XGBoost":
        X_train, y_train = np.arange(len(train)).reshape(-1, 1), train["Purchase"].values
        X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
        model = XGBRegressor(max_depth=xgb_max_depth)
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)
        results["XGBoost"] = forecast

    elif model_name == "LSTM":
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(train["Purchase"].values.reshape(-1, 1))
        X_train_lstm, y_train_lstm = [], []
        for i in range(60, len(scaled_data)):
            X_train_lstm.append(scaled_data[i-60:i, 0])
            y_train_lstm.append(scaled_data[i, 0])
        X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)
        X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))

        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
            LSTM(lstm_units),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train_lstm, y_train_lstm, epochs=lstm_epochs, batch_size=32, verbose=0)

        inputs = df_daily["Purchase"].values[-len(test)-60:]
        inputs = scaler.transform(inputs.reshape(-1, 1))
        X_test_lstm = []
        for i in range(60, len(inputs)):
            X_test_lstm.append(inputs[i-60:i, 0])
        X_test_lstm = np.array(X_test_lstm).reshape((-1, 60, 1))
        forecast = scaler.inverse_transform(model.predict(X_test_lstm)).flatten()
        results["LSTM"] = forecast

    progress_bar.progress((i+1)/len(forecast_models))

st.success('✅ All models trained successfully!')

# --- Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Forecasting", "🏬 Inventory", "📚 Metrics", "📊 Sales Insights", "📥 Download"])

# --- Forecasting Tab
with tab1:
    for model_name, forecast in results.items():
        with st.expander(f"🔮 {model_name} Forecast", expanded=True):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test.index, y=test["Purchase"], name="Actual", line=dict(color='red')))
            fig.add_trace(go.Scatter(x=test.index, y=forecast, name="Forecast", line=dict(color='blue')))

            residuals = test["Purchase"].values - forecast
            anomalies = test.index[np.abs(residuals) > 2*np.std(residuals)]
            for a in anomalies:
                fig.add_vline(x=a, line_width=1, line_dash="dash", line_color="black")

            fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Purchases")
            st.plotly_chart(fig, use_container_width=True)

            st.caption("🔎 *Dashed lines show detected anomalies based on residuals*.")
    st.subheader("🔎 Forecast Error Plot")
    residuals = test["Purchase"].values - forecast
    fig_resid = px.line(x=test.index, y=residuals, labels={'x':'Date', 'y':'Residuals'})
    fig_resid.update_traces(line=dict(color='purple'))
    st.plotly_chart(fig_resid, use_container_width=True)
    
    st.subheader("📊 Customer Analysis")
    age_distribution = df_product["Age"].value_counts().reset_index(name="count").rename(columns={"index": "Age"})
    gender_distribution = df_product["Gender"].value_counts().reset_index(name="count").rename(columns={"index": "Gender"})
    occupation_distribution = df_product["Occupation"].value_counts().reset_index(name="count").rename(columns={"index": "Occupation"})
    
    fig_age = px.bar(age_distribution, x="Age", y="count", title=f"Age Group Preferences for Product {product_id}", text_auto=True)
    fig_gender = px.pie(gender_distribution, names="Gender", values="count", title=f"Gender Preference for Product {product_id}")
    fig_occupation = px.bar(occupation_distribution, x="Occupation", y="count", title=f"Occupation Preference for Product {product_id}", text_auto=True)
    
    st.plotly_chart(fig_age)
    st.plotly_chart(fig_gender)
    st.plotly_chart(fig_occupation)


# --- Inventory Tab
with tab2:
    st.subheader("📦 Inventory Suggestions")
    selected_model = st.selectbox("Select Model for Inventory", list(results.keys()))
    forecast = results[selected_model]

    df_forecast = pd.DataFrame({
        "Date": test.index,
        "Predicted_Demand": forecast,
        "Actual_Demand": test["Purchase"].values
    }).set_index("Date")

    def inventory_suggestion(row):
        deviation = 0.10 * row["Actual_Demand"]
        diff = row["Predicted_Demand"] - row["Actual_Demand"]
        if row["Predicted_Demand"] > row["Actual_Demand"] + deviation:
            return f"Reduce 📉 (-{int(diff)} units)"
        elif row["Predicted_Demand"] < row["Actual_Demand"] - deviation:
            return f"Increase 📈 (+{int(abs(diff))} units)"
        else:
            return "Maintain ✅"

    df_forecast["Inventory_Suggestion"] = df_forecast.apply(inventory_suggestion, axis=1)
    st.dataframe(df_forecast.tail(10))

    safety_stock = st.slider("Safety Stock (%)", 5, 50, 20)
    forecast_period = st.number_input("Forecast Period (days)", min_value=1, max_value=365, value=30)

    recommended_order_quantity = df_forecast["Predicted_Demand"].mean() * (1 + safety_stock/100)
    st.success(f"✅ Recommended Order Quantity: {int(recommended_order_quantity)} units")
    
    adjusted_order = recommended_order_quantity * forecast_period / 30
    st.info(f"📦 Adjusted Order for {forecast_period} days: {int(adjusted_order)} units")

    with st.expander("📈 View Order Quantities Over Time"):
        time_series = pd.date_range(df_forecast.index[-1], periods=forecast_period, freq="D")
        order_series = [recommended_order_quantity] * forecast_period

        fig_order = go.Figure()
        fig_order.add_trace(go.Scatter(x=time_series, y=order_series, mode="lines", line=dict(color='green')))
        fig_order.update_layout(title="Recommended Order Quantities Over Time", xaxis_title="Date", yaxis_title="Order Quantity", template="plotly_white")
        st.plotly_chart(fig_order, use_container_width=True)
    

# --- Metrics Tab
with tab3:
    st.subheader("📚 Model Performance Metrics")
    mae = mean_absolute_error(test["Purchase"], forecast)
    mse = mean_squared_error(test["Purchase"], forecast)
    rmse = np.sqrt(mse)
    r2 = r2_score(test["Purchase"], forecast)

    st.metric("MAE", f"{mae:.2f}")
    st.metric("MSE", f"{mse:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R² Score", f"{r2:.2f}")

with tab4:
        st.subheader("📊 Best-Selling Products")
        # Top 10 Best-Selling Products (Yearly)
        top_products_yearly = df.groupby(["Year", "Product_ID"])["Purchase"].sum().reset_index()
        top_products_yearly = top_products_yearly.sort_values(["Year", "Purchase"], ascending=[True, False])
        top_products_yearly = top_products_yearly.groupby("Year").head(10)
        
        fig_yearly = px.bar(top_products_yearly, x="Product_ID", y="Purchase", color="Year",
                            title="📊 Top 10 Best-Selling Products (Yearly)", text_auto=True)
        st.plotly_chart(fig_yearly)

         # Top 10 Best-Selling Products (Seasonal)
        top_products_season = df.groupby(["Month", "Product_ID"])["Purchase"].sum().reset_index()
        top_products_season = top_products_season.sort_values(["Month", "Purchase"], ascending=[True, False])
        top_products_season = top_products_season.groupby("Month").head(10)
        
        fig_seasonal = px.bar(top_products_season, x="Product_ID", y="Purchase", color="Month",
                            title="🍂📊 Top 10 Best-Selling Products (Seasonal)", text_auto=True)
        st.plotly_chart(fig_seasonal)

        top_products_week = df.groupby(["Week", "Product_ID"])["Purchase"].sum().reset_index()
        top_products_week = top_products_week.sort_values(["Week", "Purchase"], ascending=[True, False])
        top_products_week = top_products_week.groupby("Week").head(1)
        
        fig_week = px.bar(top_products_week, x="Product_ID", y="Purchase", color="Week",
                            title="🍂📊 Top 10 Best-Selling Products (Weekly)", text_auto=True)
        st.plotly_chart(fig_week)

        st.subheader("📅 Seasonal Product Sales (Heatmap)")
        df_season = df.copy()
        df_season["Day"] = df_season["Date"].dt.day
        heatmap = df_season.pivot_table(values="Purchase", index="Month", columns="Day", aggfunc="sum")
        fig_heatmap = px.imshow(heatmap, labels=dict(x="Day", y="Month", color="Purchases"), title="Seasonal Sales Heatmap")
        st.plotly_chart(fig_heatmap, use_container_width=True)

# --- Download Tab
with tab5:
    st.subheader("📥 Download Forecast Files")
    for model_name, forecast in results.items():
        df_download = pd.DataFrame({"Date": test.index, "Forecasted_Purchase": forecast})
        csv = df_download.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.download_button(f"📥 Download {model_name} Forecast CSV", data=csv, file_name=f"{model_name}_forecast.csv", mime="text/csv")

