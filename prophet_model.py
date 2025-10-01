import pandas as pd
from prophet import Prophet  # Cambiado de 'fbprophet' a 'prophet'
from prophet.make_holidays import make_holidays_df  # Cambiado de 'fbprophet' a 'prophet'
import plotly.graph_objects as go

def run_prophet_model(df: pd.DataFrame, series_name: str, months_ahead: int = 6):
    """
    Run Prophet model with holidays added for USA and forecast for the next 'months_ahead' months.
    """
    # Find the column name that represents the date, it could be 'date', 'Week', 'fecha', etc.
    date_column = None
    for col in df.columns:
        if 'date' in col.lower() or 'week' in col.lower() or 'fecha' in col.lower():
            date_column = col
            break
    
    if date_column is None:
        raise ValueError("No date column found in the dataset. Please ensure it is named 'date', 'Week', or similar.")
    
    # Prepare data for Prophet
    df_prophet = df.reset_index()[[date_column, series_name]].rename(columns={date_column: 'ds', series_name: 'y'})

    # Create a holidays dataframe (USA holidays)
    holidays = make_holidays_df(year_list=[df['ds'].dt.year.min(), df['ds'].dt.year.max()], country='US')

    # Initialize and fit the Prophet model
    model = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_prophet)

    # Forecast the next months_ahead months
    future = model.make_future_dataframe(df_prophet, periods=months_ahead * 30)  # Approx. 30 days per month
    forecast = model.predict(future)

    # Plotting
    fig = go.Figure()

    # Actual data
    fig.add_trace(go.Scatter(x=df['ds'], y=df[series_name], name='Actual Data', mode='lines'))

    # Forecasted data
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Data', mode='lines'))

    # Plot the forecast
    fig.update_layout(title=f"Prophet Forecast — {series_name}", xaxis_title="Date", yaxis_title="Trend Value")

    return forecast, fig

def run_prophet_forecast(df: pd.DataFrame, series_name: str):
    """Ejecuta el modelo Prophet y muestra las predicciones para los próximos 6 meses."""
    if df.empty:
        st.warning("No data available for Prophet model. Please verify the file or keyword.")
        st.stop()

    if series_name not in df.columns:
        st.error(f"Column '{series_name}' not found in the dataset.")
        st.stop()

    with st.spinner("Running Prophet model…"):
        forecast, fig = run_prophet_model(df, series_name, months_ahead=6)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Predicciones del modelo Prophet para los próximos 6 meses:")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], use_container_width=True)

    st.download_button(
        "Download Prophet Forecast CSV",
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode("utf-8"),
        file_name=f"prophet_forecast_{series_name.strip().replace(' ','_')}.csv",
        mime="text/csv"
    )
