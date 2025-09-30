import streamlit as st
import warnings
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px
import plotly.graph_objects as go


warnings.filterwarnings('ignore')


st.title('Simple Stock Price Forecasting (SARIMA)')
st.write('SARIMA model to forecast stock closing prices using Streamlit.')


# Inputs
default_end = date.today()
default_start = default_end - timedelta(days=365)

with st.sidebar:
    ticker = st.text_input('Ticker', value='GOOGL')
    start_date = st.date_input('Start date', value=default_start)
    end_date = st.date_input('End date', value=default_end)
    n_days = st.slider('Forecast horizon (days)', min_value=1, max_value=365, value=60, step=1)
    run = st.button('Train model', type='primary', use_container_width=True)


def load_prices(ticker_symbol: str, start_d: date, end_d: date) -> pd.DataFrame:
    df_raw = yf.download(ticker_symbol, start=start_d.strftime('%Y-%m-%d'), end=end_d.strftime('%Y-%m-%d'), progress=False)
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    # Always add Date from index, then reduce to Date and Close
    if isinstance(df_raw.columns, pd.MultiIndex):
        try:
            # Prefer ('Close', ticker)
            close_series = df_raw['Close'][ticker_symbol]
        except Exception:
            # Fallback to the first available close column
            close_series = df_raw['Close'].iloc[:, 0]
        df = pd.DataFrame({'Date': df_raw.index, 'Close': close_series.values})
    else:
        df = pd.DataFrame({'Date': df_raw.index, 'Close': df_raw['Close'].values})

    df = df.reset_index(drop=True)
    return df


def train_sarima(series: pd.Series):
    p, d, q = 5, 1, 2
    model = SARIMAX(series, order=(p, d, q), seasonal_order=(p, d, q, 12))
    model_fit = model.fit(maxiter=200, disp=False)
    return model_fit


def forecast_with_model(model_fit, last_train_date: pd.Timestamp, horizon_days: int) -> pd.Series:
    fc = model_fit.get_forecast(steps=horizon_days)
    preds = fc.predicted_mean
    future_index = pd.date_range(pd.to_datetime(last_train_date) + pd.Timedelta(days=1), periods=len(preds), freq='D')
    preds.index = future_index
    return preds


if run:
    if not ticker:
        st.error('Please enter a ticker symbol.')
    elif start_date >= end_date:
        st.error('Start date must be before end date.')
    else:
        with st.spinner('Loading data...'):
            df = load_prices(ticker, start_date, end_date)
        if df.empty:
            st.error('No data returned. Check the ticker and dates.')
        else:
            st.success(f'Loaded {len(df)} rows for {ticker}.')

            # Persist training data and model
            st.session_state['df'] = df
            st.session_state['last_date'] = pd.to_datetime(df['Date'].iloc[-1])

            # Plot historical prices (Plotly)
            fig_hist = px.line(df, x='Date', y='Close', title=f'{ticker} Close Price')
            st.plotly_chart(fig_hist, use_container_width=True)

            # Train model only
            with st.spinner('Training SARIMA model...'):
                model_fit = train_sarima(df['Close'])
            st.session_state['model_fit'] = model_fit
            st.session_state['ticker'] = ticker

# If a model is available, always compute and display forecast based on the sidebar slider
if 'model_fit' in st.session_state and 'df' in st.session_state:
    df = st.session_state['df']
    ticker = st.session_state.get('ticker', ticker)
    preds = forecast_with_model(st.session_state['model_fit'], st.session_state['last_date'], int(n_days))

    # Plot training + forecast (Plotly)
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Training Data'))
    fig_forecast.add_trace(go.Scatter(x=preds.index, y=preds.values, mode='lines', name='Forecast'))
    fig_forecast.update_layout(title=f'{ticker} Forecast (next {len(preds)} days)', xaxis_title='Date', yaxis_title='Close')
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader('Forecasted values')
    st.dataframe(pd.DataFrame({'Date': preds.index, 'Predicted Close': preds.values}).set_index('Date'))

    st.caption('Model: SARIMAX with order (5,1,2) and seasonal order (5,1,2,12). Changing the slider updates the forecast without retraining.')
else:
    st.info('Set inputs and click "Train model". Use the sidebar slider to change horizon.')