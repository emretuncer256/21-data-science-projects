import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import warnings

warnings.filterwarnings("ignore")


@st.cache_data(show_spinner=False)
def load_data(default_path: str | None = None) -> pd.DataFrame:
    if default_path is None:
        default_path = "DailyDelhiClimateTrain.csv"
    try:
        df = pd.read_csv(default_path)
        return df
    except Exception:
        return pd.DataFrame()


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def main() -> None:
    st.set_page_config(page_title="Delhi Weather Forecast", layout="wide")
    st.title("Weather Forecasting with Prophet")
    st.write(
        "This app visualizes the Delhi climate dataset and produces a simple temperature forecast using Prophet."
    )

    with st.sidebar:
        st.header("Forecast Settings")
        horizon_days = st.slider("Forecast horizon (days)", min_value=30, max_value=730, value=365, step=30)
        run_forecast = st.button("Run Forecast")

    df = load_data()

    if df.empty:
        st.warning("No data found. Please ensure 'DailyDelhiClimateTrain.csv' is in this folder.")
        st.stop()

    df = prepare_dataframe(df)

    # Basic info
    st.subheader("Sample and Summary")
    st.dataframe(df.head(), width='stretch')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.write("")

    # Visualization
    st.subheader("Exploratory Plots")
    if "date" in df.columns:
        numeric_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            y_col = st.selectbox("Select variable to visualize", options=numeric_cols, index=0)
            fig = px.line(df.sort_values("date"), x="date", y=y_col, title=f"{y_col} over time")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No numeric columns available to plot.")
    else:
        st.info("No 'date' column found for time series visualization.")

    # Forecasting
    st.subheader("Temperature Forecast (Prophet)")
    required_cols = {"date", "meantemp"}
    if required_cols.issubset(df.columns):
        if run_forecast:
            train_df = (
                df[["date", "meantemp"]]
                .rename(columns={"date": "ds", "meantemp": "y"})
                .dropna()
            )

            try:
                model = Prophet()
                model.fit(train_df)

                future = model.make_future_dataframe(periods=int(horizon_days))
                forecast = model.predict(future)

                st.markdown("**Forecast Plot**")
                forecast_fig = plot_plotly(model, forecast)
                # Recolor marker dots to a bright color
                for trace in forecast_fig.data:
                    mode = getattr(trace, "mode", "") or ""
                    if "markers" in mode:
                        trace.marker.update(color="green", size=5, opacity=0.7)
                st.plotly_chart(forecast_fig, width='stretch')

                with st.expander("Show forecast data"):
                    st.dataframe(forecast.tail(20), width='stretch')

            except Exception as e:
                st.error(f"Forecasting failed: {e}")
        else:
            st.info("Set the horizon and click 'Run Forecast' to generate predictions.")
    else:
        st.warning(
            "The dataset must include 'date' and 'meantemp' columns to run the temperature forecast."
        )


if __name__ == "__main__":
    main()