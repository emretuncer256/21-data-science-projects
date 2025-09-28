import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
import joblib
from datetime import timedelta

# Configure warnings and plotting
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style('darkgrid')

# Page configuration
st.set_page_config(
    page_title="Website Traffic Forecasting",
    page_icon="📈",
    layout="wide"
)

# Load data and model
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Thecleverprogrammer.csv')
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        return df
    except FileNotFoundError:
        st.error("Data file 'Thecleverprogrammer.csv' not found. Please ensure the file is in the same directory.")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('website_traffic_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'website_traffic_model.pkl' not found. Please run the notebook first to generate the model.")
        return None

# Main app
st.title("📈 Website Traffic Forecasting Dashboard")
st.markdown("Predict website traffic patterns using SARIMA time series analysis")

# Load data and model
df = load_data()
model = load_model()

if df is not None and model is not None:
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Prediction days slider
    prediction_days = st.sidebar.slider(
        "Number of days to predict:",
        min_value=7,
        max_value=120,
        value=60,
        step=7
    )
    
    # Show data info
    st.sidebar.subheader("Dataset Info")
    st.sidebar.write(f"Total records: {len(df)}")
    st.sidebar.write(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    st.sidebar.write(f"Average daily views: {df['Views'].mean():.0f}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔍 Analysis", "🔮 Predictions", "📈 Model Performance"])
    
    with tab1:
        st.header("Data Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Views", f"{df['Views'].sum():,}")
        with col2:
            st.metric("Average Daily Views", f"{df['Views'].mean():.0f}")
        with col3:
            st.metric("Peak Views", f"{df['Views'].max():,}")
        with col4:
            st.metric("Min Views", f"{df['Views'].min():,}")
        
        # Time series plot
        fig = px.line(df, x='Date', y='Views', title='Website Traffic Over Time')
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Views"
        )
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
        
        # Data table
        st.subheader("Raw Data")
        st.dataframe(df.tail(10), width='stretch')
    
    with tab2:
        st.header("Time Series Analysis")
        
        # Seasonal decomposition
        st.subheader("Seasonal Decomposition")
        result = seasonal_decompose(df['Views'], model='multiplicative', period=30)
        
        # Create subplots for decomposition
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08
        )
        
        fig.add_trace(go.Scatter(x=df['Date'], y=result.observed, name='Original'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=result.trend, name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=result.seasonal, name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=result.resid, name='Residual'), row=4, col=1)
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, width='stretch')
        
        # Autocorrelation analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Autocorrelation Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            pd.plotting.autocorrelation_plot(df['Views'], ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Partial Autocorrelation Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_pacf(df['Views'], lags=50, ax=ax)
            st.pyplot(fig)
    
    with tab3:
        st.header("Traffic Predictions")
        
        # Generate predictions
        predictions = model.predict(len(df), len(df) + prediction_days - 1)
        
        # Create future dates
        last_date = df['Date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days, freq='D')
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Views': predictions
        })
        
        # Combined plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Views'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue', width=2)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=pred_df['Date'],
            y=pred_df['Predicted_Views'],
            mode='lines',
            name='Predictions',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Website Traffic Forecast',
            xaxis_title='Date',
            yaxis_title='Views',
            height=500
        )
        
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
        
        # Prediction summary
        st.subheader("Prediction Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Predicted Views", f"{predictions.mean():.0f}")
        with col2:
            st.metric("Max Predicted Views", f"{predictions.max():.0f}")
        with col3:
            st.metric("Min Predicted Views", f"{predictions.min():.0f}")
        
        # Prediction table
        st.subheader("Detailed Predictions")
        pred_df['Date'] = pred_df['Date'].dt.strftime('%Y-%m-%d')
        pred_df['Predicted_Views'] = pred_df['Predicted_Views'].round(0).astype(int)
        st.dataframe(pred_df, width='stretch')
    
    with tab4:
        st.header("Model Performance")
        
        # Model summary
        st.subheader("SARIMA Model Summary")
        st.text(str(model.summary()))
        
        # Model parameters
        st.subheader("Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Order (p,d,q):** (5,1,2)")
            st.write("**Seasonal Order (P,D,Q,s):** (5,1,2,12)")
        
        with col2:
            st.write("**AIC:** {:.2f}".format(model.aic))
            st.write("**BIC:** {:.2f}".format(model.bic))
        
        # Residuals analysis
        st.subheader("Residuals Analysis")
        residuals = model.resid
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            ax.set_title('Residuals Distribution')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(range(len(residuals)), residuals, alpha=0.6)
            ax.set_title('Residuals vs Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Residuals')
            ax.axhline(y=0, color='red', linestyle='--')
            st.pyplot(fig)

else:
    st.error("Please ensure both the data file and model file are available to run the application.")
