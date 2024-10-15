import os
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# Load Data
@st.cache_data
def load_data():
    iip_data = pd.read_excel('IIP2024.xlsx')
    stock_data = {}
    stock_data_folder = 'stockdata'
    for filename in os.listdir(stock_data_folder):
        if filename.endswith('.csv'):
            stock_name = filename.replace('.csv', '')
            stock_data[stock_name] = pd.read_csv(os.path.join(stock_data_folder, filename))
    
    # Load correlation results
    correlation_results = pd.read_excel(os.path.join(stock_data_folder, 'Manufacture_of_Food_Products_correlation_results.xlsx'))
    
    # Load financial data
    financial_data = {}
    financial_folder = 'financial'
    for filename in os.listdir(financial_folder):
        if filename.endswith('.xlsx'):
            stock_name = filename.replace('.xlsx', '')
            stock_file_path = os.path.join(financial_folder, filename)
            financial_data[stock_name] = pd.read_excel(stock_file_path, sheet_name=None)
    
    return iip_data, stock_data, correlation_results, financial_data

iip_data, stock_data, correlation_results, financial_data = load_data()

# Define Industry and Indicators
indicators = {
    'Manufacture of Food Products': {
        'Leading': ['Consumer Spending Trends', 'Agricultural Output', 'Retail Sales Data'],
        'Lagging': ['Inventory Levels', 'Employment Data']
    }
}

# Sidebar for selecting between manual input or file upload
st.sidebar.header("Input Options")
data_input_method = st.sidebar.radio("Choose Data Input Method:", ("Manual Input", "Upload Excel with Leading Indicator Data"))

# Upload Excel File for Leading Indicators (May-24 and Jun-24 data)
if data_input_method == "Upload Excel with Leading Indicator Data":
    uploaded_file = st.sidebar.file_uploader("Upload Excel file with Expected Consumer Spending, Agricultural Output, and Retail Sales Data", type="xlsx")
    if uploaded_file:
        uploaded_data = pd.read_excel(uploaded_file)

# Select Industry from Sidebar
selected_industry = st.sidebar.selectbox(
    'Select Industry',
    list(indicators.keys()),  # Industry options
    index=0  # Default selection
)

# Prepare Data for Modeling based on selected industry
if selected_industry:
    st.header(f'Industry: {selected_industry}')
    
    # Check if data is uploaded or manual input
    if data_input_method == "Upload Excel with Leading Indicator Data" and uploaded_file:
        if 'Date' in uploaded_data.columns and all(col in uploaded_data.columns for col in indicators[selected_industry]['Leading']):
            # Use uploaded data for predictions (May-24, Jun-24)
            X = uploaded_data[indicators[selected_industry]['Leading']].dropna()
            st.write("**Uploaded Leading Indicator Data**")
            st.write(uploaded_data)

            y = iip_data[selected_industry].iloc[:len(X)]  # Match the length of X with the iip_data
        else:
            st.error("Uploaded file does not have the required columns or Date field.")
            X = None
            y = None
    else:
        # Manual Input for the data
        st.subheader("Manual Input for Leading Indicators")
        input_data = {}
        for indicator in indicators[selected_industry]['Leading']:
            input_data[indicator] = st.number_input(f'Expected {indicator} Value:', value=100.0)
        
        X = pd.DataFrame(input_data, index=[0])
        y = pd.Series([100.0])  # Placeholder for industry data
    
    if X is not None and y is not None:
        # Train models
        reg_model = LinearRegression()
        reg_model.fit(X, y)
        reg_pred = reg_model.predict(X)

        arima_model = ARIMA(y, order=(5, 1, 0))
        arima_result = arima_model.fit()
        arima_pred = arima_result.predict(start=1, end=len(y), dynamic=False)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(X)

        # Visualize Predictions
        st.subheader('Model Performance')
        st.write(f"Linear Regression RMSE: {mean_squared_error(y, reg_pred, squared=False):.2f}")
        st.write(f"ARIMA RMSE: {mean_squared_error(y, arima_pred, squared=False):.2f}")
        st.write(f"Random Forest RMSE: {mean_squared_error(y, rf_pred, squared=False):.2f}")

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y.index, y=y, mode='lines', name='Actual Industry Data'))
        fig.add_trace(go.Scatter(x=y.index, y=reg_pred, mode='lines', name='Linear Regression Prediction'))
        fig.add_trace(go.Scatter(x=y.index, y=arima_pred, mode='lines', name='ARIMA Prediction'))
        fig.add_trace(go.Scatter(x=y.index, y=rf_pred, mode='lines', name='Random Forest Prediction'))

        fig.update_layout(
            title=f'Industry Data Prediction for {selected_industry}',
            xaxis_title='Date',
            yaxis_title='Industry Data',
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Predict Future Values
        st.subheader('Predict Future Values')

        if data_input_method == "Manual Input":
            input_data = {indicator: st.number_input(f'Expected {indicator} Value:', value=float(X[indicator].iloc[-1]) if not X.empty else 100.0) for indicator in indicators[selected_industry]['Leading']}
            future_df = pd.DataFrame(input_data, index=[0])
        else:
            # Use uploaded data for prediction
            future_df = pd.DataFrame(uploaded_data[indicators[selected_industry]['Leading']].iloc[-1:])

        # Predictions
        future_reg_pred = reg_model.predict(future_df)
        future_rf_pred = rf_model.predict(future_df)

        st.write(f"Linear Regression Prediction: {future_reg_pred[0]:.2f}")
        st.write(f"Random Forest Prediction: {future_rf_pred[0]:.2f}")

        # Stock Selection and Correlation Analysis
        st.sidebar.subheader('Stock Selection')
        stock_name = st.sidebar.selectbox('Select Stock for Financial Data', list(financial_data.keys()), index=0)

        if stock_name:
            stock_financial_data = financial_data[stock_name].get('IncomeStatement', pd.DataFrame())
            
            if not stock_financial_data.empty:
                latest_income_statement = stock_financial_data[stock_financial_data['Date'] == 'Jun 2024'].iloc[-1]
                st.subheader(f"Latest Financial Data for {stock_name}")
                st.write("**Income Statement (Jun 2024):**")
                st.write(latest_income_statement)

                # Predicted Income Statement Results
                st.subheader(f"Predicted Income Statement Results for {stock_name}")
                predicted_income_statement = latest_income_statement.copy()

                for column in latest_income_statement.index:
                    if column != 'Date':
                        predicted_income_statement[column] = latest_income_statement[column] * future_reg_pred[0] / y.mean()

                # Display predicted results
                st.write(predicted_income_statement)
