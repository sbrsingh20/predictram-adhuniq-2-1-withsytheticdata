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
    iip_data = pd.read_excel('IIP2024.xlsx')  # Existing IIP Data
    stock_data = {}
    stock_data_folder = 'stockdata'
    for filename in os.listdir(stock_data_folder):
        if filename.endswith('.csv'):
            stock_name = filename.replace('.csv', '')
            stock_data[stock_name] = pd.read_csv(os.path.join(stock_data_folder, filename))
    correlation_results = pd.read_excel(os.path.join(stock_data_folder, 'Manufacture_of_Food_Products_correlation_results.xlsx'))  # Correlation data
    financial_data = {}
    financial_folder = 'financial'
    for filename in os.listdir(financial_folder):
        if filename.endswith('.xlsx'):
            stock_name = filename.replace('.xlsx', '')
            financial_data[stock_name] = pd.read_excel(os.path.join(financial_folder, filename), sheet_name=None)
    return iip_data, stock_data, correlation_results, financial_data

iip_data, stock_data, correlation_results, financial_data = load_data()

# Sidebar for selecting between manual input or file upload
data_input_method = st.sidebar.radio("Choose Data Input Method:", ("Manual Input", "Upload Synthetic Data"))

if data_input_method == "Upload Synthetic Data":
    # File Upload for Synthetic Data
    uploaded_file = st.sidebar.file_uploader("Upload Synthetic Data", type="xlsx")
    if uploaded_file:
        synthetic_data = pd.read_excel(uploaded_file, sheet_name=None)
else:
    # Manual Input
    synthetic_data = None

# Define Industry and Indicators (Existing structure)
indicators = {
    'Manufacture of Food Products': {
        'Leading': ['Consumer Spending Trends', 'Agricultural Output', 'Retail Sales Data'],
        'Lagging': ['Inventory Levels', 'Employment Data']
    }
    # You can add more industries here
}

# Select Industry from Sidebar
selected_industry = st.sidebar.selectbox(
    'Select Industry',
    list(indicators.keys()),  # Industry options
    index=0  # Default selection
)

if selected_industry:
    # Normalize and match industry names between uploaded file and the indicators
    normalized_industry = selected_industry.strip().lower()
    matched_sheet_name = None
    
    if synthetic_data:
        for sheet_name in synthetic_data.keys():
            if sheet_name.strip().lower() == normalized_industry:
                matched_sheet_name = sheet_name
                break
    
    st.header(f'Industry: {selected_industry}')
    
    if matched_sheet_name and synthetic_data:
        # Use uploaded synthetic data
        def prepare_data(industry, data, iip_data):
            leading_indicators = indicators[industry]['Leading']
            X = data[leading_indicators].shift(1).dropna()
            y = iip_data[industry].loc[X.index]
            return X, y

        X, y = prepare_data(selected_industry, synthetic_data[matched_sheet_name], iip_data)
        
    else:
        # Use manual input for data
        input_data = {}
        for indicator in indicators[selected_industry]['Leading']:
            input_data[indicator] = st.number_input(f'Expected {indicator} Value:', value=100.0)
        input_df = pd.DataFrame(input_data, index=[0])
        X = input_df  # Manually entered input data for prediction
        y = pd.Series([100.0])  # Placeholder for industry data

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

    # Show predictions
    st.subheader('Model Performance')
    st.write(f"Linear Regression RMSE: {mean_squared_error(y, reg_pred, squared=False):.2f}")
    st.write(f"ARIMA RMSE: {mean_squared_error(y, arima_pred, squared=False):.2f}")
    st.write(f"Random Forest RMSE: {mean_squared_error(y, rf_pred, squared=False):.2f}")

        # Visualize Predictions
    fig.update_layout(
        title=f'Industry Data Prediction for {selected_industry}',
        xaxis_title='Date',
        yaxis_title='Industry Data',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Predict Future Values based on manual input or uploaded data
    st.subheader('Predict Future Values')

    if data_input_method == "Manual Input":
        input_data = {}
        for indicator in indicators[selected_industry]['Leading']:
            input_data[indicator] = st.number_input(f'Expected {indicator} Value:', value=float(X[indicator].iloc[-1]) if not X.empty else 100.0)
        input_df = pd.DataFrame(input_data, index=[0])
    else:
        # If synthetic data is uploaded
        input_df = pd.DataFrame({
            indicator: synthetic_data[matched_sheet_name][indicator].iloc[-1] for indicator in indicators[selected_industry]['Leading']
        }, index=[0])

    # Predict future values using models
    future_reg_pred = reg_model.predict(input_df)
    future_rf_pred = rf_model.predict(input_df)

    st.write(f"Linear Regression Prediction: {future_reg_pred[0]:.2f}")
    st.write(f"Random Forest Prediction: {future_rf_pred[0]:.2f}")

    # Select stock for correlation and financial data analysis
    st.sidebar.subheader('Stock Selection')
    stock_name = st.sidebar.selectbox('Select Stock for Financial Data', list(financial_data.keys()), index=0)

    # Get latest financial data for selected stock
    if stock_name:
        stock_financial_data = financial_data[stock_name].get('IncomeStatement', pd.DataFrame())

        if not stock_financial_data.empty:
            # Display the latest income statement
            latest_income_statement = stock_financial_data[stock_financial_data['Date'] == 'Jun 2024'].iloc[-1]
            st.subheader(f"Latest Financial Data for {stock_name}")
            st.write("**Income Statement (Jun 2024):**")
            st.write(latest_income_statement)

            # Predict Income Statement Results
            st.subheader(f"Predicted Income Statement Results for {stock_name}")
            predicted_income_statement = latest_income_statement.copy()

            for column in latest_income_statement.index:
                if column != 'Date':
                    predicted_income_statement[column] = latest_income_statement[column] * future_reg_pred[0] / y.mean()

            # Display predicted results
            st.write(predicted_income_statement)

            # Correlation Analysis and Prediction
            st.subheader(f"Correlation Analysis for {stock_name}")
            if stock_name in correlation_results['Stock Name'].values:
                stock_corr_data = correlation_results[correlation_results['Stock Name'] == stock_name]
                st.write(stock_corr_data)

                # Adjust correlation based on predicted industry value
                industry_mean = y.mean()
                for col in stock_corr_data.columns:
                    if 'correlation' in col:
                        stock_corr_data[f'Adjusted {col}'] = stock_corr_data[col] * (future_reg_pred[0] / industry_mean)

                st.write('**Adjusted Correlation Results:**')
                st.write(stock_corr_data)

                # Visualization of actual and predicted correlation results
                fig_corr = go.Figure()

                actual_corr_cols = [col for col in stock_corr_data.columns if 'correlation' in col]
                predicted_corr_cols = [f'Adjusted {col}' for col in actual_corr_cols]

                for col in actual_corr_cols:
                    fig_corr.add_trace(go.Bar(x=stock_corr_data['Stock Name'], y=stock_corr_data[col], name=f'Actual {col}', marker_color='blue'))

                for col in predicted_corr_cols:
                    fig_corr.add_trace(go.Bar(x=stock_corr_data['Stock Name'], y=stock_corr_data[col], name=f'Predicted {col}', marker_color='orange'))

                fig_corr.update_layout(
                    title='Comparison of Actual and Predicted Correlation Results',
                    xaxis_title='Stock',
                    yaxis_title='Correlation Value',
                    barmode='group',
                    hovermode='x unified'
                )

                st.plotly_chart(fig_corr, use_container_width=True)

