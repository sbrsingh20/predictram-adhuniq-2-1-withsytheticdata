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
    synthetic_data = pd.read_excel('Synthetic_Industry_Data.xlsx', sheet_name=None)
    
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
    
    return iip_data, synthetic_data, stock_data, correlation_results, financial_data

iip_data, synthetic_data, stock_data, correlation_results, financial_data = load_data()

# Define Industry and Indicators
indicators = {
    'Manufacture of Food Products': {
        'Leading': ['Consumer Spending Trends', 'Agricultural Output', 'Retail Sales Data'],
        'Lagging': ['Inventory Levels', 'Employment Data']
    },
    'Manufacture of Beverages': {
        'Leading': ['Consumer Confidence', 'Raw Material Prices'],
        'Lagging': ['Production Output', 'Profit Margins']
    },
    'Manufacture of Tobacco Products': {
        'Leading': ['Regulatory Changes', 'Consumer Trends'],
        'Lagging': ['Sales Volume', 'Market Share']
    },
    'Manufacture of Textiles': {
        'Leading': ['Fashion Trends', 'Raw Material Prices'],
        'Lagging': ['Export Data', 'Inventory Levels']
    },
    'Manufacture of Wearing Apparel': {
        'Leading': ['Retail Sales of Apparel', 'Consumer Spending on Fashion'],
        'Lagging': ['Production and Sales Data', 'Employment Trends in Apparel Sector']
    },
    'Manufacture of Leather and Related Products': {
        'Leading': ['Fashion Industry Trends', 'Raw Material Prices (Leather)'],
        'Lagging': ['Sales and Revenue Data', 'Inventory Levels']
    },
    'Manufacture of Wood and Products of Wood and Cork': {
        'Leading': ['Housing Market Data', 'Building Permits'],
        'Lagging': ['Production Volume', 'Employment Data']
    },
    'Manufacture of Paper and Paper Products': {
        'Leading': ['Consumer Spending on Paper Goods', 'Raw Material Prices (Wood Pulp)'],
        'Lagging': ['Production Output', 'Sales Data']
    },
    'Printing and Reproduction of Recorded Media': {
        'Leading': ['Trends in Media Consumption', 'Technological Advances'],
        'Lagging': ['Production Volume', 'Revenue and Profit Margins']
    },
    'Manufacture of Coke and Refined Petroleum Products': {
        'Leading': ['Crude Oil Prices', 'Energy Demand Trends'],
        'Lagging': ['Refined Product Output', 'Profit Margins']
    },
    'Manufacture of Chemicals and Chemical Products': {
        'Leading': ['Raw Material Prices', 'Industrial Production Data'],
        'Lagging': ['Production Data', 'Sales Revenue']
    },
    'Manufacture of Pharmaceuticals, Medicinal Chemicals, and Botanical Products': {
        'Leading': ['Regulatory Approvals', 'Research and Development Investment'],
        'Lagging': ['Drug Sales Data', 'Profit Margins']
    },
    'Manufacture of Rubber and Plastics Products': {
        'Leading': ['Raw Material Prices', 'Automotive Industry Trends'],
        'Lagging': ['Production and Sales Data', 'Inventory Levels']
    },
    'Manufacture of Other Non-Metallic Mineral Products': {
        'Leading': ['Construction and Infrastructure Projects', 'Raw Material Prices'],
        'Lagging': ['Production Output', 'Sales Revenue']
    },
    'Manufacture of Basic Metals': {
        'Leading': ['Industrial Production', 'Raw Material Prices'],
        'Lagging': ['Production Data', 'Employment Trends']
    },
    'Manufacture of Fabricated Metal Products, Except Machinery and Equipment': {
        'Leading': ['Construction and Manufacturing Activity', 'Raw Material Prices'],
        'Lagging': ['Production Volume', 'Sales Data']
    },
    'Manufacture of Computer, Electronic, and Optical Products': {
        'Leading': ['Technology Trends', 'Consumer Electronics Sales'],
        'Lagging': ['Sales Revenue', 'Production Output']
    },
    'Manufacture of Electrical Equipment': {
        'Leading': ['Industrial Production Trends', 'Investment in Infrastructure'],
        'Lagging': ['Production and Sales Data', 'Profit Margins']
    },
    'Manufacture of Machinery and Equipment n.e.c.': {
        'Leading': ['Capital Expenditure', 'Industrial Production Data'],
        'Lagging': ['Production Data', 'Sales Revenue']
    },
    'Manufacture of Motor Vehicles, Trailers, and Semi-Trailers': {
        'Leading': ['Automobile Sales Trends', 'Consumer Confidence'],
        'Lagging': ['Production and Sales Data', 'Employment Trends']
    },
    'Manufacture of Other Transport Equipment': {
        'Leading': ['Transportation Infrastructure Investment', 'Global Trade Trends'],
        'Lagging': ['Production Output', 'Sales Revenue']
    },
    'Manufacture of Furniture': {
        'Leading': ['Housing Market Trends', 'Consumer Spending Trends'],
        'Lagging': ['Production and Sales Data', 'Inventory Levels']
    },
    'Other Manufacturing': {
        'Leading': ['Sector-Specific Trends', 'Raw Material Prices'],
        'Lagging': ['Production Output', 'Sales Data']
    },
    'India Inflation CPI (Consumer Price Index)': {
        'Leading': ['Consumer Spending Trends', 'Wage Growth', 'Raw Material Prices'],
        'Lagging': ['Previous CPI Data', 'Retail Sales Data', 'Employment Cost Index']
    },
    'India Inflation WPI (Wholesale Price Index)': {
        'Leading': ['Producer Price Trends', 'Raw Material Prices', 'Commodity Prices'],
        'Lagging': ['Previous WPI Data', 'Import and Export Prices', 'Manufacturing Output Data']
    },
    'India GDP Growth': {
        'Leading': ['Business Investment', 'Consumer Confidence Index', 'Industrial Production'],
        'Lagging': ['Previous GDP Data', 'Employment Data', 'Corporate Earnings']
    },
    'RBI Interest Rate': {
        'Leading': ['Inflation Data', 'Economic Growth Data', 'Global Interest Rates'],
        'Lagging': ['Previous RBI Interest Rates', 'Credit Growth Data', 'Inflation Adjustments']
    },
    'India Infrastructure Output': {
        'Leading': ['Building Permits', 'Government Infrastructure Spending', 'Construction Sector Activity'],
        'Lagging': ['Previous Infrastructure Output Data', 'Project Completion Rates', 'Employment in Construction Sector']
    },
    'India Banks Loan Growth Rate': {
        'Leading': ['Consumer Confidence', 'Business Investment Trends', 'Interest Rate Trends'],
        'Lagging': ['Previous Loan Growth Data', 'Bank Credit Demand', 'Non-Performing Loans']
    },
    'India Forex FX Reserves': {
        'Leading': ['Trade Balance Data', 'Foreign Investment Inflows', 'Global Economic Conditions'],
        'Lagging': ['Previous Forex Reserves Data', 'Currency Exchange Rates', 'International Financial Assistance']
    }
}

# Function to interpret correlation
def interpret_correlation(value):
    if value > 0.8:
        return "Strong Positive"
    elif 0.3 < value <= 0.8:
        return "Slight Positive"
    elif -0.3 <= value <= 0.3:
        return "Neutral"
    elif -0.8 <= value < -0.3:
        return "Slight Negative"
    else:
        return "Strong Negative"

# Function to adjust correlation values based on prediction
def adjust_correlation(correlation_value, predicted_value, industry_mean):
    return correlation_value * (predicted_value / industry_mean)

# Function to get detailed interpretation with Indian economy context
def get_detailed_interpretation(parameter_name, correlation_interpretation):
    interpretations = {
        "correlation with Total Revenue/Income": {
            "Slight Positive": (
                "* **Economic Context**: A slight increase in revenue associated with this metric indicates modest economic growth or improved market conditions in India. This could reflect a favorable domestic economic environment, such as increased consumer spending or supportive government policies. Globally, similar trends could be driven by synchronized economic recovery or regional trade dynamics. However, the overall impact remains modest, suggesting that other factors are also influencing revenue changes.\n"
                "* **Business Strategies**: Businesses may see a slight uplift in revenue due to incremental improvements in operational efficiencies, modestly successful marketing campaigns, or minor innovations. Companies should continue to focus on enhancing product quality and customer satisfaction to maintain this trend.\n"
                "* **Financial Environment**: In a slightly positive scenario, the financial environment remains stable with moderate inflation and manageable interest rates. Companies may experience slight improvements in profit margins as costs remain under control.\n"
                "* **Global Conditions**: Globally, this trend could be driven by consistent but not extraordinary economic growth. The impact might be visible through stable export performance or gradual market expansion in international territories.\n"
                "* **Strategy for Investors**: Investors should be cautious but optimistic. While the slight positive trend is encouraging, it may not be sufficient to make significant investment decisions. It is advisable to monitor industry trends and economic policies closely for any signs of accelerated growth."
            ),
            "Strong Positive": (
                "* **Economic Context**: A significant increase in revenue linked to this metric highlights robust economic growth or exceptional market conditions in India. This could be driven by substantial increases in domestic demand, favorable fiscal policies, or successful market expansion strategies. On a global scale, this might align with strong economic growth in key markets or advantageous trade agreements. The company is likely capitalizing on these conditions to achieve substantial revenue gains.\n"
                "* **Business Strategies**: Companies might benefit from successful product launches, aggressive market penetration, or exceptional operational efficiencies. High revenue growth could result from scaling operations, entering new markets, or leveraging strategic partnerships.\n"
                "* **Financial Environment**: A strong positive trend suggests favorable financial conditions, such as low interest rates, favorable exchange rates, and strong capital markets. Companies may see improved profitability and higher returns on investment.\n"
                "* **Global Conditions**: On a global scale, strong economic growth, favorable trade agreements, or recovery in key international markets could contribute to the positive trend. Businesses might experience increased demand for their products and services internationally.\n"
                "* **Strategy for Investors**: Investors should consider increasing their investments in companies showing strong positive correlations. It may be beneficial to capitalize on growth opportunities and align with businesses that are outperforming the market. Close monitoring of company performance and market conditions is essential."
            ),
            "Neutral": (
                "* **Economic Context**: A Neutral correlation implies that revenue changes are largely unaffected by this metric, both in the context of the Indian economy and globally. This suggests that other factors, such as broad economic trends, industry-specific developments, or global market conditions, are the primary drivers of revenue growth, rather than this specific metric.\n"
                "* **Business Strategies**: Businesses might be operating in a steady state with no significant changes in their revenue patterns. Strategies should focus on maintaining efficiency and managing costs effectively.\n"
                "* **Financial Environment**: The financial environment is stable with no major fluctuations in interest rates or inflation. Companies may not experience significant revenue growth or decline but should remain vigilant to any potential changes.\n"
                "* **Global Conditions**: Globally, a neutral correlation suggests that international economic conditions are neither driving significant revenue growth nor causing declines. This may reflect a period of stability in global markets.\n"
                "* **Strategy for Investors**: Investors should adopt a wait-and-see approach. While the current situation is stable, it’s crucial to keep an eye on any emerging trends or changes in the economic environment that could impact future revenue."
            ),
            "Slight Negative": (
                "* **Economic Context**: A slight decrease in revenue associated with this metric could signal early signs of economic or operational challenges in India, such as minor disruptions in consumer demand or rising operational costs. Globally, this might be compounded by trade tensions or economic slowdowns in key markets. The company might be experiencing slight headwinds, reflecting initial impacts on revenue.\n"
                "* **Business Strategies**: Companies should focus on cost control, optimizing operational efficiencies, and diversifying their product lines to counteract the slight revenue decline. Strategic adjustments might be necessary to sustain profitability.\n"
                "* **Financial Environment**: A slightly negative trend may indicate minor financial pressures, such as rising interest rates or inflation. Companies might need to manage their financial resources carefully to mitigate the impact on revenue.\n"
                "* **Global Conditions**: Globally, this could be linked to minor economic slowdowns or trade uncertainties affecting revenue. Companies should remain adaptable and seek new opportunities to offset the negative impact.\n"
                "* **Strategy for Investors**: Investors should be cautious and evaluate the potential for recovery. It may be wise to review the company's strategies and financial health to ensure it can navigate the slight downturn effectively."
            ),
            "Strong Negative": (
                "* **Economic Context**: A significant decrease in revenue linked to this metric indicates severe economic or operational difficulties in India, such as substantial declines in consumer spending or major disruptions in supply chains. Globally, this could be exacerbated by economic downturns, geopolitical instability, or adverse trade conditions. The company may be facing considerable challenges, impacting its revenue significantly.\n"
                "* **Business Strategies**: Companies might be facing substantial operational challenges or strategic missteps. There may be a need for comprehensive restructuring, cost reduction measures, or strategic pivots to address the severe revenue decline.\n"
                "* **Financial Environment**: The financial environment could be marked by high interest rates, severe inflation, or unstable capital markets. Companies need to implement robust financial strategies to manage their resources and mitigate the adverse effects.\n"
                "* **Global Conditions**: On a global scale, economic downturns, geopolitical tensions, or severe market disruptions could exacerbate revenue declines. Companies should focus on global market diversification and risk management strategies.\n"
                "* **Strategy for Investors**: Investors should exercise extreme caution and reassess their investments in companies showing strong negative correlations. It may be prudent to explore alternative investment opportunities or seek companies with better resilience to economic challenges."
            )
        },
        "correlation with Total Operating Expense": {
            "Slight Positive": (
                "* **Economic Context**: A slight increase in operating expenses associated with this metric suggests modest operational growth or rising costs in India. This could reflect factors such as increased production scale, higher raw material costs, or incremental investments in operations. Globally, similar trends might be observed in growing markets or due to rising input costs.\n"
                "* **Business Strategies**: Companies might be experiencing increased operational costs due to expansion efforts or higher input costs. To manage these expenses, businesses should focus on efficiency improvements, cost control measures, and strategic procurement.\n"
                "* **Financial Environment**: A slightly positive trend in operating expenses indicates stable but increasing cost pressures. Companies may need to navigate moderate inflation or rising costs of raw materials and services.\n"
                "* **Global Conditions**: Globally, this trend could be related to increasing costs in production or supply chains. Businesses should monitor global cost trends and adjust their strategies accordingly.\n"
                "* **Strategy for Investors**: Investors should be aware of rising operational costs as a potential risk factor. It’s essential to assess how well companies manage their expenses and whether they have strategies in place to maintain profitability despite rising costs."
            ),
            "Strong Positive": (
                "* **Economic Context**: A significant increase in operating expenses linked to this metric indicates substantial operational expansion or significant cost pressures in India. This might be due to major investments in infrastructure, increased production scale, or significant rises in input costs. Globally, this trend could reflect widespread cost inflation or substantial operational growth.\n"
                "* **Business Strategies**: Companies may be facing high operational costs due to large-scale expansions, significant increases in input prices, or major upgrades in technology and infrastructure. Effective cost management strategies, including cost-benefit analyses and operational efficiencies, are crucial.\n"
                "* **Financial Environment**: A strong positive trend suggests substantial cost pressures. Companies should focus on financial planning and cost control measures to manage the impact on their profitability.\n"
                "* **Global Conditions**: Globally, this trend could be driven by widespread cost increases or significant growth in operational activities. Companies should be prepared for similar trends in international markets and adjust their strategies accordingly.\n"
                "* **Strategy for Investors**: Investors should closely monitor companies' ability to manage rising operational costs. It is important to evaluate whether increased expenses are leading to higher revenues and whether companies have strategies in place to maintain profitability."
            ),
            "Neutral": (
                "* **Economic Context**: A Neutral correlation with operating expenses suggests that changes in operating costs are not significantly affecting or being affected by broader economic conditions in India. This may indicate stable operational efficiency or balanced cost management.\n"
                "* **Business Strategies**: Businesses are likely managing their operating expenses effectively, with no major fluctuations impacting overall performance. Maintaining this balance is key to sustaining profitability.\n"
                "* **Financial Environment**: The financial environment remains stable with no significant changes in cost pressures. Companies should continue to monitor their cost structures and adapt to any emerging trends.\n"
                "* **Global Conditions**: Globally, a neutral correlation indicates stable operating costs with no major external influences affecting expenses. Companies should remain aware of any potential changes in global cost conditions.\n"
                "* **Strategy for Investors**: Investors should focus on companies with stable cost management practices. While a neutral trend may suggest stability, ongoing vigilance is necessary to ensure that cost structures remain optimized."
            ),
            "Slight Negative": (
                "* **Economic Context**: A slight decrease in operating expenses associated with this metric might suggest minor improvements in cost efficiency or operational reductions in India. This could be due to cost-cutting measures, process optimizations, or reductions in input prices.\n"
                "* **Business Strategies**: Companies might be benefiting from cost reductions or improved operational efficiencies. Continued focus on cost management and efficiency improvements can help sustain this trend.\n"
                "* **Financial Environment**: A slightly negative trend in operating expenses indicates decreasing cost pressures. Companies may benefit from lower inflation rates or reduced input costs.\n"
                "* **Global Conditions**: Globally, this trend might reflect improving cost conditions or efficiency gains. Companies should stay informed about global cost trends and leverage opportunities for cost savings.\n"
                "* **Strategy for Investors**: Investors should consider the impact of decreasing operational costs on overall profitability. It’s important to assess whether these reductions are sustainable and if they contribute to improved financial performance."
            ),
            "Strong Negative": (
                "* **Economic Context**: A significant decrease in operating expenses linked to this metric indicates major improvements in cost management or significant reductions in operational costs in India. This might be due to extensive cost-cutting measures, significant reductions in input costs, or operational streamlining. Globally, similar trends could reflect widespread cost reductions or operational efficiencies.\n"
                "* **Business Strategies**: Companies likely have implemented effective cost-cutting strategies or streamlined operations significantly. It’s essential to ensure that cost reductions do not compromise operational capabilities or long-term growth.\n"
                "* **Financial Environment**: A strong negative trend suggests substantial decreases in operational costs. Companies should capitalize on these cost savings to enhance profitability and invest in growth opportunities.\n"
                "* **Global Conditions**: On a global scale, this trend could be related to extensive cost reductions or improvements in operational efficiency. Companies should remain vigilant about maintaining efficiency and monitoring global cost trends.\n"
                "* **Strategy for Investors**: Investors should view strong reductions in operating expenses positively if they contribute to improved profitability. However, it’s crucial to ensure that cost reductions do not negatively impact the company’s growth potential or operational effectiveness."
            )
        },
        "correlation with Operating Income/Profit": {
            "Slight Positive": (
                "* **Economic Context**: A slight increase in operating income/profit associated with this metric suggests modest improvements in operational performance or cost management in India. This could indicate minor enhancements in revenue generation or operational efficiency. Globally, similar trends might be due to gradual economic recovery or improvements in market conditions.\n"
                "* **Business Strategies**: Companies may experience a slight uplift in operating income/profit due to incremental improvements in business operations, cost control measures, or minor increases in revenue. Continued focus on operational efficiency and effective cost management is crucial.\n"
                "* **Financial Environment**: A slightly positive trend in operating income/profit suggests stable financial conditions with moderate improvements. Companies might benefit from gradual reductions in operational costs or slight increases in revenue.\n"
                "* **Global Conditions**: Globally, a slight positive correlation might reflect gradual improvements in operational performance or market conditions. Companies should monitor global trends that could impact their operating income/profit.\n"
                "* **Strategy for Investors**: Investors should be cautious but optimistic about the slight positive trend. While modest improvements are encouraging, they may not yet justify major investment decisions. It is advisable to keep an eye on ongoing operational performance and market conditions."
            ),
            "Strong Positive": (
                "* **Economic Context**: A significant increase in operating income/profit linked to this metric indicates robust improvements in operational performance or significant cost management successes in India. This could be due to strong revenue growth, substantial operational efficiencies, or successful strategic initiatives. On a global scale, this trend might be driven by major economic recoveries or favorable market conditions.\n"
                "* **Business Strategies**: Companies experiencing strong positive correlations may be benefiting from major operational improvements, successful product innovations, or significant revenue growth. Effective cost management and strategic investments are likely contributing to higher operating income/profit.\n"
                "* **Financial Environment**: A strong positive trend in operating income/profit suggests favorable financial conditions, such as strong revenue growth, reduced operational costs, or improved profit margins. Companies may see substantial gains in profitability and operational efficiency.\n"
                "* **Global Conditions**: Globally, this trend might reflect significant improvements in market conditions or economic recoveries. Companies should leverage this positive momentum to further enhance their operational performance and profitability.\n"
                "* **Strategy for Investors**: Investors should view strong positive correlations favorably, considering them a sign of robust operational performance and potential for higher returns. Increasing investments in companies demonstrating significant improvements in operating income/profit could be beneficial."
            ),
            "Neutral": (
                "* **Economic Context**: A Neutral correlation with operating income/profit suggests that changes in this metric are not significantly influenced by broader economic conditions in India. This may indicate stable operational performance with no major fluctuations affecting profitability.\n"
                "* **Business Strategies**: Companies are likely maintaining steady operational performance with no significant changes in operating income/profit. Focus should remain on sustaining current strategies and managing costs effectively.\n"
                "* **Financial Environment**: The financial environment is stable with no major changes impacting operating income/profit. Companies should continue to monitor their financial health and operational performance.\n"
                "* **Global Conditions**: Globally, a neutral correlation indicates stable conditions with no significant impacts on operating income/profit. Companies should stay aware of global trends but expect no major changes in profitability.\n"
                "* **Strategy for Investors**: Investors should maintain a steady approach, monitoring the company's operational performance and financial health. While a neutral trend suggests stability, it’s important to stay alert for any emerging trends or changes."
            ),
            "Slight Negative": (
                "* **Economic Context**: A slight decrease in operating income/profit associated with this metric suggests minor challenges in operational performance or cost management in India. This could be due to slight increases in costs, minor revenue declines, or operational inefficiencies. Globally, this might reflect emerging economic or market pressures.\n"
                "* **Business Strategies**: Companies facing slight negative trends might be dealing with small-scale operational issues or increased costs. Strategies should focus on improving efficiency, cost control, and addressing minor operational challenges.\n"
                "* **Financial Environment**: A slightly negative trend in operating income/profit indicates modest financial pressures, such as increased operational costs or slight revenue declines. Companies should work on mitigating these pressures through effective cost management and operational improvements.\n"
                "* **Global Conditions**: Globally, this trend might indicate emerging economic or market pressures affecting profitability. Companies should monitor international conditions and adjust their strategies to maintain profitability.\n"
                "* **Strategy for Investors**: Investors should be cautious about slight negative trends, evaluating the company's strategies for addressing operational challenges. It’s important to assess whether the company has a plan to reverse the trend and improve profitability."
            ),
            "Strong Negative": (
                "* **Economic Context**: A significant decrease in operating income/profit linked to this metric indicates substantial operational or financial difficulties in India. This could be due to severe cost increases, major revenue declines, or significant inefficiencies. On a global scale, this trend might be exacerbated by economic downturns or adverse market conditions.\n"
                "* **Business Strategies**: Companies experiencing strong negative correlations may be facing severe operational challenges or significant financial difficulties. Comprehensive strategies are needed to address cost issues, operational inefficiencies, and revenue declines.\n"
                "* **Financial Environment**: A strong negative trend in operating income/profit suggests challenging financial conditions, such as high operational costs, significant revenue declines, or severe market pressures. Companies should implement robust financial strategies to manage these challenges.\n"
                "* **Global Conditions**: Globally, this trend might reflect broader economic or market disruptions affecting profitability. Companies should focus on global diversification and risk management strategies to counteract adverse effects.\n"
                "* **Strategy for Investors**: Investors should exercise caution and reassess their investments in companies with strong negative correlations. It may be prudent to explore alternative investments or closely monitor the company’s recovery strategies and overall financial health."
            )
        },
         "correlation with EBITDA": {
            "Slight Positive": (
                "* **Economic Context**: A slight increase in EBITDA associated with this metric indicates minor improvements in operational efficiency or revenue generation in India. This suggests that the company is experiencing modest enhancements in its ability to generate earnings before interest, taxes, depreciation, and amortization. Globally, this trend may be due to gradual economic improvements or effective cost management strategies.\n"
                "* **Business Strategies**: Companies may benefit from incremental gains in EBITDA through improved operational processes, slight revenue increases, or cost-saving measures. It is essential to continue focusing on optimizing operations and managing costs effectively.\n"
                "* **Financial Environment**: A slight positive trend in EBITDA reflects stable financial conditions with moderate improvements. Companies might see gradual improvements in profitability as a result of better cost management or operational efficiencies.\n"
                "* **Global Conditions**: Globally, a slight positive correlation suggests that companies are experiencing small but positive changes in EBITDA. This could be driven by favorable international market conditions or gradual economic recoveries.\n"
                "* **Strategy for Investors**: Investors should be cautiously optimistic about the slight positive trend in EBITDA. While the improvements are encouraging, they may not be substantial enough for major investment decisions. Monitoring ongoing operational performance and market conditions is advisable."
            ),
            "Strong Positive": (
                "* **Economic Context**: A significant increase in EBITDA linked to this metric highlights robust improvements in operational efficiency or substantial revenue growth in India. This indicates that the company is generating significantly more earnings before interest, taxes, depreciation, and amortization. On a global scale, this trend might be driven by strong economic recoveries or effective strategic initiatives.\n"
                "* **Business Strategies**: Companies showing strong positive correlations in EBITDA are likely benefiting from major operational enhancements, successful cost control measures, or significant revenue increases. Strategic initiatives and effective operational management play a crucial role in achieving higher EBITDA.\n"
                "* **Financial Environment**: A strong positive trend in EBITDA suggests favorable financial conditions, such as increased profitability and improved operational efficiencies. Companies might see substantial gains in earnings and better financial health.\n"
                "* **Global Conditions**: Globally, this trend may reflect strong market conditions or significant economic recoveries. Companies should leverage these favorable conditions to further enhance their EBITDA and expand their market presence.\n"
                "* **Strategy for Investors**: Investors should view strong positive correlations with EBITDA favorably, as they indicate robust operational performance and potential for high returns. Increasing investments in companies demonstrating substantial EBITDA growth could be advantageous."
            ),
            "Neutral": (
                "* **Economic Context**: A Neutral correlation with EBITDA suggests that changes in this metric are not significantly influenced by broader economic conditions in India. This indicates stable operational performance with no major fluctuations in EBITDA.\n"
                "* **Business Strategies**: Companies are likely maintaining steady EBITDA with no significant changes in their earnings before interest, taxes, depreciation, and amortization. Strategies should focus on sustaining current performance and managing costs effectively.\n"
                "* **Financial Environment**: The financial environment is stable, with no major changes impacting EBITDA. Companies should continue to monitor their financial health and operational performance to maintain stability.\n"
                "* **Global Conditions**: Globally, a neutral correlation indicates stability with no significant impacts on EBITDA. Companies should be aware of global trends but expect no major changes in their earnings.\n"
                "* **Strategy for Investors**: Investors should maintain a steady approach, keeping an eye on the company’s EBITDA performance and broader economic conditions. While a neutral trend suggests stability, ongoing monitoring of any emerging trends is important."
            ),
            "Slight Negative": (
                "* **Economic Context**: A slight decrease in EBITDA associated with this metric suggests minor challenges in operational efficiency or revenue generation in India. This may indicate slight declines in earnings before interest, taxes, depreciation, and amortization due to increased costs or decreased revenue. Globally, this might reflect emerging economic pressures or market challenges.\n"
                "* **Business Strategies**: Companies facing a slight negative trend in EBITDA should focus on addressing operational inefficiencies, managing costs, and exploring strategies to improve revenue. It is crucial to identify and mitigate any minor issues affecting profitability.\n"
                "* **Financial Environment**: A slightly negative trend in EBITDA suggests minor financial pressures, such as increased operational costs or slight revenue declines. Companies should work on controlling costs and enhancing operational efficiencies.\n"
                "* **Global Conditions**: Globally, this trend might indicate minor economic or market pressures affecting EBITDA. Companies should remain vigilant to international conditions that could impact their earnings.\n"
                "* **Strategy for Investors**: Investors should be cautious about slight negative trends and evaluate the company’s strategies for addressing operational challenges. Assessing the company’s ability to reverse the trend and improve EBITDA is essential."
            ),
            "Strong Negative": (
                "* **Economic Context**: A significant decrease in EBITDA linked to this metric indicates severe operational or financial difficulties in India. This suggests substantial declines in earnings before interest, taxes, depreciation, and amortization due to major cost increases, significant revenue declines, or severe operational inefficiencies. On a global scale, this trend may be exacerbated by economic downturns or adverse market conditions.\n"
                "* **Business Strategies**: Companies experiencing strong negative correlations with EBITDA might be facing significant operational challenges or financial difficulties. Comprehensive strategies are needed to address major cost issues, improve operational efficiencies, and mitigate revenue declines.\n"
                "* **Financial Environment**: A strong negative trend in EBITDA suggests challenging financial conditions, such as high costs, severe revenue declines, or substantial market pressures. Companies should implement robust financial strategies to manage these challenges and improve profitability.\n"
                "* **Global Conditions**: Globally, this trend could reflect broader economic downturns or market disruptions affecting EBITDA. Companies should focus on risk management and global market diversification to counteract adverse effects.\n"
                "* **Strategy for Investors**: Investors should exercise extreme caution with companies showing strong negative correlations in EBITDA. It may be prudent to reassess investments, explore alternative opportunities, and closely monitor the company’s recovery strategies and financial health."
            )
        },
        "correlation with EBIT": {
            "Slight Positive": (
                "* **Economic Context**: A slight positive correlation with EBIT indicates minor improvements in operational profitability in India. This suggests that the company is experiencing modest gains in earnings before interest and taxes. Globally, this might be attributed to incremental economic growth or improvements in operational efficiencies.\n"
                "* **Business Strategies**: Companies benefiting from a slight positive EBIT correlation should focus on enhancing operational efficiency and managing costs effectively. Small improvements in operational processes or revenue generation can positively impact EBIT.\n"
                "* **Financial Environment**: A slight positive trend in EBIT reflects stable financial conditions with minor improvements. Companies may see small gains in profitability as a result of better cost management or slight increases in revenue.\n"
                "* **Global Conditions**: On a global scale, this trend may reflect modest improvements in market conditions or economic stability. Companies should leverage these improvements to boost their EBIT.\n"
                "* **Strategy for Investors**: Investors should view the slight positive trend in EBIT with cautious optimism. While the improvements are encouraging, they may not be substantial enough for major investment decisions. Monitoring ongoing operational performance is advisable."
            ),
            "Strong Positive": (
                "* **Economic Context**: A strong positive correlation with EBIT signifies significant improvements in operational profitability in India. This indicates that the company is achieving substantial gains in earnings before interest and taxes. Globally, this trend may be driven by robust economic growth or effective strategic initiatives.\n"
                "* **Business Strategies**: Companies experiencing strong positive correlations in EBIT likely benefit from major operational enhancements, successful cost management, or significant revenue increases. Strategic initiatives and effective management are key drivers of higher EBIT.\n"
                "* **Financial Environment**: A strong positive trend in EBIT suggests favorable financial conditions, such as increased profitability and improved operational efficiencies. Companies may see substantial gains in earnings and better financial health.\n"
                "* **Global Conditions**: Globally, this trend may reflect strong market conditions or economic recoveries. Companies should leverage these favorable conditions to enhance their EBIT further.\n"
                "* **Strategy for Investors**: Investors should view companies with strong positive EBIT correlations favorably, as they indicate robust operational performance and potential for high returns. Increasing investments in such companies could be advantageous."
            ),
            "Neutral": (
                "* **Economic Context**: A Neutral correlation with EBIT suggests that changes in this metric are not significantly influenced by broader economic conditions in India. This indicates stable operational performance with no major fluctuations in EBIT.\n"
                "* **Business Strategies**: Companies maintaining a neutral EBIT correlation are likely experiencing steady earnings before interest and taxes. Strategies should focus on sustaining current performance and managing costs effectively.\n"
                "* **Financial Environment**: The financial environment is stable, with no major changes impacting EBIT. Companies should continue to monitor their financial health and operational performance to maintain stability.\n"
                "* **Global Conditions**: Globally, a neutral correlation suggests stability with no significant impacts on EBIT. Companies should be aware of global trends but expect no major changes in their earnings.\n"
                "* **Strategy for Investors**: Investors should maintain a steady approach, keeping an eye on the company’s EBIT performance and broader economic conditions. While a neutral trend suggests stability, ongoing monitoring of any emerging trends is important."
            ),
            "Slight Negative": (
                "* **Economic Context**: A slight negative correlation with EBIT indicates minor declines in operational profitability in India. This suggests that the company is experiencing slight decreases in earnings before interest and taxes. Globally, this might be due to emerging economic pressures or operational challenges.\n"
                "* **Business Strategies**: Companies with a slight negative EBIT correlation should focus on addressing operational inefficiencies, managing costs, and exploring strategies to improve profitability. It is important to identify and mitigate minor issues affecting EBIT.\n"
                "* **Financial Environment**: A slightly negative trend in EBIT reflects minor financial pressures, such as increased operational costs or slight revenue declines. Companies should work on controlling costs and improving operational efficiencies.\n"
                "* **Global Conditions**: Globally, this trend might indicate minor economic or market pressures affecting EBIT. Companies should remain vigilant to international conditions that could impact their earnings.\n"
                "* **Strategy for Investors**: Investors should be cautious about slight negative trends and evaluate the company’s strategies for addressing operational challenges. Assessing the company’s ability to reverse the trend and improve EBIT is essential."
            ),
            "Strong Negative": (
                "* **Economic Context**: A strong negative correlation with EBIT indicates severe declines in operational profitability in India. This suggests significant reductions in earnings before interest and taxes due to major cost increases, significant revenue declines, or severe operational inefficiencies. Globally, this trend may be exacerbated by economic downturns or adverse market conditions.\n"
                "* **Business Strategies**: Companies experiencing strong negative correlations with EBIT might be facing substantial operational challenges or financial difficulties. Comprehensive strategies are needed to address major cost issues, improve operational efficiencies, and mitigate revenue declines.\n"
                "* **Financial Environment**: A strong negative trend in EBIT suggests challenging financial conditions, such as high costs, severe revenue declines, or substantial market pressures. Companies should implement robust financial strategies to manage these challenges and improve profitability.\n"
                "* **Global Conditions**: Globally, this trend could reflect broader economic downturns or market disruptions affecting EBIT. Companies should focus on risk management and global market diversification to counteract adverse effects.\n"
                "* **Strategy for Investors**: Investors should exercise extreme caution with companies showing strong negative correlations in EBIT. It may be prudent to reassess investments, explore alternative opportunities, and closely monitor the company’s recovery strategies and financial health."
            )
        },
        "correlation with Income/Profit Before Tax": {
            "Slight Positive": (
                "* **Economic Context**: A slight positive correlation with Income/Profit Before Tax suggests modest improvements in profitability before taxes in India. This indicates that the company is experiencing minor gains in its income or profit before tax, reflecting incremental positive changes in business performance.\n"
                "* **Business Strategies**: Companies with a slight positive correlation in Income/Profit Before Tax should continue to focus on enhancing operational efficiency and cost management. Small increases in revenue or reductions in expenses could further improve profitability.\n"
                "* **Financial Environment**: A slight positive trend in Income/Profit Before Tax points to stable financial conditions with minor gains. Companies might see small benefits from improved operational performance or cost controls.\n"
                "* **Global Conditions**: Globally, this trend may reflect modest improvements in market conditions or economic stability. Companies should leverage positive global trends to boost their pre-tax income.\n"
                "* **Strategy for Investors**: Investors should view the slight positive trend with cautious optimism. While the improvements are encouraging, they may not be substantial enough for major investment decisions. Ongoing monitoring of operational performance and broader economic conditions is advisable."
            ),
            "Strong Positive": (
                "* **Economic Context**: A strong positive correlation with Income/Profit Before Tax signifies significant improvements in profitability before taxes in India. This suggests that the company is achieving substantial gains in its income or profit before tax, driven by robust business performance and favorable economic conditions.\n"
                "* **Business Strategies**: Companies experiencing strong positive correlations in Income/Profit Before Tax likely benefit from major operational improvements, successful strategic initiatives, or significant revenue growth. Continued focus on strategic initiatives and operational excellence is essential.\n"
                "* **Financial Environment**: A strong positive trend indicates favorable financial conditions, such as increased profitability and improved operational efficiencies. Companies may see substantial gains in pre-tax income and better financial health.\n"
                "* **Global Conditions**: On a global scale, this trend may be attributed to strong market conditions or economic growth. Companies should capitalize on these favorable conditions to further enhance their pre-tax income.\n"
                "* **Strategy for Investors**: Investors should consider increasing investments in companies with strong positive correlations in Income/Profit Before Tax. This reflects robust financial performance and potential for high returns. It's advantageous to align with companies showing strong profitability growth."
            ),
            "Neutral": (
                "* **Economic Context**: A neutral correlation with Income/Profit Before Tax suggests that changes in this metric are not significantly influenced by broader economic conditions in India. This indicates stable profitability before taxes with no major fluctuations.\n"
                "* **Business Strategies**: Companies with a neutral correlation should focus on maintaining steady operational performance and managing costs effectively. There are no significant changes in profitability to address.\n"
                "* **Financial Environment**: The financial environment remains stable, with no major impacts on Income/Profit Before Tax. Companies should continue monitoring their financial performance and operational efficiencies.\n"
                "* **Global Conditions**: Globally, a neutral trend suggests stability in profitability with no major influences from international markets. Companies should stay informed of global economic conditions but expect no significant changes.\n"
                "* **Strategy for Investors**: Investors should maintain a steady approach. While the neutral trend indicates stability, it's important to keep an eye on any emerging trends or changes in economic conditions that could impact future profitability."
            ),
            "Slight Negative": (
                "* **Economic Context**: A slight negative correlation with Income/Profit Before Tax indicates minor declines in profitability before taxes in India. This suggests that the company is experiencing slight decreases in income or profit before tax, potentially due to emerging operational or economic challenges.\n"
                "* **Business Strategies**: Companies with a slight negative correlation should focus on addressing minor operational inefficiencies, managing costs, and exploring ways to improve profitability. Small adjustments can help counteract the slight decline.\n"
                "* **Financial Environment**: A slight negative trend reflects minor financial pressures, such as increased operational costs or slight revenue declines. Companies need to manage these pressures carefully to maintain profitability.\n"
                "* **Global Conditions**: Globally, this trend might indicate minor economic or market pressures affecting profitability. Companies should stay aware of international conditions that could impact their income or profit before tax.\n"
                "* **Strategy for Investors**: Investors should be cautious about the slight negative trend and evaluate the company's strategies for addressing operational challenges. Assessing the company's ability to mitigate the decline and improve profitability is important."
            ),
            "Strong Negative": (
                "* **Economic Context**: A strong negative correlation with Income/Profit Before Tax indicates significant declines in profitability before taxes in India. This suggests substantial reductions in income or profit before tax due to major operational or economic difficulties.\n"
                "* **Business Strategies**: Companies with a strong negative correlation may be facing severe operational challenges or financial difficulties. Comprehensive strategies are needed to address major cost issues, enhance operational efficiencies, and improve profitability before taxes.\n"
                "* **Financial Environment**: A strong negative trend in Income/Profit Before Tax points to challenging financial conditions, such as high costs or significant revenue declines. Companies must implement robust financial strategies to manage these challenges and improve their profitability.\n"
                "* **Global Conditions**: Globally, this trend may reflect broader economic downturns or market disruptions affecting profitability. Companies should focus on risk management and explore strategies to mitigate adverse effects.\n"
                "* **Strategy for Investors**: Investors should exercise extreme caution with companies showing strong negative correlations in Income/Profit Before Tax. It may be prudent to reassess investments, seek alternative opportunities, and monitor the company’s recovery strategies and financial health closely."
            )
        },
        "correlation with Net Income From Continuing Operations": {
            "Slight Positive": (
                "* **Economic Context**: A slight positive correlation with Net Income From Continuing Operations suggests minor improvements in profitability from ongoing operations in India. This indicates that the company is seeing small gains in net income from its core business activities, reflecting modest positive changes in operational performance.\n"
                "* **Business Strategies**: Companies with a slight positive correlation should continue to focus on enhancing core business operations and managing costs efficiently. Small improvements in revenue or operational efficiencies can further boost net income from continuing operations.\n"
                "* **Financial Environment**: A slight positive trend in Net Income From Continuing Operations indicates generally favorable financial conditions with incremental gains. Companies may experience slight benefits from improved operational performance and cost management.\n"
                "* **Global Conditions**: Globally, this trend could be influenced by stable or improving international markets, contributing to modest gains in net income from ongoing operations. Companies should leverage positive global conditions to enhance profitability.\n"
                "* **Strategy for Investors**: Investors should view the slight positive trend with cautious optimism. While the improvements are encouraging, they may not be substantial enough to make major investment decisions. Monitoring operational performance and broader economic conditions is advisable."
            ),
            "Strong Positive": (
                "* **Economic Context**: A strong positive correlation with Net Income From Continuing Operations signifies substantial improvements in profitability from ongoing operations in India. This suggests that the company is achieving significant gains in net income from its core business activities, driven by robust business performance and favorable economic conditions.\n"
                "* **Business Strategies**: Companies experiencing strong positive correlations in Net Income From Continuing Operations are likely benefiting from major operational improvements, successful strategic initiatives, or significant revenue growth. Continued focus on strategic initiatives and operational excellence is crucial.\n"
                "* **Financial Environment**: A strong positive trend indicates favorable financial conditions, such as increased profitability and improved operational efficiencies. Companies may see substantial gains in net income from continuing operations and better financial health.\n"
                "* **Global Conditions**: On a global scale, this trend may be attributed to strong market conditions or economic growth. Companies should capitalize on these favorable conditions to further enhance their net income from continuing operations.\n"
                "* **Strategy for Investors**: Investors should consider increasing investments in companies with strong positive correlations in Net Income From Continuing Operations. This reflects robust financial performance and potential for high returns. It’s advantageous to align with companies showing significant profitability growth."
            ),
            "Neutral": (
                "* **Economic Context**: A neutral correlation with Net Income From Continuing Operations indicates that changes in this metric are not significantly influenced by broader economic conditions in India. This suggests stable profitability from ongoing operations with no major fluctuations.\n"
                "* **Business Strategies**: Companies with a neutral correlation should focus on maintaining steady operational performance and managing costs effectively. There are no significant changes in net income to address.\n"
                "* **Financial Environment**: The financial environment remains stable, with no major impacts on Net Income From Continuing Operations. Companies should continue to monitor their financial performance and operational efficiencies.\n"
                "* **Global Conditions**: Globally, a neutral trend suggests stability in profitability from continuing operations with no major influences from international markets. Companies should stay informed of global economic conditions but expect no significant changes.\n"
                "* **Strategy for Investors**: Investors should maintain a steady approach. While the neutral trend indicates stability, it's important to keep an eye on any emerging trends or changes in economic conditions that could impact future profitability."
            ),
            "Slight Negative": (
                "* **Economic Context**: A slight negative correlation with Net Income From Continuing Operations indicates minor declines in profitability from ongoing operations in India. This suggests that the company is experiencing slight decreases in net income from its core business activities, potentially due to emerging operational or economic challenges.\n"
                "* **Business Strategies**: Companies with a slight negative correlation should focus on addressing minor operational inefficiencies, managing costs, and exploring ways to improve net income. Small adjustments can help counteract the slight decline.\n"
                "* **Financial Environment**: A slight negative trend reflects minor financial pressures, such as increased operational costs or slight revenue declines. Companies need to manage these pressures carefully to maintain profitability.\n"
                "* **Global Conditions**: Globally, this trend might indicate minor economic or market pressures affecting net income from continuing operations. Companies should stay aware of international conditions that could impact their profitability.\n"
                "* **Strategy for Investors**: Investors should be cautious about the slight negative trend and evaluate the company's strategies for addressing operational challenges. Assessing the company's ability to mitigate the decline and improve profitability is important."
            ),
            "Strong Negative": (
                "* **Economic Context**: A strong negative correlation with Net Income From Continuing Operations indicates significant declines in profitability from ongoing operations in India. This suggests substantial reductions in net income from core business activities due to major operational or economic difficulties.\n"
                "* **Business Strategies**: Companies with a strong negative correlation may be facing severe operational challenges or financial difficulties. Comprehensive strategies are needed to address major cost issues, enhance operational efficiencies, and improve net income from continuing operations.\n"
                "* **Financial Environment**: A strong negative trend in Net Income From Continuing Operations points to challenging financial conditions, such as high costs or significant revenue declines. Companies must implement robust financial strategies to manage these challenges and improve their profitability.\n"
                "* **Global Conditions**: Globally, this trend may reflect broader economic downturns or market disruptions affecting net income. Companies should focus on risk management and explore strategies to mitigate adverse effects.\n"
                "* **Strategy for Investors**: Investors should exercise extreme caution with companies showing strong negative correlations in Net Income From Continuing Operations. It may be prudent to reassess investments, seek alternative opportunities, and monitor the company’s recovery strategies and financial health closely."
            )
        },
        "correlation with Net Income": {
            "Slight Positive": (
                "* **Economic Context**: A slight positive correlation with Net Income indicates that there are minor improvements in overall profitability. This suggests that the company's net income is experiencing small but favorable changes due to either operational efficiencies, increased revenues, or reduced costs.\n"
                "* **Business Strategies**: Companies experiencing a slight positive correlation should continue focusing on enhancing operational efficiencies and cost management. Small improvements in various aspects of business operations are beneficial and should be leveraged for better financial outcomes.\n"
                "* **Financial Environment**: A slight positive trend in Net Income implies generally stable financial conditions with incremental gains. Companies may benefit from favorable market conditions or cost controls that contribute to increased profitability.\n"
                "* **Global Conditions**: Globally, this trend might be influenced by steady economic growth or favorable international market conditions. Companies can benefit from stable global conditions that support their profitability.\n"
                "* **Strategy for Investors**: Investors should view the slight positive trend with cautious optimism. While the improvements are encouraging, they are not substantial enough to warrant major investment changes. Monitoring ongoing financial performance and market conditions is advised."
            ),
            "Strong Positive": (
                "* **Economic Context**: A strong positive correlation with Net Income signifies substantial improvements in overall profitability. This indicates that the company is achieving significant gains in net income due to robust business performance, effective strategies, or favorable economic conditions.\n"
                "* **Business Strategies**: Companies showing a strong positive correlation should focus on scaling successful strategies, capitalizing on favorable market conditions, and further improving operational efficiencies to sustain high levels of profitability.\n"
                "* **Financial Environment**: A strong positive trend reflects excellent financial conditions, with significant gains in net income. This may result from increased revenues, cost reductions, or advantageous market conditions. Companies are likely experiencing improved financial health.\n"
                "* **Global Conditions**: Globally, this trend could be driven by strong economic growth or favorable conditions in key markets. Companies should leverage international opportunities and favorable global conditions to enhance their profitability.\n"
                "* **Strategy for Investors**: Investors should consider increasing their investments in companies with strong positive correlations in Net Income. Significant profitability growth is a positive sign, suggesting strong business performance and potential for high returns."
            ),
            "Neutral": (
                "* **Economic Context**: A neutral correlation with Net Income suggests that changes in net income are not significantly impacted by broader economic conditions. This indicates stable profitability with no major fluctuations in performance.\n"
                "* **Business Strategies**: Companies with a neutral correlation should focus on maintaining steady operational performance. Strategies should emphasize cost management and operational stability to ensure consistent profitability.\n"
                "* **Financial Environment**: The financial environment is stable with no major impacts on Net Income. Companies may experience steady financial performance but should remain vigilant for any emerging trends or changes.\n"
                "* **Global Conditions**: A neutral trend globally indicates stability in net income with no significant influences from international markets. Companies should monitor global conditions but expect consistent performance.\n"
                "* **Strategy for Investors**: Investors should adopt a steady approach. While the neutral trend indicates stability, it is important to remain informed about potential changes in the economic environment that could affect profitability."
            ),
            "Slight Negative": (
                "* **Economic Context**: A slight negative correlation with Net Income indicates minor declines in profitability. This suggests that the company is experiencing slight reductions in net income, possibly due to increased costs or decreased revenues.\n"
                "* **Business Strategies**: Companies with a slight negative correlation should focus on addressing minor operational issues, controlling costs, and enhancing revenue streams. Small adjustments can help mitigate the negative impact on profitability.\n"
                "* **Financial Environment**: A slight negative trend in Net Income suggests minor financial pressures. Companies may face slight challenges in maintaining profitability and should manage their financial resources carefully.\n"
                "* **Global Conditions**: Globally, this trend might be influenced by minor economic challenges or market pressures. Companies should stay informed about international conditions that could affect their net income.\n"
                "* **Strategy for Investors**: Investors should be cautious with companies showing slight negative correlations in Net Income. Evaluating the company’s strategies for addressing profitability declines and monitoring financial performance is important."
            ),
            "Strong Negative": (
                "* **Economic Context**: A strong negative correlation with Net Income signifies significant declines in overall profitability. This suggests that the company is experiencing major reductions in net income due to severe operational or economic challenges.\n"
                "* **Business Strategies**: Companies with a strong negative correlation may need to implement comprehensive strategies to address substantial operational or financial difficulties. This could involve major cost-cutting measures, restructuring, or strategic pivots to improve profitability.\n"
                "* **Financial Environment**: A strong negative trend indicates challenging financial conditions, such as high costs or significant revenue declines. Companies need robust financial strategies to manage these challenges and improve their net income.\n"
                "* **Global Conditions**: Globally, this trend could be exacerbated by economic downturns or adverse market conditions. Companies should focus on risk management and explore strategies to navigate international challenges.\n"
                "* **Strategy for Investors**: Investors should exercise extreme caution with companies showing strong negative correlations in Net Income. Reassessing investments, exploring alternative opportunities, and monitoring the company's recovery strategies closely is crucial."
            )
        },
        "correlation with Net Income Applicable to Common Share": {
            "Slight Positive": (
                "* **Economic Context**: A slight positive correlation with Net Income Applicable to Common Shares indicates minor improvements in the earnings attributed to common shareholders. This suggests that there are small but favorable changes in profitability per share, possibly due to better operational efficiency or cost management.\n"
                "* **Business Strategies**: Companies experiencing a slight positive correlation should focus on strategies that enhance earnings per share, such as increasing revenue, improving cost controls, or optimizing capital structure. Even small improvements in net income applicable to common shares can be beneficial.\n"
                "* **Financial Environment**: A slight positive trend in Net Income Applicable to Common Shares implies generally stable financial conditions with incremental gains in shareholder earnings. Companies should continue to leverage favorable market conditions or cost-saving measures to maintain this positive trend.\n"
                "* **Global Conditions**: Globally, a slight positive trend might reflect stable economic conditions or minor improvements in international markets. Companies can benefit from consistent global conditions that support better profitability per share.\n"
                "* **Strategy for Investors**: Investors should view a slight positive correlation with Net Income Applicable to Common Shares with cautious optimism. While the improvements are encouraging, they are not substantial enough to warrant major investment decisions. Monitoring ongoing performance and market conditions is advised."
            ),
            "Strong Positive": (
                "* **Economic Context**: A strong positive correlation with Net Income Applicable to Common Shares signifies substantial improvements in earnings per share. This suggests that the company is experiencing significant gains in net income that directly benefit common shareholders, due to robust business performance or favorable economic conditions.\n"
                "* **Business Strategies**: Companies showing a strong positive correlation should focus on scaling successful strategies, optimizing operations, and capitalizing on favorable market conditions to sustain high levels of profitability per share. Effective strategies and operational efficiencies are key drivers of this positive trend.\n"
                "* **Financial Environment**: A strong positive trend reflects excellent financial conditions with significant gains in earnings per share. This may result from increased revenues, reduced costs, or advantageous financial conditions. Companies are likely experiencing improved financial health and shareholder value.\n"
                "* **Global Conditions**: Globally, this trend might be influenced by strong economic growth or favorable international market conditions. Companies should leverage international opportunities to enhance their profitability per share and maximize shareholder returns.\n"
                "* **Strategy for Investors**: Investors should consider increasing their investments in companies with strong positive correlations in Net Income Applicable to Common Shares. Significant improvements in earnings per share indicate strong business performance and potential for high returns."
            ),
            "Neutral": (
                "* **Economic Context**: A neutral correlation with Net Income Applicable to Common Shares suggests that changes in earnings per share are not significantly influenced by broader economic conditions. This indicates stable profitability with no major fluctuations in shareholder earnings.\n"
                "* **Business Strategies**: Companies with a neutral correlation should focus on maintaining steady performance and profitability per share. Strategies should emphasize cost management, operational stability, and consistent financial performance.\n"
                "* **Financial Environment**: The financial environment is stable with no major impacts on earnings per share. Companies may experience consistent financial performance but should remain vigilant for any emerging trends or changes.\n"
                "* **Global Conditions**: A neutral trend globally suggests stability in net income applicable to common shares, with no significant influences from international markets. Companies should monitor global conditions but expect consistent performance.\n"
                "* **Strategy for Investors**: Investors should adopt a steady approach. While the neutral trend indicates stability, it is important to remain informed about potential changes in the economic environment that could affect shareholder earnings."
            ),
            "Slight Negative": (
                "* **Economic Context**: A slight negative correlation with Net Income Applicable to Common Shares indicates minor declines in earnings per share. This suggests that the company is experiencing slight reductions in net income available to common shareholders, potentially due to increased costs or decreased revenues.\n"
                "* **Business Strategies**: Companies with a slight negative correlation should focus on addressing minor operational issues, controlling costs, and improving revenue streams. Small adjustments can help mitigate the negative impact on earnings per share.\n"
                "* **Financial Environment**: A slight negative trend suggests minor financial pressures affecting earnings per share. Companies should manage their financial resources carefully to address the decline in profitability per share.\n"
                "* **Global Conditions**: Globally, this trend might be influenced by minor economic challenges or market pressures. Companies should stay informed about international conditions that could affect their earnings per share.\n"
                "* **Strategy for Investors**: Investors should be cautious with companies showing slight negative correlations in Net Income Applicable to Common Shares. Evaluating the company’s strategies for addressing profitability declines and monitoring financial performance is important."
            ),
            "Strong Negative": (
                "* **Economic Context**: A strong negative correlation with Net Income Applicable to Common Shares signifies significant declines in earnings per share. This indicates that the company is facing major reductions in net income available to common shareholders, possibly due to severe operational or economic challenges.\n"
                "* **Business Strategies**: Companies with a strong negative correlation may need to implement comprehensive strategies to address substantial declines in earnings per share. This could involve major cost-cutting measures, restructuring, or strategic pivots to improve profitability.\n"
                "* **Financial Environment**: A strong negative trend indicates challenging financial conditions, such as high costs or significant revenue declines. Companies need robust financial strategies to manage these challenges and improve their earnings per share.\n"
                "* **Global Conditions**: Globally, this trend could be exacerbated by economic downturns or adverse market conditions. Companies should focus on risk management and explore strategies to navigate international challenges.\n"
                "* **Strategy for Investors**: Investors should exercise extreme caution with companies showing strong negative correlations in Net Income Applicable to Common Shares. Reassessing investments, exploring alternative opportunities, and monitoring the company's recovery strategies closely is crucial."
            )
        },
        "correlation with EPS (Earnings Per Share)": {
            "Slight Positive": (
                "* **Economic Context**: A slight positive correlation with EPS indicates minor improvements in earnings per share. This suggests that the company's earnings available to each shareholder have experienced small but positive changes, potentially due to improved business operations or cost management.\n"
                "* **Business Strategies**: Companies with a slight positive correlation should focus on strategies that enhance EPS, such as increasing revenue, improving efficiency, or reducing costs. Even small gains in EPS can be beneficial for shareholder value.\n"
                "* **Financial Environment**: A slight positive trend in EPS implies generally favorable financial conditions with incremental gains in shareholder earnings. Companies should continue to optimize their performance to maintain this positive trend.\n"
                "* **Global Conditions**: Globally, a slight positive correlation in EPS may reflect stable or slightly improving economic conditions. Companies can benefit from consistent global conditions that support better earnings per share.\n"
                "* **Strategy for Investors**: Investors should view a slight positive correlation with EPS with cautious optimism. While the improvements are positive, they are not substantial enough to make significant investment decisions. It’s advisable to monitor ongoing performance and economic conditions."
            ),
            "Strong Positive": (
                "* **Economic Context**: A strong positive correlation with EPS signifies substantial improvements in earnings per share. This suggests that the company is experiencing significant gains in EPS, likely due to strong operational performance, increased revenues, or favorable economic conditions.\n"
                "* **Business Strategies**: Companies showing a strong positive correlation should leverage successful strategies to continue improving EPS. This may involve scaling up successful operations, exploring new markets, or enhancing financial management to sustain high levels of EPS.\n"
                "* **Financial Environment**: A strong positive trend in EPS indicates excellent financial conditions with substantial gains in shareholder earnings. Companies are likely benefiting from favorable market conditions, effective cost controls, or increased revenues.\n"
                "* **Global Conditions**: Globally, this trend may be influenced by robust economic growth or favorable international market conditions. Companies should capitalize on international opportunities to further enhance their EPS.\n"
                "* **Strategy for Investors**: Investors should consider increasing their investments in companies with strong positive correlations in EPS. Significant improvements in EPS suggest strong business performance and potential for high returns."
            ),
            "Neutral": (
                "* **Economic Context**: A neutral correlation with EPS suggests that earnings per share are stable and not significantly influenced by broader economic conditions. This indicates that EPS has remained consistent without major fluctuations.\n"
                "* **Business Strategies**: Companies with a neutral correlation should focus on maintaining steady performance. Strategies should involve cost management and consistent operational practices to ensure stable EPS.\n"
                "* **Financial Environment**: The financial environment is stable with no major impacts on EPS. Companies should continue to monitor their financial health and remain prepared for any potential changes.\n"
                "* **Global Conditions**: A neutral trend globally suggests that international conditions are stable and not significantly affecting EPS. Companies can expect consistent performance without substantial changes.\n"
                "* **Strategy for Investors**: Investors should adopt a wait-and-see approach. While the EPS trend is stable, it’s important to stay informed about any potential changes in the economic or market environment that could impact future earnings."
            ),
            "Slight Negative": (
                "* **Economic Context**: A slight negative correlation with EPS indicates minor declines in earnings per share. This suggests that the company is experiencing slight reductions in EPS, possibly due to increased costs or slight decreases in revenues.\n"
                "* **Business Strategies**: Companies should focus on addressing minor issues affecting EPS. This could involve optimizing operational efficiencies, managing costs more effectively, or improving revenue streams to counteract the slight decline.\n"
                "* **Financial Environment**: A slight negative trend in EPS suggests minor financial pressures impacting earnings per share. Companies should carefully manage their financial resources to mitigate the negative effects.\n"
                "* **Global Conditions**: Globally, this trend might reflect minor economic challenges or market pressures affecting EPS. Companies should stay informed about international developments that could influence their earnings per share.\n"
                "* **Strategy for Investors**: Investors should be cautious with companies showing slight negative correlations in EPS. Reviewing the company's strategies for addressing the decline and monitoring financial performance is important."
            ),
            "Strong Negative": (
                "* **Economic Context**: A strong negative correlation with EPS signifies significant declines in earnings per share. This indicates that the company is facing major reductions in EPS, likely due to severe operational challenges or adverse economic conditions.\n"
                "* **Business Strategies**: Companies with a strong negative correlation need to implement comprehensive strategies to address substantial declines in EPS. This may involve significant restructuring, cost-cutting measures, or strategic pivots to improve earnings per share.\n"
                "* **Financial Environment**: A strong negative trend in EPS reflects challenging financial conditions with severe impacts on earnings per share. Companies need robust financial strategies to manage these challenges and improve EPS.\n"
                "* **Global Conditions**: Globally, this trend could be exacerbated by economic downturns, adverse market conditions, or geopolitical issues. Companies should focus on risk management and explore strategies to mitigate the global challenges affecting their EPS.\n"
                "* **Strategy for Investors**: Investors should exercise extreme caution with companies showing strong negative correlations in EPS. Reassessing investments, exploring alternative opportunities, and closely monitoring the company's recovery strategies is crucial."
            )
        }
    }
    return interpretations.get(parameter_name, {}).get(correlation_interpretation, "No interpretation available.")

# Function to get the latest financial data
def get_latest_financial_data(stock_name):
    if stock_name in financial_data:
        stock_financial_data = financial_data[stock_name]
        balance_sheet = stock_financial_data.get('BalanceSheet', pd.DataFrame())
        income_statement = stock_financial_data.get('IncomeStatement', pd.DataFrame())
        cash_flow = stock_financial_data.get('CashFlow', pd.DataFrame())
        
        latest_balance_sheet = balance_sheet[balance_sheet['Date'] == 'Dec 2023'].iloc[-1] if not balance_sheet.empty else pd.Series()
        latest_income_statement = income_statement[income_statement['Date'] == 'Jun 2024'].iloc[-1] if not income_statement.empty else pd.Series()
        latest_cash_flow = cash_flow[cash_flow['Date'] == 'Dec 2023'].iloc[-1] if not cash_flow.empty else pd.Series()
        
        return latest_balance_sheet, latest_income_statement, latest_cash_flow
    else:
        return pd.Series(), pd.Series(), pd.Series()

# Streamlit App Interface
st.title('Industry and Financial Data Prediction')
st.sidebar.header('Select Options')

selected_industry = st.sidebar.selectbox(
    'Select Industry',
    list(indicators.keys()),  # Use the keys from the indicators dictionary
    index=0  # Default index if needed, here 'Manufacture of Food Products'
)

if selected_industry:
    # Normalize and match the industry name with sheet names
    normalized_industry = selected_industry.strip().lower()
    matched_sheet_name = None
    
    for sheet_name in synthetic_data.keys():
        if sheet_name.strip().lower() == normalized_industry:
            matched_sheet_name = sheet_name
            break
    
    if matched_sheet_name:
        st.header(f'Industry: {selected_industry}')
        
        if selected_industry in indicators:
            # Prepare Data for Modeling
            def prepare_data(industry, data, iip_data):
                leading_indicators = indicators[industry]['Leading']
                
                X = data[leading_indicators].shift(1).dropna()
                y = iip_data[industry].loc[X.index]
                return X, y
            
            X, y = prepare_data(selected_industry, synthetic_data[matched_sheet_name], iip_data)
            
            # Regression Model
            reg_model = LinearRegression()
            reg_model.fit(X, y)
            reg_pred = reg_model.predict(X)

            # ARIMA Model
            arima_model = ARIMA(y, order=(5, 1, 0))  # Adjust order parameters as needed
            arima_result = arima_model.fit()
            arima_pred = arima_result.predict(start=1, end=len(y), dynamic=False)

            # Machine Learning Model (Random Forest)
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            rf_pred = rf_model.predict(X)

            # Model Comparison
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

            input_data = {}
            for indicator in indicators[selected_industry]['Leading']:
                input_data[indicator] = st.number_input(f'Expected {indicator} Value:', value=float(X[indicator].iloc[-1]))

            input_df = pd.DataFrame(input_data, index=[0])

            # Predictions
            future_reg_pred = reg_model.predict(input_df)
            future_rf_pred = rf_model.predict(input_df)

            st.write(f"Linear Regression Prediction: {future_reg_pred[0]:.2f}")
            st.write(f"Random Forest Prediction: {future_rf_pred[0]:.2f}")

            # Display Latest Data
            st.subheader('Industry and Indicator Data')

                        # Industry Data
            latest_industry_data = iip_data[[selected_industry]].tail()  # Show the last few rows
            st.write('**Industry Data:**')
            st.write(latest_industry_data)

            # Leading Indicators Data
            latest_leading_indicators_data = synthetic_data[matched_sheet_name][indicators[selected_industry]['Leading']].tail()
            st.write('**Leading Indicators Data:**')
            st.write(latest_leading_indicators_data)

            # Lagging Indicators Data
            latest_lagging_indicators_data = synthetic_data[matched_sheet_name][indicators[selected_industry]['Lagging']].tail()
            st.write('**Lagging Indicators Data:**')
            st.write(latest_lagging_indicators_data)

            # Stock Selection and Correlation Analysis
            if correlation_results is not None:
                # Allow multiple stock selection
                selected_stocks = st.sidebar.multiselect('Select Stocks', correlation_results['Stock Name'].tolist())

                if selected_stocks:
                    # Filter correlation results for selected stocks
                    selected_corr_data = correlation_results[correlation_results['Stock Name'].isin(selected_stocks)]
                    
                    # Initialize a DataFrame to store adjusted correlations
                    all_adjusted_corr_data = []

                    for stock in selected_stocks:
                        st.subheader(f'Correlation Analysis with {stock}')
                        
                        # Fetch correlation data for the selected stock
                        stock_correlation_data = selected_corr_data[selected_corr_data['Stock Name'] == stock]

                        if not stock_correlation_data.empty:
                            st.write('**Actual Correlation Results:**')
                            st.write(stock_correlation_data)

                            # Prepare predicted correlation data
                            st.subheader('Predicted Correlation Analysis')

                            industry_mean = y.mean()
                            updated_corr_data = stock_correlation_data.copy()
                            updated_corr_data['Predicted Industry Value'] = future_reg_pred[0]

                            for col in [
                                'correlation with Total Revenue/Income',
                                'correlation with Net Income',
                                'correlation with Total Operating Expense',
                                'correlation with Operating Income/Profit',
                                'correlation with EBITDA',
                                'correlation with EBIT',
                                'correlation with Income/Profit Before Tax',
                                'correlation with Net Income From Continuing Operation',
                                'correlation with Net Income Applicable to Common Share',
                                'correlation with EPS (Earning Per Share)',
                                'correlation with Operating Margin',
                                'correlation with EBITDA Margin',
                                'correlation with Net Profit Margin',
                                'Annualized correlation with Total Revenue/Income',
                                'Annualized correlation with Total Operating Expense',
                                'Annualized correlation with Operating Income/Profit',
                                'Annualized correlation with EBITDA',
                                'Annualized correlation with EBIT',
                                'Annualized correlation with Income/Profit Before Tax',
                                'Annualized correlation with Net Income From Continuing Operation',
                                'Annualized correlation with Net Income',
                                'Annualized correlation with Net Income Applicable to Common Share',
                                'Annualized correlation with EPS (Earning Per Share)'
                            ]:
                                if col in updated_corr_data.columns:
                                    updated_corr_data[f'Interpreted {col}'] = updated_corr_data[col].apply(interpret_correlation)
                                    updated_corr_data[f'Adjusted {col}'] = updated_corr_data.apply(
                                        lambda row: adjust_correlation(row[col], row['Predicted Industry Value'], industry_mean),
                                        axis=1
                                    )
                                
                            all_adjusted_corr_data.append(updated_corr_data)
                        
                    # Combine all adjusted correlation data
                    if all_adjusted_corr_data:
                        combined_corr_data = pd.concat(all_adjusted_corr_data, ignore_index=True)

                        st.write('**Predicted Correlation Results:**')
                        st.write(combined_corr_data[['Stock Name', 'Predicted Industry Value'] +
                                                    [col for col in combined_corr_data.columns if 'Adjusted' in col]])

                        # Interactive Comparison Chart
                        st.subheader('Interactive Comparison of Actual and Predicted Correlation Results')

                        # Preparing data for plotting
                        actual_corr_cols = [col for col in correlation_results.columns if 'correlation' in col]
                        predicted_corr_cols = [f'Adjusted {col}' for col in actual_corr_cols if f'Adjusted {col}' in combined_corr_data.columns]

                        fig = go.Figure()
                        for col in actual_corr_cols:
                            if col in selected_corr_data.columns:
                                fig.add_trace(go.Bar(
                                    x=selected_corr_data['Stock Name'],
                                    y=selected_corr_data[col],
                                    name=f'Actual {col}',
                                    marker_color='blue'
                                ))

                        for col in predicted_corr_cols:
                            if col in combined_corr_data.columns:
                                fig.add_trace(go.Bar(
                                    x=combined_corr_data['Stock Name'],
                                    y=combined_corr_data[col],
                                    name=f'Predicted {col}',
                                    marker_color='orange'
                                ))

                        fig.update_layout(
                            title='Comparison of Actual and Predicted Correlation Results',
                            xaxis_title='Stock',
                            yaxis_title='Correlation Value',
                            barmode='group',
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Calculate Predicted Income Statement Results
                        st.subheader('Predicted Income Statement Results')

                        # Initialize a DataFrame to store predicted results for each stock
                        all_predicted_results = []

                        for stock in selected_stocks:
                            income_statement_data = financial_data.get(stock, {}).get('IncomeStatement', pd.DataFrame())
                            income_statement_data = income_statement_data[income_statement_data['Date'] == 'Jun 2024'].iloc[-1] if not income_statement_data.empty else pd.Series()
                            
                            if not income_statement_data.empty:
                                income_statement_values = income_statement_data.dropna()
                                income_statement_dict = income_statement_values.to_dict()
                                
                                # Create a DataFrame for correlation and income statement data
                                corr_cols = [
                                    'Adjusted correlation with Total Revenue/Income',
                                    'Adjusted correlation with Net Income',
                                    'Adjusted correlation with Total Operating Expense',
                                    'Adjusted correlation with Operating Income/Profit',
                                    'Adjusted correlation with EBITDA',
                                    'Adjusted correlation with EBIT',
                                    'Adjusted correlation with Income/Profit Before Tax',
                                    'Adjusted correlation with Net Income From Continuing Operation',
                                    'Adjusted correlation with Net Income Applicable to Common Share',
                                    'Adjusted correlation with EPS (Earning Per Share)'
                                ]
                                
                                # Initialize a dictionary to store the predicted results
                                predicted_results = {}
                                
                                for col in corr_cols:
                                    if col in combined_corr_data.columns:
                                        correlation_value = combined_corr_data[combined_corr_data['Stock Name'] == stock].iloc[0].get(col, 0)
                                        statement_value = income_statement_dict.get(col.replace('Adjusted correlation with ', ''), 0)
                                        predicted_results[col.replace('Adjusted correlation with ', '')] = (statement_value * correlation_value) + statement_value
                                
                                # Convert the predicted results dictionary to a DataFrame
                                predicted_results_df = pd.DataFrame(predicted_results, index=[f'Predicted Income Statement Result ({stock})']).T
                                all_predicted_results.append(predicted_results_df)
                                
                                # Display Latest Financial Data
                                st.write(f"**Latest Financial Data for {stock}:**")
                                if not income_statement_data.empty:
                                    st.write("**Income Statement (Jun 2024):**")
                                    st.write(income_statement_data)
                                
                                # Display the full predicted results
                                st.write(f"**Predicted Income Statement Results for {stock}:**")
                                st.write(predicted_results_df)
                        
                        # Combine all predicted results
                        if all_predicted_results:
                            combined_predicted_results = pd.concat(all_predicted_results)
                            st.write('**All Predicted Income Statement Results:**')
                            st.write(combined_predicted_results)

                            # Display Detailed Interpretation with Indian Economy Context
                            st.write('**Detailed Interpretation with Indian Economy Context:**')
                            
                            parameters = [
                                'Total Revenue/Income',
                                'Total Operating Expense',
                                'Operating Income/Profit',
                                'EBITDA',
                                'EBIT',
                                'Income/Profit Before Tax',
                                'Net Income From Continuing Operation',
                                'Net Income',
                                'Net Income Applicable to Common Share',
                                'EPS (Earning Per Share)'
                            ]

                            for parameter in parameters:
                                for stock in selected_stocks:
                                    correlation_value = combined_corr_data[
                                        combined_corr_data['Stock Name'] == stock
                                    ].iloc[0].get(f'Interpreted correlation with {parameter}', "Neutral")
                                    
                                    interpretation = get_detailed_interpretation(f'correlation with {parameter}', correlation_value)
                                    st.write(f"**{stock} - correlation with {parameter}:**")
                                    st.write(interpretation)

else:
    st.error('No correlation results available.')

           
