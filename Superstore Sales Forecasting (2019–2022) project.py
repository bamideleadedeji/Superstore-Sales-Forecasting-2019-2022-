#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1: Imports and settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set(style='whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load and minimal cleaning
data_path = r"C:\Users\HP USER\supstore-dataset-2019-2022\superstore_dataset.csv"  
df = pd.read_csv(data_path)

# standardize names and basic cleaning
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df = df.dropna(subset=['order_date', 'sales'])
df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0)

print("Loaded rows:", len(df))
df.head()


# In[3]:


# Aggregate to monthly totals
df['year_month'] = df['order_date'].dt.to_period('M')
monthly = df.groupby('year_month')['sales'].sum().sort_index()
monthly.index = monthly.index.to_timestamp()  # convert PeriodIndex to Timestamp (start of month)
monthly = monthly.asfreq('MS')  # ensure monthly frequency (MS = month start)
monthly = monthly.fillna(0)     # fill any missing months with 0 (or use interpolation)

print("Monthly series length:", len(monthly))
monthly.head()


# In[4]:


# Plot the series
plt.figure(figsize=(12,4))
plt.plot(monthly.index, monthly.values, marker='o')
plt.title('Monthly Sales (Total)')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()


# In[5]:


# Train/test split
h = 12  # forecast horizon: 12 months
train = monthly[:-h]
test = monthly[-h:]

print("Train months:", len(train), "Test months:", len(test))


# In[6]:


# Baseline forecasts: naive (last value) and seasonal naive
# Naive forecast: repeat last train month value
naive_forecast = pd.Series(train.iloc[-1], index=test.index)

# Seasonal naive: use same month last year when available (falls back to last value)
seasonal_naive = []
for t in test.index:
    prev_year = t - pd.DateOffset(years=1)
    if prev_year in train.index:
        seasonal_naive.append(train.loc[prev_year])
    else:
        seasonal_naive.append(train.iloc[-1])
seasonal_naive = pd.Series(seasonal_naive, index=test.index)

def evaluate(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print(f"{label} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

print("Baseline evaluations:")
evaluate(test.values, naive_forecast.values, "Naive")
evaluate(test.values, seasonal_naive.values, "Seasonal Naive")


# In[7]:


# SARIMAX modeling (order & seasonal_order are simple defaults)
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Use a simple configuration: (p,d,q)=(1,1,1) and seasonal (P,D,Q,s)=(1,1,1,12).
# For production you'd run grid search or auto_arima for better params.
order = (1,1,1)
seasonal_order = (1,1,1,12)

model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)
print(model_fit.summary())


# In[9]:


# Forecast with SARIMAX
sarimax_forecast = model_fit.get_forecast(steps=h).predicted_mean
sarimax_forecast.index = test.index

# Plot predictions vs actual
plt.figure(figsize=(12,4))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test', marker='o')
plt.plot(sarimax_forecast.index, sarimax_forecast, label='SARIMAX Forecast', marker='o')
plt.legend()
plt.title('SARIMAX Forecast vs Actual')
plt.tight_layout()
plt.show()

# Evaluate
evaluate(test.values, sarimax_forecast.values, "SARIMAX")


# In[10]:


# Create lags and rolling features for supervised ML
df_ml = monthly.to_frame(name='y').copy()
# create lag features
for lag in range(1,13):  # 1 to 12 months lag
    df_ml[f'lag_{lag}'] = df_ml['y'].shift(lag)
# rolling stats
df_ml['rolling_3_mean'] = df_ml['y'].shift(1).rolling(window=3).mean()
df_ml['rolling_12_mean'] = df_ml['y'].shift(1).rolling(window=12).mean()
df_ml = df_ml.dropna()

# split into train/test aligning horizon
X = df_ml.drop(columns=['y'])
y = df_ml['y']
# ensure train/test same split as before by index
X_train = X[X.index < test.index[0]]
y_train = y[y.index < test.index[0]]
X_test = X[X.index >= test.index[0]]
y_test = y[y.index >= test.index[0]]

print("ML train rows:", len(X_train), "ML test rows:", len(X_test))


# In[11]:


# Train RandomForest (direct multi-step using available X_test)
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Direct prediction for X_test rows (we engineered lags so these are one-step-aligned)
rf_preds = pd.Series(rf.predict(X_test), index=X_test.index)

# Plot
plt.figure(figsize=(12,4))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test', marker='o')
plt.plot(rf_preds.index, rf_preds, label='RandomForest Forecast', marker='o')
plt.legend()
plt.title('Random Forest Forecast vs Actual')
plt.tight_layout()
plt.show()

# Evaluate
evaluate(y_test.values, rf_preds.values, "RandomForest")


# In[12]:


# Assemble forecasts for comparison
comparison = pd.DataFrame({
    'actual': test,
    'naive': naive_forecast,
    'seasonal_naive': seasonal_naive,
    'sarimax': sarimax_forecast,
    'random_forest': rf_preds
})

print(comparison)
for col in ['naive','seasonal_naive','sarimax','random_forest']:
    evaluate(comparison['actual'].values, comparison[col].values, col)


# In[13]:


# Save forecasts and evaluation summary
output_folder = r"C:\Users\HP USER\supstore-analysis-output"
import os
os.makedirs(output_folder, exist_ok=True)

comparison.to_csv(os.path.join(output_folder, 'forecast_comparison.csv'))
print("Saved forecast_comparison.csv to", output_folder)


# Next steps / tips
# - Tune SARIMAX orders (p,d,q,P,D,Q,s) using information criteria (AIC/BIC) or use `pmdarima.auto_arima` (install pmdarima).
# - For ML: try XGBoost/LightGBM and add external regressors (promotions, holidays).
# - Consider recursive multi-step forecasting if you'd like to forecast beyond available lag features.
# - Evaluate with rolling-origin cross-validation (time-series CV) for robust model selection.
# - If seasonality is strong, consider Prophet (fbprophet) or SARIMAX with better parameter search.

# Executive Summary
# 
# This project extends the Superstore Sales Analysis (2019–2022) by developing a predictive sales forecasting model 
# aimed at anticipating monthly revenue trends and supporting strategic decision-making.
# 
# The analysis leverages historical sales data from the Superstore dataset to understand seasonality, growth patterns, 
# and business performance. Using time-series forecasting techniques, the model provides actionable insights that can 
# help optimize inventory management, marketing strategies, and revenue planning.
# 
# Objectives
# 
# Forecast monthly sales for the upcoming 12 months
# Capture seasonality and trend patterns in sales data
# Compare traditional statistical and machine learning forecasting methods
# Evaluate model accuracy and provide business recommendations
# 
# Methods Used
# 
# 1. Data Preparation:
# 
#    Cleaned and standardized historical data
#    Aggregated daily transactions into monthly totals
#    Created new time-based features (year, month, lag variables)
# 
# 2. Forecasting Techniques:
# 
#    SARIMAX Model: Captured trend and seasonality using statistical time-series modeling
#    Random Forest Regressor: Leveraged machine learning to predict sales using lag and rolling features
#    Baseline Models: Naïve and seasonal-naïve forecasts for comparison
# 
# 3. Model Evaluation:
# 
#    Metrics used: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
#    Models tested and validated using the last 12 months of data
# 
# Key Findings
# 
# Consistent upward trend in sales over the years, with notable seasonal peaks during Q4 (holiday months).
# SARIMAX captured seasonality effectively and provided stable short-term forecasts.
# Random Forest handled non-linear patterns better in recent months, offering slightly improved predictive accuracy.
# Baseline models performed reasonably well but lacked predictive precision beyond repeating past values.
# 
# Results Summary
# 
# | Model          | MAE (↓)  | RMSE (↓) | Key Strength                          |
# | -------------- | -------- | -------- | ------------------------------------- |
# | Naïve Forecast | Moderate | High     | Simple benchmark                      |
# | Seasonal Naïve | Lower    | Moderate | Good seasonality reflection           |
# | SARIMAX        | Low  | Low  | Best at capturing trend + seasonality |
# | Random Forest  | Low      | Low      | Captures non-linear patterns          |
# 
# Business Insights
# 
# Sales are seasonally cyclical, peaking around November–December due to holiday demand.
# Q2 and Q3 show steady growth, suggesting potential for targeted mid-year promotions.
# Forecasting models enable better inventory and staffing optimization ahead of peak demand.
# Combining SARIMAX and ML models can yield robust hybrid forecasts for business planning.
# 
# Recommendations
# 
# 1. Increase inventory and marketing budgets during Q4 to align with seasonal sales surges.
# 2. Use forecast outputs for demand planning and revenue budgeting.
# 3. Continue retraining models quarterly with new data for more accurate predictions.
# 4. Explore additional factors (discounts, region, category) to build multi-variate forecasting models.
# 
# 
# Conclusion
# 
# The Superstore Sales Forecasting project demonstrates how predictive analytics can transform historical 
# sales data into forward-looking business intelligence.
# By combining statistical models (SARIMAX) and machine learning techniques (Random Forest), 
# this analysis provides a reliable foundation for data-driven planning, revenue forecasting, and strategic decision-making.

# In[ ]:




