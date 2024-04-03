# Databricks notebook source
# MAGIC %md
# MAGIC # Previsão de Demanda

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

import os
import holidays
import mlflow
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor



# COMMAND ----------

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)
%config InlineBackend.figure_format = 'retina'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aux functions

# COMMAND ----------

def is_national_holiday(date):
    """
    Function to get all dates that are national holidays in US
    """
    national_holidays = holidays.CountryHoliday('US')
    return 1 if date in national_holidays else 0

# COMMAND ----------

def is_state_holiday(date):
    state_holidays = ['07-19', '03-04', '03-29']
    return 1 if date in state_holidays else 0

# COMMAND ----------

# MAGIC %md
# MAGIC # Train Models

# COMMAND ----------

# Load datasets
df_train_transformed = pd.read_csv(os.getcwd() + '/data/02-interim/df_train_transformed.csv', index_col=0)
df_valid_transformed = pd.read_csv(os.getcwd() + '/data/02-interim/df_valid_transformed.csv', index_col=0)

# Separate target and features
X_train = df_train_transformed.drop('qty', axis=1)
y_train = df_train_transformed['qty']
X_valid = df_valid_transformed.drop('qty', axis=1)
y_valid = df_valid_transformed['qty']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear regression

# COMMAND ----------

X_train_lr = X_train.loc[:, [f'qty_D-{i+1}' for i in range(14)]]
y_train_lr = y_train.copy()

X_valid_lr = X_valid.loc[:, [f'qty_D-{i+1}' for i in range(14)]]
y_valid_lr = y_valid.copy()

# COMMAND ----------

def make_multistep_target(ts, steps):
    """Creating a 14 days forward for target variable y (=qty)
    """
    return pd.concat(
        {f'y_D+{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)

# COMMAND ----------

# Two-week forecast
y_train_lr = make_multistep_target(y_train_lr, steps=14).dropna()
y_valid_lr = make_multistep_target(y_valid_lr, steps=14).dropna()

# Shifting has created indexes that don't match. Only keep times for which we have both targets and features.
y_train_lr, X_train_lr = y_train_lr.align(X_train_lr, join='inner', axis=0)
y_valid_lr, X_valid_lr = y_valid_lr.align(X_valid_lr, join='inner', axis=0)

# COMMAND ----------

with mlflow.start_run(run_name='Linear_Regression_Daily') as run:
    tags = {
        "features": "grouped daily",
        "scaling": "None",
        "strategy": "model predict next 14 days all at once"
       }
    # Define model
    model_lr = LinearRegression()
    model_name = 'Linear Regression'

    #Training
    model_lr.fit(X_train_lr, y_train_lr)
    y_fit_lr = pd.DataFrame(model_lr.predict(X_train_lr), index=y_train_lr.index, columns=y_train_lr.columns)
    y_pred_lr = pd.DataFrame(model_lr.predict(X_valid_lr), index=y_valid_lr.index, columns=y_valid_lr.columns)

    mlflow.sklearn.log_model(model_lr, artifact_path="lr_model")

    # Performance metrics
    mae = mean_absolute_error( y_train_lr, y_fit_lr )
    mape = mean_absolute_percentage_error( y_train_lr, y_fit_lr )
    rmse = np.sqrt( mean_squared_error( y_train_lr, y_fit_lr ) )

    mae_val = mean_absolute_error( y_valid_lr, y_pred_lr )
    mape_val = mean_absolute_percentage_error( y_valid_lr, y_pred_lr )
    rmse_val = np.sqrt( mean_squared_error( y_valid_lr, y_pred_lr ) )

    df_metrics_lr = pd.DataFrame( { 'Model Name': model_name, 
                    'MAE': mae, 
                    'MAPE': mape,
                    'RMSE': rmse }, index=[0] )

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("rmse", rmse)
    
    mlflow.log_metric("mae_val", mae_val)
    mlflow.log_metric("mape_val", mape_val)
    mlflow.log_metric("rmse_val", rmse_val)
    
    mlflow.set_tags(tags)
    
    print("rmse: {}".format(rmse))
    print("mae: {}".format(mae))
    print("MAPE:", round(mape, 2), "%")
    
    print("rmse_val: {}".format(rmse_val))
    print("mae_val: {}".format(mae_val))
    print("MAPE_val:", round(mape_val, 2), "%")

    mlflow.end_run()

# COMMAND ----------

# # Plot results
# ax = y_train_lr.plot(**plot_params)
# ax = y_valid_lr.plot(**plot_params)
# ax = y_fit_lr.plot(ax=ax)
# _ = y_pred_lr.plot(ax=ax, color='C3')

# COMMAND ----------

# MAGIC %md
# MAGIC ## XGBoost

# COMMAND ----------

year_freq_dict = {2019: 717, 2020: 1068, 2021: 1095, 2022: 1095, 2023: 912}

# COMMAND ----------

N_ESTIMATORS = 300, 
ETA = 0.01, 
MAX_DEPTH = 15,
SUBSAMPLE = 0.7,
COLSAMPLE_BYTREE = 0.8

# COMMAND ----------

with mlflow.start_run(run_name='XGBoost_Recursively_Daily') as run:
    tags = {
        "features": "grouped daily",
        "scaling": "None",
        "strategy": "model run 14 times to predict next day with the previous predict day"
       }

    # Recursive time series strategy
    forecast = []

    # Train XGBoost model and make forecasts recursively
    for i in range(14):
        print(f"Prediction nº {i+1}/14")

        N_ESTIMATORS = 500
        ETA = 0.01
        MAX_DEPTH = 15
        SUBSAMPLE = 0.7
        COLSAMPLE_BYTREE = 0.8

        params = {
        'objective': 'reg:squarederror',
        'n_estimators': N_ESTIMATORS,
        'eta': ETA,
        'max_depth': MAX_DEPTH,
        'subsample': SUBSAMPLE,
        'colsample_bytree': COLSAMPLE_BYTREE
        }
        model_xgb = xgb.XGBRegressor( **params)

        model_xgb.fit(X_train, y_train)
        # Forecast for the next day
        day_to_forecast = pd.to_datetime(X_train.index[-1]) + pd.Timedelta(days=1)
        X_train_forecast = pd.DataFrame(
            {
                'day_step': [X_train['day_step'].iloc[-1] + 1], 
                'qty_D-1': [y_train[-1]],
                'day_step_D-1': [X_train['day_step'].iloc[-1]], 
                'qty_D-2': [X_train['qty_D-1'].iloc[-1]],
                'day_step_D-2': [X_train['day_step_D-1'].iloc[-1]],  
                'qty_D-3': [X_train['qty_D-2'].iloc[-1]], 
                'day_step_D-3': [X_train['day_step_D-2'].iloc[-1]], 
                'qty_D-4': [X_train['qty_D-3'].iloc[-1]], 
                'day_step_D-4': [X_train['day_step_D-3'].iloc[-1]],
                'qty_D-5': [X_train['qty_D-4'].iloc[-1]], 
                'day_step_D-5': [X_train['day_step_D-4'].iloc[-1]],
                'qty_D-6': [X_train['qty_D-5'].iloc[-1]], 
                'day_step_D-6': [X_train['day_step_D-5'].iloc[-1]],
                'qty_D-7': [X_train['qty_D-6'].iloc[-1]], 
                'day_step_D-7': [X_train['day_step_D-6'].iloc[-1]], 
                'qty_D-8': [X_train['qty_D-7'].iloc[-1]], 
                'day_step_D-8': [X_train['day_step_D-7'].iloc[-1]], 
                'qty_D-9': [X_train['qty_D-8'].iloc[-1]], 
                'day_step_D-9': [X_train['day_step_D-8'].iloc[-1]], 
                'qty_D-10': [X_train['qty_D-9'].iloc[-1]], 
                'day_step_D-10': [X_train['day_step_D-9'].iloc[-1]], 
                'qty_D-11': [X_train['qty_D-10'].iloc[-1]], 
                'day_step_D-11': [X_train['day_step_D-10'].iloc[-1]],
                'qty_D-12': [X_train['qty_D-12'].iloc[-1]], 
                'day_step_D-12': [X_train['day_step_D-12'].iloc[-1]], 
                'qty_D-13': [X_train['qty_D-12'].iloc[-1]], 
                'day_step_D-13': [X_train['day_step_D-12'].iloc[-1]], 
                'qty_D-14': [X_train['qty_D-13'].iloc[-1]],
                'day_step_D-14': [X_train['day_step_D-13'].iloc[-1]],
                'year': [day_to_forecast.year], 
                'month': [day_to_forecast.month], 
                'day': [day_to_forecast.day],
                'week_of_year': [day_to_forecast.isocalendar().week], 
                'day_of_week': [day_to_forecast.dayofweek], 
                'is_national_holiday': [is_national_holiday(day_to_forecast)],
                'is_state_holiday': [is_state_holiday(day_to_forecast)],
                'is_holiday': [is_national_holiday(day_to_forecast) + is_state_holiday(day_to_forecast)],
                'year_freq': year_freq_dict.get(X_train['year'].iloc[-1], 0)
            },
            index=[day_to_forecast]
        )
        forecast.append(model_xgb.predict(X_train_forecast)[0])

        X_train = pd.concat([X_train, X_train_forecast], axis=0)
        y_train = pd.concat([y_train, pd.Series([forecast[-1]], index=[day_to_forecast])], axis=0)

    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("eta", ETA)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("subsample", SUBSAMPLE)
    mlflow.log_param("colsample_bytree", COLSAMPLE_BYTREE)

    mlflow.sklearn.log_model(model_lr, artifact_path="xgb_model")

    model_name = "XGBRegressor Recursive"

    mae = mean_absolute_error( y_valid[:14], y_train.iloc[-14:] )
    mape = mean_absolute_percentage_error( y_valid[:14], y_train.iloc[-14:] )
    rmse = np.sqrt( mean_squared_error( y_valid[:14], y_train.iloc[-14:] ) )

    df_metrics_xgb = pd.DataFrame( { 'Model Name': model_name, 
                        'MAE': mae, 
                        'MAPE': mape,
                        'RMSE': rmse }, index=[0] )
    
    mlflow.log_metric("mae_val", mae)
    mlflow.log_metric("mape_val", mape)
    mlflow.log_metric("rmse_val", rmse)
    
    mlflow.set_tags(tags)
    
    print("rmse_val: {}".format(rmse))
    print("mae_val: {}".format(mae))
    print("MAPE_val:", round(mape, 2), "%")

    mlflow.end_run()


# COMMAND ----------

# MAGIC %md
# MAGIC # Save best model

# COMMAND ----------

# Get best model name

df_metrics = pd.concat([df_metrics_lr, df_metrics_xgb])
df_metrics = df_metrics.reset_index(drop=True)

best_model_name = df_metrics.loc[df_metrics["RMSE"].idxmin(), "Model Name"]
print(best_model_name)

# COMMAND ----------

# Get model
if best_model_name == "Linear Regression":
    best_model = model_lr
elif best_model_name == "XGBoost":
    best_model = model_xgb

# COMMAND ----------

import joblib

basePath = os.getcwd()
joblib.dump(best_model, basePath + "/models/best_model")
