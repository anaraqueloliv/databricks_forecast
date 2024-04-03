# Databricks notebook source
# MAGIC %md
# MAGIC # Previsão de Demanda

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

import os
import joblib
import datetime
import holidays
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from pyspark.sql.types import *

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor



# COMMAND ----------

# Used for linaer regression
def make_multistep_target(ts, steps):
    """Creating a 14 days forward for target variable y (=qty)
    """
    return pd.concat(
        {f'y_D+{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Input variables

# COMMAND ----------

best_model_name = "Linear Regression"

# COMMAND ----------

# MAGIC %md
# MAGIC # Load model and predict

# COMMAND ----------

df_test_transformed = pd.read_csv(os.getcwd() + '/data/02-interim/df_test_transformed.csv')
X_test = df_test_transformed.drop('qty', axis=1)
y_test = df_test_transformed['qty']

# COMMAND ----------

basePath = os.getcwd()
loaded_model = joblib.load(basePath + "/models/best_model")

# COMMAND ----------

if best_model_name == "Linear Regression":
    X_test_lr = X_test.loc[:, [f'qty_D-{i+1}' for i in range(14)]]
    y_test_lr = y_test.copy()

    # Two-week forecast
    y_test_lr = make_multistep_target(y_test_lr, steps=14).dropna()

    # Shifting has created indexes that don't match. Only keep times for which we have both targets and features.
    y_test_lr, X_test_lr = y_test_lr.align(X_test_lr, join='inner', axis=0)

    # Prediction
    y_pred_lr = pd.DataFrame(loaded_model.predict(X_test_lr), index=y_test_lr.index, columns=y_test_lr.columns)


# COMMAND ----------

mae = mean_absolute_error( y_test_lr, y_pred_lr )
mape = mean_absolute_percentage_error( y_test_lr, y_pred_lr )
rmse = np.sqrt( mean_squared_error( y_test_lr, y_pred_lr ) )

df_metrics_lr = pd.DataFrame( { 'Model Name': best_model_name, 
                    'MAE': mae, 
                    'MAPE': mape,
                    'RMSE': rmse }, index=[0] )

print(df_metrics_lr)

# COMMAND ----------

# Save predictions
predictions = pd.concat([X_test, y_pred_lr], axis=1)
predictions.to_csv(os.getcwd() + '/data/03-processed/predictions.csv')
