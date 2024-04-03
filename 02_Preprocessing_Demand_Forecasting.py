# Databricks notebook source
# MAGIC %md
# MAGIC # Previsão de Demanda

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

import os
import holidays
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# COMMAND ----------

# MAGIC %md
# MAGIC # Load data

# COMMAND ----------

# Load datasets
df_train = pd.read_csv(os.getcwd() + '/data/02-interim/df_train.csv')
df_valid = pd.read_csv(os.getcwd() + '/data/02-interim/df_valid.csv')
df_test = pd.read_csv(os.getcwd() + '/data/02-interim/df_test.csv')

# Separate target and features
X_train = df_train.drop('qty', axis=1)
y_train = df_train['qty']
X_valid = df_valid.drop('qty', axis=1)
y_valid = df_valid['qty']
X_test = df_test.drop('qty', axis=1)
y_test = df_test['qty']


# COMMAND ----------

# MAGIC %md
# MAGIC # Data preparation

# COMMAND ----------

def is_national_holiday(date):
    """
    Function to get all dates that are national holidays in US
    """
    national_holidays = holidays.CountryHoliday('US')
    return 1 if date in national_holidays else 0



# COMMAND ----------

def is_state_holiday(date):
    """
    Function to get all dates that are state holidays in US
    """
    state_holidays = ['07-19', '03-04', '03-29']
    return 1 if date in state_holidays else 0

# COMMAND ----------

def data_preprocessing(df):
    df['created_at'] = pd.to_datetime(df['created_at'])

    df = df.rename(columns={'created_at': 'date'})

    # Add non existentent dates with value equal to zero
    start_date = df['date'].min()
    end_date = df['date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df_all_dates = pd.DataFrame(all_dates, columns=['date'])
    df_all_dates['qty'] = 0
    df = pd.merge(df_all_dates, df, on='date', how='left', suffixes=('_all_dates', '_df'))
    df['qty_df'].fillna(0, inplace=True)
    df.drop(['qty_all_dates'], axis=1, inplace=True)
    df.rename(columns={'qty_df': 'qty'}, inplace=True)
    df = df.sort_values(by='date').reset_index(drop=True)
    df.fillna(method='bfill', inplace=True)

    # Create time step feature
    df['day_step'] = df.index

    # Create 14 day lag features
    df.sort_values(by='date', inplace=True)

    lag_values = np.arange(1, 15)
    for lag in lag_values:
        df[f'qty_D-{lag}'] = df['qty'].shift(lag)
        df[f'day_step_D-{lag}'] = df['day_step'].shift(lag)

    # NA after shift, since there is not many we will remove them
    df=df.dropna()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["day_of_week"] = df["date"].dt.dayofweek
    df['month_day'] = df['date'].dt.strftime('%m-%d')

    # Change date types
    df['day_of_week'] = df['day_of_week'].astype('float')
    df['month'] = df['month'].astype('float')
    df['day'] = df['day'].astype('float')
    df['week_of_year'] = df['week_of_year'].astype('float')

    # Creatin holiday variables
    df['is_national_holiday'] = df['date'].apply(is_national_holiday)
    df["is_state_holiday"] = df['month_day'].apply(is_state_holiday)
    df["is_holiday"] = df['is_national_holiday'] + df["is_state_holiday"]

    # Creating frequency encoding for year feature since the frequency of year is influential in the target variable
    freq = df['year'].value_counts()
    df['year_freq'] = df['year'].map(freq)

    # Renove dates that are strings and distribution center columns that will not be use since we are creating a single step prediction
    df_preprocessed = df.drop(['month_day', 'product_distribution_center_id', 'distribution_center_name', 	'distribution_center_latitude',	'distribution_center_longitude'], axis=1)
    df_preprocessed.set_index('date', inplace=True)

    return df_preprocessed


# COMMAND ----------

# def data_transformation(df_preprocessed):
#     # Frequency encoding
#     # Since the frequency of year is influential in the target variable
#     freq = df_preprocessed['year'].value_counts()
#     df_preprocessed['year'] = df_preprocessed['year'].map(freq)

#     # Nature transformations
#     # day_of_week
#     df_preprocessed['day_of_week'] = df_preprocessed['day_of_week'].astype('float')
#     df_preprocessed['day_of_week_sin'] = df_preprocessed['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
#     df_preprocessed['day_of_week_cos'] = df_preprocessed['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

#     # month
#     df_preprocessed['month'] = df_preprocessed['month'].astype('float')
#     df_preprocessed['month_sin'] = df_preprocessed['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
#     df_preprocessed['month_cos'] = df_preprocessed['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

#     # day
#     df_preprocessed['day'] = df_preprocessed['day'].astype('float')
#     df_preprocessed['day_sin'] = df_preprocessed['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
#     df_preprocessed['day_cos'] = df_preprocessed['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

#     # week_of_year
#     df_preprocessed['week_of_year'] = df_preprocessed['week_of_year'].astype('float')
#     df_preprocessed['week_of_year_sin'] = df_preprocessed['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/53 ) ) )
#     df_preprocessed['week_of_year_cos'] = df_preprocessed['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/53 ) ) )

#     df_transformed = df_preprocessed.copy()

#     return df_transformed


# COMMAND ----------

df_train_preprocessed = data_preprocessing(df_train)
# df_train_transformed = data_transformation(df_train_preprocessed)

df_valid_preprocessed = data_preprocessing(df_valid)
# df_valid_transformed = data_transformation(df_valid_preprocessed)

df_test_preprocessed = data_preprocessing(df_test)
# df_test_transformed = data_transformation(df_test_preprocessed)

# Save datasets
df_train_preprocessed.to_csv(os.getcwd() + '/data/02-interim/df_train_transformed.csv', index=True)
df_valid_preprocessed.to_csv(os.getcwd() + '/data/02-interim/df_valid_transformed.csv', index=True)
df_test_preprocessed.to_csv(os.getcwd() + '/data/02-interim/df_test_transformed.csv', index=True)

