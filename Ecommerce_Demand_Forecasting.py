# Databricks notebook source
# MAGIC %md
# MAGIC # Previsão de Demanda

# COMMAND ----------

# MAGIC %md
# MAGIC ## Situação atual
# MAGIC
# MAGIC Um grande e-commerce internacional está observando problemas relacionadas a sua gestão de estoque. Perde-se oportunidade de vender produtos que estão sem estoque e há produtos com muito estoque e sem vendas há longos períodos.
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Otimizar o estoque de um e-commerce com base na demanda real de produtos.
# MAGIC
# MAGIC ## Premissas
# MAGIC
# MAGIC As compras são planejadas bimestralmente
# MAGIC
# MAGIC Todos os itens tem mais ou menos o mesmo leadtime de 1 semana
# MAGIC
# MAGIC Estoque de segurança é a quantidade para suprir a demanda de 1 semana
# MAGIC
# MAGIC Desconsiderando pedidos cancelados
# MAGIC
# MAGIC ## Entregáveis
# MAGIC
# MAGIC - Prever a demanda de cada centro de distribuição nas próximas 2 semanas (em unidades vendidas)
# MAGIC - Assumindo que cada o de mix de produtos de cada centro de distribuição muda a cada três meses, é possível calcular a quantidade em demanda de cada produto e os items que o compõe

# COMMAND ----------

# MAGIC %md
# MAGIC # 0.0 - Imports

# COMMAND ----------

import os
import datetime
import holidays
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from pyspark.sql.types import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
# MAGIC # 1.0 - Collect data

# COMMAND ----------

# # read the training file into a dataframe
# order_items = spark.read.csv(
#   'dbfs:/Workspace/Users/ana.oliveira@bixtecnologia.com.br/data/order_items.csv', 
#   )
 
# # make the dataframe queryable as a temporary view
# order_items.createOrReplaceTempView('order_items')
 
# # show data
# display(order_items)

# COMMAND ----------

# %sql
# SELECT *
# FROM order_items
# limit 10

# COMMAND ----------

# We will not have information about the users activity by the time of prediction
df_events = pd.read_csv("data/01-raw/events.csv")
df_events.head(3)

# COMMAND ----------

# This dataset contains the target information, product_id ordered by day
df_order_items = pd.read_csv("data/01-raw/order_items.csv")
df_order_items.head(3)

# COMMAND ----------

df_order_items[df_order_items.order_id==8]

# COMMAND ----------

# We will not have information about orders by the time of prediction
df_orders = pd.read_csv("data/01-raw/orders.csv")
df_orders.head(3)

# COMMAND ----------

# We will not have information about users by the time of prediction
df_users = pd.read_csv("data/01-raw/users.csv")
df_users = df_users.add_prefix('user_')
df_users.head(3)

# COMMAND ----------

# This dataset will not bring any new information so it will not be merged
df_inventory_items = pd.read_csv("data/01-raw/inventory_items.csv")
df_inventory_items.head(3)

# COMMAND ----------

df_distribuition_centers = pd.read_csv("data/01-raw/distribution_centers.csv")
df_distribuition_centers = df_distribuition_centers.add_prefix('distribution_center_')
df_distribuition_centers = df_distribuition_centers.rename(columns={'distribution_center_id': 'product_distribution_center_id'})
df_distribuition_centers.head(3)

# COMMAND ----------

df_products = pd.read_csv("data/01-raw/products.csv")
df_products = df_products.add_prefix('product_')
df_products = df_products.merge(df_distribuition_centers, how='left', on='product_distribution_center_id')
df_products.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Merge datasets

# COMMAND ----------

df_merged = df_order_items.merge(df_products, how='left', on='product_id')
df_merged.head(3)

# COMMAND ----------

df_merged.columns

# COMMAND ----------

# We will consider all status including 'cancelled', assuming that the ordered products must be in stock by the time the order is created.
df_merged.status.value_counts()

# COMMAND ----------

# Now we want drop columns about order and users in the merged dataframe
# We do not need information about inventory_item_id since we are predict product demand and not items
# The sale_price information is dropped because we want predict quantity of products and not sales, it could be use as target in another prediction
df_dropped = df_merged.drop(['order_id', 'user_id', 'inventory_item_id', 'status', 'shipped_at', 'delivered_at', 'returned_at', 'sale_price'], axis=1)

# COMMAND ----------

df_dropped.nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 - Choosing how to group by

# COMMAND ----------

df_dropped['created_at'] = pd.to_datetime(df_dropped['created_at']).dt.date

# COMMAND ----------

# We will simplify our prediction to only a few products, do let's apply 80/20 analysis

aux_qty_by_product = df_dropped.groupby('product_id')['created_at'].count()
aux_qty_by_product.sort_values(ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Since we do not have few products with a high number of sales we can not apply 80/20 rule. Since every product comes from only one specific distribution center and we only have 10 of them, let's simplify by distribution center.

# COMMAND ----------

aux_qty_by_distribution_center = df_dropped.groupby('product_distribution_center_id')['created_at'].count()
aux_qty_by_distribution_center.sort_values(ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC - All distribution center has a considerable amount of produts ordered, so there is no need to exclude any from analysis
# MAGIC - All the product information we will be dropped since we have chosen to pursue an analysis by distribution center
# MAGIC

# COMMAND ----------

# Column 'qty' is the number of items ordered
df_grouped = df_dropped[[
                        'created_at', 
                        'product_distribution_center_id', 
                        'distribution_center_name', 
                        'distribution_center_latitude', 
                        'distribution_center_longitude', 
                        'id']].groupby([
                                'created_at', 
                                'product_distribution_center_id', 
                                'distribution_center_name', 
                                'distribution_center_latitude', 
                                'distribution_center_longitude']).count()
df_grouped = df_grouped.reset_index()
df_grouped = df_grouped.rename(columns={'id':'qty'})
df_grouped.sort_values(by='created_at', ascending=False)

# COMMAND ----------

df_grouped[df_grouped.created_at==datetime.date(2024, 1, 3)]

# COMMAND ----------

df_grouped.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC Since is just a demo we will simplify the problem filtering only 3 distribution centers

# COMMAND ----------

df_grouped.product_distribution_center_id.value_counts()

# COMMAND ----------

df_filtered = df_grouped[df_grouped.product_distribution_center_id.isin([1, 2 ,3])]
df_filtered

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.0 - Split data

# COMMAND ----------

# MAGIC %md
# MAGIC Split data into test, training and validation
# MAGIC

# COMMAND ----------

df = df_filtered.copy()

# COMMAND ----------

df['created_at'].min()

# COMMAND ----------

df['created_at'].max()

# COMMAND ----------


plt.figure(figsize=(10, 6))
plt.plot('created_at', 'qty', data=df)
plt.xticks(rotation=45) 

# COMMAND ----------

# Check how is the behaviour for the last year only
aux_last_year = df[(df['created_at']<datetime.date(2024, 1, 1))&(df['created_at']>datetime.date(2023, 1, 1))]
plt.figure(figsize=(10, 6))
plt.plot('created_at', 'qty', data=aux_last_year)
plt.xticks(rotation=45) 

# COMMAND ----------

# Check how is the behaviour for the last month
aux_last_months = df[df['created_at']>datetime.date(2024, 1, 1)]
plt.figure(figsize=(10, 6))
plt.plot('created_at', 'qty', data=aux_last_months)
plt.xticks(rotation=45)

# COMMAND ----------

# We will exclude data after February 15th since it has a very anoumalous behaviour, we could be something with the dataset mantainance
df1 = df[df['created_at']<datetime.date(2024, 2, 16)]

# COMMAND ----------

# Check how is the behaviour for filtered data
plt.figure(figsize=(10, 6))
plt.plot('created_at', 'qty', data=df1)
plt.xticks(rotation=45)

# COMMAND ----------

# MAGIC %md
# MAGIC Since we are dealing with timed data and the target it is a prediction over time, we will split considering date
# MAGIC - Test set will have the last 2 weeks: data >= 2024-02-01
# MAGIC - Validation set will have 3 months: data >= 2023-11-01
# MAGIC - Train set: all data before < 2023-11-01
# MAGIC

# COMMAND ----------

# Split
df_train = df1[df1['created_at']<datetime.date(2023, 11, 1)]
df_valid = df1[(df1['created_at']>=datetime.date(2023, 11, 1))&(df1['created_at']<datetime.date(2024, 2, 1))]
df_test = df1[df1['created_at']>=datetime.date(2024, 2, 1)]

# Save datasets
df_train.to_csv(os.getcwd() + '/data/02-interim/df_train.csv', index=False)
df_valid.to_csv(os.getcwd() + '/data/02-interim/df_valid.csv', index=False)
df_test.to_csv(os.getcwd() + '/data/02-interim/df_test.csv', index=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 - Baseline model

# COMMAND ----------

# regr = LinearRegression()

# X_train["created_at"] = X_train["created_at"] .astype(str)
# regr.fit(X_train, y_train)
# print(regr.score(X_test, y_test))

# COMMAND ----------

# MAGIC %md
# MAGIC # 3.0 - Data cleaning

# COMMAND ----------

df3 = df_train.copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1  - Data description

# COMMAND ----------

print(f"Number of rows: {df3.shape[0]}")
print(f"Number of columns: {df3.shape[1]}")

# COMMAND ----------

# Columns
df3.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 - Check nulls

# COMMAND ----------

df3.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 - Check dtypes

# COMMAND ----------

df3.dtypes

# COMMAND ----------

# Change types
df3['created_at'] = pd.to_datetime(df3['created_at'])
df3['product_distribution_center_id'] = df3['product_distribution_center_id'].astype('object')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 - Rename columns

# COMMAND ----------

df3 = df3.rename(columns={'created_at': 'date'})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 - Check duplicates

# COMMAND ----------

duplicates = df3.duplicated()
print("Number of duplicates:", duplicates.sum())

# COMMAND ----------

# MAGIC %md
# MAGIC # 4.0 - Feature engineering

# COMMAND ----------

df4 = df3.copy()

# COMMAND ----------

df4.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 - Time series features

# COMMAND ----------

# Add non existentent dates with value equal to zero
start_date = df4['date'].min()
end_date = df4['date'].max()
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
distribution_centers = df4['product_distribution_center_id'].unique()
df_all_dates = pd.DataFrame([(date, distribution_center) for date in all_dates for distribution_center in distribution_centers], columns=['date', 'product_distribution_center_id'])
df_all_dates['qty'] = 0

df4 = pd.merge(df_all_dates, df4, on=['date', 'product_distribution_center_id'], how='left', suffixes=('_all_dates', '_df4'))
df4['qty_df4'].fillna(0, inplace=True)
df4.drop(['qty_all_dates'], axis=1, inplace=True)
df4.rename(columns={'qty_df4': 'qty'}, inplace=True)
df4 = df4.sort_values(by=['product_distribution_center_id', 'date']).reset_index(drop=True)
df4.fillna(method='bfill', inplace=True)

# COMMAND ----------

# Create time step feature
df4['day_step'] = df4.groupby('product_distribution_center_id').cumcount() + 1

# COMMAND ----------

# Create 14 day lag features
df4.sort_values(by='date', inplace=True)

lag_values = np.arange(1, 15)
for lag in lag_values:
    df4[f'qty_D-{lag}'] = df4.groupby('product_distribution_center_id')['qty'].shift(lag)
    df4[f'day_step_D-{lag}'] = df4.groupby('product_distribution_center_id')['day_step'].shift(lag)

# COMMAND ----------

# NA after shift, since there is not many we will remove them
df4=df4.dropna()
print(df4.isna().sum())

# COMMAND ----------

df4[df4["product_distribution_center_id"]==3].head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 - Derive features from date

# COMMAND ----------

# MAGIC %md
# MAGIC We will extract the following features from variable 'date':
# MAGIC - year
# MAGIC - month
# MAGIC - week_of_year
# MAGIC - day
# MAGIC - day_of_week
# MAGIC - year_week

# COMMAND ----------

df4["year"] = df4["date"].dt.year
df4["month"] = df4["date"].dt.month
df4["day"] = df4["date"].dt.day
df4["week_of_year"] = df4["date"].dt.isocalendar().week
df4["day_of_week"] = df4["date"].dt.dayofweek
df4["year_week"] = df4["date"].dt.strftime('%Y-%W')
df4['month_day'] = df4['date'].dt.strftime('%m-%d')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 - Get variable for holidays

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2.1 - National holidays

# COMMAND ----------

# MAGIC %md
# MAGIC We will get national holidays using the library holidays

# COMMAND ----------

def is_national_holiday(date):
    """
    Function to get all dates that are national holidays in US
    """
    national_holidays = holidays.CountryHoliday('US')
    return 1 if date in national_holidays else 0



# COMMAND ----------

df4['is_national_holiday'] = df4['date'].apply(is_national_holiday)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2.2 - State holidays

# COMMAND ----------

# MAGIC %md
# MAGIC We will get state holidays manually since there is no libraru and we have only three states

# COMMAND ----------

df4.distribution_center_name.unique()

# COMMAND ----------

# MAGIC %md
# MAGIC - Illinois
# MAGIC
# MAGIC     - Lincoln's Birthday	- February 19
# MAGIC
# MAGIC     - Pulaski Day	Monday - March 4
# MAGIC
# MAGIC - Texas
# MAGIC
# MAGIC     - No State holidays
# MAGIC
# MAGIC - Tennessee
# MAGIC
# MAGIC     - Good Friday - March 29

# COMMAND ----------

def is_state_holiday(date):
    state_holidays = ['07-19', '03-04', '03-29']
    return 1 if date in state_holidays else 0

# COMMAND ----------

df4["is_state_holiday"] = df4['month_day'].apply(is_state_holiday)


# COMMAND ----------

df4.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 - Geografic features

# COMMAND ----------

# MAGIC %md
# MAGIC If we have more distribution centers we could explore more to group by region, state and etc.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.0 - Variable filtering

# COMMAND ----------

# MAGIC %md
# MAGIC Since we are using only three distribution center the geographic information will not be explore.

# COMMAND ----------

df5 = df4.copy()

# COMMAND ----------

df5 = df5.drop(['distribution_center_latitude', 'distribution_center_longitude', 'distribution_center_name'], axis=1)

# COMMAND ----------

# Change dtypes
df5.dtypes

# COMMAND ----------

df5["year"] = df5["year"].astype(str)
df5["month"] = df5["month"].astype(str)
df5["day"] = df5["day"].astype(str)
df5["week_of_year"] = df5["week_of_year"].astype(str)
df5["day_of_week"] = df5["day_of_week"].astype(str)
df5["is_national_holiday"] = df5["is_national_holiday"].astype(str)
df5["is_state_holiday"] = df5["is_state_holiday"].astype(str)

# COMMAND ----------

# Save dataframe
df5.to_csv(os.getcwd() + '/data/02-interim/df5.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # 6.0 - EDA - Exploratory Data Analysis

# COMMAND ----------

dtype_dict = {
    'date': 'object',
    'product_distribution_center_id': 'object',
    'qty': 'int64',
    'year': 'object',
    'month': 'object',
    'day': 'object',
    'week_of_year': 'object',
    'day_of_week': 'object',
    'year_week': 'object',
    'month_day': 'object',
    'is_national_holiday': 'object',
    'is_state_holiday': 'object'
}


# COMMAND ----------

df6 = pd.read_csv(os.getcwd() + '/data/df5.csv', dtype=dtype_dict)

# COMMAND ----------

df6.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.1 - Descriptive statistics

# COMMAND ----------

df6.describe()

# COMMAND ----------

df6.describe(include='object')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 - Response variable

# COMMAND ----------

sns.displot( df6['qty'], kde=False  )
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.3 - Univariate and Bivariate Analysis

# COMMAND ----------

from ydata_profiling import ProfileReport
ProfileReport(df6)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.5 - Check normality

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.6 - Check outliers

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 7.0 - Data preparation

# COMMAND ----------

df7 = df5.copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.1 - Response variable

# COMMAND ----------

# # Since the response variable is not normal we will apply a log transformation

# df7['qty'] = np.log1p( df7['qty'] )

# sns.displot( df7['qty'], kde=False  )
# plt.show()


# COMMAND ----------

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# qty_scaled = scaler.fit_transform(df7['qty'].values.reshape(-1, 1))
# df7["qty"] = qty_scaled.flatten()

# sns.displot( df7["qty"], kde=False  )
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.2 - Categorical encoders

# COMMAND ----------

# Frequency encoding
# Since the frequency of year is influential in the target variable
freq = df7['year'].value_counts()
df7['year'] = df7['year'].map(freq)

# COMMAND ----------

df7.is_national_holiday = df7.is_national_holiday.astype('int')
df7.is_state_holiday = df7.is_state_holiday.astype('int')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.3 - Nature transformations

# COMMAND ----------

df7.dtypes

# COMMAND ----------

# # day_of_week
df7['day_of_week'] = df7['day_of_week'].astype('float')
# df7['day_of_week_sin'] = df7['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
# df7['day_of_week_cos'] = df7['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

# # month
df7['month'] = df7['month'].astype('float')
# df7['month_sin'] = df7['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
# df7['month_cos'] = df7['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

# # day
df7['day'] = df7['day'].astype('float')
# df7['day_sin'] = df7['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
# df7['day_cos'] = df7['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

# # week_of_year
df7['week_of_year'] = df7['week_of_year'].astype('float')
# df7['week_of_year_sin'] = df7['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/53 ) ) )
# df7['week_of_year_cos'] = df7['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/53 ) ) )


# COMMAND ----------

# MAGIC %md
# MAGIC # 8.0 - Feature selection

# COMMAND ----------

df8 = df7.reindex().copy()

# COMMAND ----------

# Drop string columns after data preparations

string_cols = ['year_week', 'month_day']
df8 = df8.drop(string_cols, axis=1)
df8.set_index('date', inplace=True)

# COMMAND ----------

df8.dtypes

# COMMAND ----------

# Model definition
forest = ExtraTreesRegressor( n_estimators=250, random_state=0, n_jobs=-1 )

# Data preparation
X_train_fs = df8.drop("qty", axis=1)
y_train_fs = df8["qty"]

forest.fit( X_train_fs, y_train_fs )

# COMMAND ----------


# Ordenate by importance - columns numbers
importances = forest.feature_importances_
indices = np.argsort( importances )[::-1]

# Print feature ranking
print('Feature Ranking: ')
df = pd.DataFrame()

for i, j in zip( X_train_fs, importances):
    aux = pd.DataFrame( {'feature': i, 'importance': j}, index=[0] )
    df = pd.concat( [df, aux] , axis=0)
print(df.sort_values('importance', ascending=False))
    
# Plot feature importance

plt.figure()
plt.title('Feature Importance')
plt.bar( range(X_train_fs.shape[1]), importances[indices]);
plt.xticks( range(X_train_fs.shape[1]), X_train_fs.columns[indices], rotation=90 );
plt.xlim( -1, X_train_fs.shape[1] )

# COMMAND ----------

# MAGIC %md
# MAGIC # 9.0 - Data modeling

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.1 - Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load data

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

X_train.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data preparation

# COMMAND ----------

def data_preprocessing(df):
    df['created_at'] = pd.to_datetime(df['created_at'])

    df = df.rename(columns={'created_at': 'date'})


    # Add non existentent dates with value equal to zero
    start_date = df['date'].min()
    end_date = df['date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    distribution_centers = df['product_distribution_center_id'].unique()
    df_all_dates = pd.DataFrame([(date, distribution_center) for date in all_dates for distribution_center in distribution_centers], columns=['date', 'product_distribution_center_id'])
    df_all_dates['qty'] = 0

    df = pd.merge(df_all_dates, df, on=['date', 'product_distribution_center_id'], how='left', suffixes=('_all_dates', '_df'))
    print(df.columns)
    df['qty_df'].fillna(0, inplace=True)
    df.drop(['qty_all_dates'], axis=1, inplace=True)
    df.rename(columns={'qty_df': 'qty'}, inplace=True)
    df = df.sort_values(by=['product_distribution_center_id', 'date']).reset_index(drop=True)
    df.fillna(method='bfill', inplace=True)

    # Create time step feature
    df['day_step'] = df.groupby('product_distribution_center_id').cumcount() + 1

    # Create 14 day lag features
    df.sort_values(by=['product_distribution_center_id', 'date'], inplace=True)

    lag_values = np.arange(1, 15) # from D-1 to D-14
    for lag in lag_values:
        df[f'qty_D-{lag}'] = df.groupby('product_distribution_center_id')['qty'].shift(lag)

    df=df.dropna()


    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["day_of_week"] = df["date"].dt.dayofweek
    df["year_week"] = df["date"].dt.strftime('%Y-%W')
    df['month_day'] = df['date'].dt.strftime('%m-%d')

    df['is_national_holiday'] = df['date'].apply(is_national_holiday)
    df["is_state_holiday"] = df['month_day'].apply(is_state_holiday)

    df_preprocessed = df.drop(['distribution_center_latitude', 'distribution_center_longitude', 'distribution_center_name', 'year_week', 'month_day', 'date'], axis=1)

    return df_preprocessed


# COMMAND ----------

df_preprocessed = data_preprocessing(df_valid)
display(df_preprocessed.head())

# COMMAND ----------

df_preprocessed.dtypes

# COMMAND ----------

def data_transformation(df_preprocessed):

    # Frequency encoding
    # Since the frequency of year is influential in the target variable
    freq = df_preprocessed['year'].value_counts()
    df_preprocessed['year'] = df_preprocessed['year'].map(freq)

    # Nature transformations
    # day_of_week
    df_preprocessed['day_of_week'] = df_preprocessed['day_of_week'].astype('float')
    df_preprocessed['day_of_week_sin'] = df_preprocessed['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
    df_preprocessed['day_of_week_cos'] = df_preprocessed['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

    # month
    df_preprocessed['month'] = df_preprocessed['month'].astype('float')
    df_preprocessed['month_sin'] = df_preprocessed['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
    df_preprocessed['month_cos'] = df_preprocessed['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

    # day
    df_preprocessed['day'] = df_preprocessed['day'].astype('float')
    df_preprocessed['day_sin'] = df_preprocessed['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
    df_preprocessed['day_cos'] = df_preprocessed['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

    # week_of_year
    df_preprocessed['week_of_year'] = df_preprocessed['week_of_year'].astype('float')
    df_preprocessed['week_of_year_sin'] = df_preprocessed['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/53 ) ) )
    df_preprocessed['week_of_year_cos'] = df_preprocessed['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/53 ) ) )

    df_transformed = df_preprocessed.copy()

    return df_transformed


# COMMAND ----------

df_transformed = data_transformation(df_preprocessed)
display(df_transformed.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pipeline

# COMMAND ----------

def get_predictions(model, model_name):

    # Load datasets
    df_train = pd.read_csv(os.getcwd() + '/data/02-interim/df_train.csv')
    df_valid = pd.read_csv(os.getcwd() + '/data/02-interim/df_valid.csv')

    # Separate target and features
    X_train = df_train.drop('qty', axis=1)
    y_train = df_train['qty']
    X_valid = df_valid.drop('qty', axis=1)
    y_valid = df_valid['qty']

    # Data preparation
    X_train = data_preprocessing(X_train)
    X_train = data_transformation(X_train)
    X_valid = data_preprocessing(X_valid)
    X_valid = data_transformation(X_valid)

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        # ('label', LabelEncoder()),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    categorical_cols = ['is_national_holiday', 'is_state_holiday', 'product_distribution_center_id']
    numerical_cols = X_train.select_dtypes( exclude='object' ).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('model', model)
                        ])

    # Preprocessing of training data, fit model 
    clf.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = clf.predict(X_valid)


    mae = mean_absolute_error( y_valid, preds )
    mape = mean_absolute_percentage_error( y_valid, preds )
    rmse = np.sqrt( mean_squared_error( y_valid, preds ) )
    
    return pd.DataFrame( { 'Model Name': model_name, 
                           'MAE': mae, 
                           'MAPE': mape,
                           'RMSE': rmse }, index=[0] )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.2 - Linear regression

# COMMAND ----------

aux_dc_0 = df8[df8['product_distribution_center_id']==0]

# COMMAND ----------

X=aux_dc_0.loc[:, [f'qty_D-{i+1}' for i in range(14)]]
y = aux_dc_0['qty']

# COMMAND ----------

X

# COMMAND ----------

y

# COMMAND ----------

# y, X = y.align(X, join='inner')  # drop corresponding values in target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=60, shuffle=False)

# Fit and predict
model = LinearRegression()  # `fit_intercept=True` since we didn't use DeterministicProcess
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_train), index=y_train.index)
y_fore = pd.Series(model.predict(X_test), index=y_test.index)

# COMMAND ----------

ax = y_train.plot(**plot_params)
ax = y_test.plot(**plot_params)
ax = y_pred.plot(ax=ax)
_ = y_fore.plot(ax=ax, color='C3')

# COMMAND ----------

from sklearn.metrics import mean_absolute_percentage_error

# Assuming y is the true target values and y_pred is the predicted values
train_rmse = mean_squared_error(y_train, y_pred, squared=False)
percentage_error = mean_absolute_percentage_error(y_train, y_pred)
print("RMSE:", train_rmse)
print("Percentage Error:", percentage_error)

# COMMAND ----------

# Assuming y is the true target values and y_pred is the predicted values
test_rmse = mean_squared_error(y_test, y_fore, squared=False)
test_percentage_error = mean_absolute_percentage_error(y_test, y_fore)
print("RMSE:", test_rmse)
print("Percentage Error:", test_percentage_error)

# COMMAND ----------

import matplotlib.pyplot as plt

# Plotting y against y_pred
plt.scatter(y_train, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.show()


# COMMAND ----------

def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_D+{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)


# Eight-week forecast
y = make_multistep_target(y, steps=14).dropna()

# COMMAND ----------

y

# COMMAND ----------

# Shifting has created indexes that don't match. Only keep times for which we have both targets and features.
y, X = y.align(X, join='inner', axis=0)

# COMMAND ----------

X

# COMMAND ----------

# Create splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

# COMMAND ----------

from sklearn.metrics import mean_absolute_percentage_error

# Assuming y is the true target values and y_pred is the predicted values
train_rmse = mean_squared_error(y_train, y_fit, squared=False)
percentage_error = mean_absolute_percentage_error(y_train, y_fit)
print("RMSE:", train_rmse)
print("Percentage Error:", percentage_error)

# COMMAND ----------

# Assuming y is the true target values and y_pred is the predicted values
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
test_percentage_error = mean_absolute_percentage_error(y_test, y_pred)
print("RMSE:", test_rmse)
print("Percentage Error:", test_percentage_error)

# COMMAND ----------

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(XGBRegressor())
model.fit(X_train, y_train)

y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

# COMMAND ----------

from sklearn.metrics import mean_absolute_percentage_error

# Assuming y is the true target values and y_pred is the predicted values
train_rmse = mean_squared_error(y_train, y_fit, squared=False)
percentage_error = mean_absolute_percentage_error(y_train, y_fit)
print("RMSE:", train_rmse)
print("Percentage Error:", percentage_error)

# COMMAND ----------

# Assuming y is the true target values and y_pred is the predicted values
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
test_percentage_error = mean_absolute_percentage_error(y_test, y_pred)
print("RMSE:", test_rmse)
print("Percentage Error:", test_percentage_error)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.3 - Random forest

# COMMAND ----------

# Define model
model = RandomForestRegressor( n_estimators = 100, n_jobs = -1, random_state = 42 )
model_name = 'Random Forest'

# Get predictions
get_predictions(model, model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.4 - XGBoost

# COMMAND ----------

selected_cols = ['qty', 'day_step'] + [f'qty_D-{i+1}' for i in range(14)] + ['year', 'month', 'day', 'week_of_year', 'day_of_week', 'is_national_holiday', 'is_state_holiday']

# COMMAND ----------

aux_recursive = df8[df8['product_distribution_center_id']==1][selected_cols]

# COMMAND ----------

X = aux_recursive.drop('qty', axis=1)
y = aux_recursive['qty']

# COMMAND ----------

# Create splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# COMMAND ----------

y_train

# COMMAND ----------

# Recursive time series strategy
forecast = []

# Train XGBoost model and make forecasts recursively
for i in range(14):
    print(f"Prediction nº {i+1}/14")
    model = xgb.XGBRegressor( objective = 'reg:squarederror', 
                        n_estimators = 500, 
                        eta = 0.01, 
                        max_depth = 15,
                        subsample = 0.7,
                        colsample_bytree = 0.8)

    model.fit(X_train, y_train)
    # Forecast for the next day
    day_to_forecast = pd.to_datetime(X_train.index[-1]) + pd.Timedelta(days=1)
    X_train_forecast = pd.DataFrame(
        {
            'day_step': [X_train['day_step'].iloc[-1] + 1], 
            'qty_D-1': [y_train[-1]], 
            'qty_D-2': [X_train['qty_D-1'].iloc[-1]], 
            'qty_D-3': [X_train['qty_D-2'].iloc[-1]], 
            'qty_D-4': [X_train['qty_D-3'].iloc[-1]], 
            'qty_D-5': [X_train['qty_D-4'].iloc[-1]],
            'qty_D-6': [X_train['qty_D-5'].iloc[-1]], 
            'qty_D-7': [X_train['qty_D-6'].iloc[-1]], 
            'qty_D-8': [X_train['qty_D-7'].iloc[-1]], 
            'qty_D-9': [X_train['qty_D-8'].iloc[-1]], 
            'qty_D-10': [X_train['qty_D-9'].iloc[-1]], 
            'qty_D-11': [X_train['qty_D-10'].iloc[-1]],
            'qty_D-12': [X_train['qty_D-12'].iloc[-1]], 
            'qty_D-13': [X_train['qty_D-12'].iloc[-1]], 
            'qty_D-14': [X_train['qty_D-13'].iloc[-1]], 
            'year': [day_to_forecast.year], 
            'month': [day_to_forecast.month], 
            'day': [day_to_forecast.day],
            'week_of_year': [day_to_forecast.isocalendar().week], 
            'day_of_week': [day_to_forecast.dayofweek], 
            'is_national_holiday': [is_national_holiday(day_to_forecast)],
            'is_state_holiday': [is_state_holiday(day_to_forecast)] 
        },
        index=[day_to_forecast]
    )
    forecast.append(model.predict(X_train_forecast)[0])

    X_train = pd.concat([X_train, X_train_forecast], axis=0)
    y_train = pd.concat([y_train, pd.Series([forecast[-1]], index=[day_to_forecast])], axis=0)

# COMMAND ----------

y_train.iloc[-14:]

# COMMAND ----------

y_test[:14]

# COMMAND ----------

model_name = "XGBRegressor Recursive"

mae = mean_absolute_error( y_test[:14], y_train.iloc[-14:] )
mape = mean_absolute_percentage_error( y_test[:14], y_train.iloc[-14:] )
rmse = np.sqrt( mean_squared_error( y_test[:14], y_train.iloc[-14:] ) )

print(pd.DataFrame( { 'Model Name': model_name, 
                    'MAE': mae, 
                    'MAPE': mape,
                    'RMSE': rmse }, index=[0] ) )

# COMMAND ----------

# Define hyperparameters
params = {
    'objective': 'reg:squarederror',  # Regression task
    'eta': 0.1,                       # Learning rate
    'max_depth': 3,                   # Maximum depth of each tree
    'subsample': 0.8,                 # Subsample ratio of the training instance
    'colsample_bytree': 0.8,          # Subsample ratio of columns when constructing each tree
    'n_estimators': 100,              # Number of boosting rounds (trees)
    'random_state': 42                # Random seed for reproducibility
}

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 10.0 - Hyperparameter fine tuning

# COMMAND ----------

# MAGIC %md
# MAGIC # 11.0 - Error evaluation and interpretation

# COMMAND ----------


