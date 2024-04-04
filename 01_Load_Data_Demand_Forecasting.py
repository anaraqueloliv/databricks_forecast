# Databricks notebook source
# MAGIC %md
# MAGIC # Previsão de Demanda

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

import os
import datetime
import pandas as pd

import functions.load_utils as ld_utils

# COMMAND ----------

import logging

# Configure logging to display INFO level messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# COMMAND ----------

# MAGIC %md
# MAGIC # Input variables

# COMMAND ----------

selected_distribution_center = 1

# COMMAND ----------

# MAGIC %md
# MAGIC # Collect data

# COMMAND ----------

RAW_DATA_DIR = "data/01-raw/"
ORDER_ITEMS_FILE = "order_items.csv"
DISTR_CENTERS_FILE = "distribution_centers.csv"
PRODUCTS_FILE = "products.csv"

ORDER_ITEMS_PATH = RAW_DATA_DIR + ORDER_ITEMS_FILE
DISTR_CENTERS_PATH = RAW_DATA_DIR + DISTR_CENTERS_FILE
PRODUCTS_DIR_PATH = RAW_DATA_DIR + PRODUCTS_FILE

# COMMAND ----------

df_order_items = pd.read_csv(ORDER_ITEMS_PATH).head(10)
df_distribuition_centers = pd.read_csv(DISTR_CENTERS_PATH).head(10)
df_products = pd.read_csv(PRODUCTS_DIR_PATH).head(10)

# COMMAND ----------

df_order_items.to_csv("/Workspace/Users/ana.oliveira@bixtecnologia.com.br/databricks_forecast/tests/test_data/valid_order_items.csv")
df_distribuition_centers.to_csv("/Workspace/Users/ana.oliveira@bixtecnologia.com.br/databricks_forecast/tests/test_data/valid_distr_centers.csv")
df_products.to_csv("/Workspace/Users/ana.oliveira@bixtecnologia.com.br/databricks_forecast/tests/test_data/valid_products.csv")

# COMMAND ----------

order_items_path = ORDER_ITEMS_PATH
distr_centers_path = DISTR_CENTERS_PATH
products_path = PRODUCTS_DIR_PATH

df_order_items, df_distribuition_centers, df_products = ld_utils.load_data(order_items_path, distr_centers_path, products_path)
display(df_order_items.head(3))
display(df_distribuition_centers.head(3))
display(df_products.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge datasets

# COMMAND ----------

df_products = df_products.merge(df_distribuition_centers, how='left', on='product_distribution_center_id')
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
# MAGIC ## Choosing how to group by

# COMMAND ----------

df_dropped['created_at'] = pd.to_datetime(df_dropped['created_at']).dt.date

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

# MAGIC %md
# MAGIC ## Filter by distribution center

# COMMAND ----------

df_filtered = df_grouped[df_grouped['product_distribution_center_id']==selected_distribution_center]
df_filtered

# COMMAND ----------

# MAGIC %md
# MAGIC # Split data

# COMMAND ----------

# MAGIC %md
# MAGIC Split data into test, training and validation
# MAGIC

# COMMAND ----------

df = df_filtered.copy()

# COMMAND ----------

# We will exclude data after February 15th since it has a very anoumalous behaviour, we could be something with the dataset mantainance
df1 = df[df['created_at']<datetime.date(2024, 2, 16)]

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

