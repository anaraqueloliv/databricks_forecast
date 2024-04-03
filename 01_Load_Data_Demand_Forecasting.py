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

# COMMAND ----------

# MAGIC %md
# MAGIC # Input variables

# COMMAND ----------

selected_distribution_center = 1

# COMMAND ----------

# MAGIC %md
# MAGIC # Collect data

# COMMAND ----------

# This dataset contains the target information, product_id ordered by day
df_order_items = pd.read_csv("data/01-raw/order_items.csv")
df_order_items.head(3)

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
# MAGIC ## Merge datasets

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

