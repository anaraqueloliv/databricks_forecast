import pandas as pd
import pytest

import sys
import os
# notebook_dir = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath())
# functions_folder_path = os.path.join(notebook_dir, 'functions')
# print(functions_folder_path)
# dbutils.notebook.run("/Workspace/Users/ana.oliveira@bixtecnologia.com.br/databricks_forecast/functions", 60)
# %run "/Workspace/Users/ana.oliveira@bixtecnologia.com.br/databricks_forecast/functions/load_utils.py"
from functions.load_utils import load_data

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


ORDER_ITEMS_PATH_VALID = 'valid_order_items.csv'
DISTR_CENTERS_PATH_VALID = 'valid_distr_centers.csv'
PRODUCTS_PATH_VALID = 'valid_products.csv'


def test_load_data_valid():
    result = load_data(ORDER_ITEMS_PATH_VALID, DISTR_CENTERS_PATH_VALID, PRODUCTS_PATH_VALID)
    assert isinstance(result, tuple)
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], pd.DataFrame)
    assert isinstance(result[2], pd.DataFrame)

