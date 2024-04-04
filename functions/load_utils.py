import pandas as pd
import logging

# Configure logging
logging.basicConfig(filename='output.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# def load_data(order_items_path, distr_centers_path, products_path):
#     # This dataset contains the target information, product_id ordered by day
#     df_order_items = pd.read_csv(order_items_path)

#     df_distribuition_centers = pd.read_csv(distr_centers_path)
#     df_distribuition_centers = df_distribuition_centers.add_prefix('distribution_center_')
#     df_distribuition_centers = df_distribuition_centers.rename(columns={'distribution_center_id': 'product_distribution_center_id'})

#     df_products = pd.read_csv(products_path)
#     df_products = df_products.add_prefix('product_')

#     display(df_order_items.head(3))
#     display(df_distribuition_centers.head(3))
#     display(df_products.head(3))

#     return df_order_items, df_distribuition_centers, df_products


def load_data(order_items_path, distr_centers_path, products_path):
    try:
        # Check if the paths are valid
        if not all(map(lambda x: isinstance(x, str), [order_items_path, distr_centers_path, products_path])):
            raise ValueError("Invalid path(s) provided.")

        # Load order items data
        df_order_items = pd.read_csv(order_items_path)

        # Load distribution centers data
        df_distribuition_centers = pd.read_csv(distr_centers_path)
        df_distribuition_centers = df_distribuition_centers.add_prefix('distribution_center_')

        # Check if 'distribution_center_id' column exists
        if 'distribution_center_id' not in df_distribuition_centers.columns:
            raise ValueError("Column 'distribution_center_id' not found in distribution centers data.")

        # Rename column if it exists
        if 'distribution_center_id' in df_distribuition_centers.columns:
            df_distribuition_centers = df_distribuition_centers.rename(columns={'distribution_center_id': 'product_distribution_center_id'})

        # Load products data
        df_products = pd.read_csv(products_path)
        df_products = df_products.add_prefix('product_')

        # Log success message
        logging.info('Data loaded successfully')

        return df_order_items, df_distribuition_centers, df_products

    except FileNotFoundError as e:
        # Log file not found error
        logging.error(f'File not found: {str(e)}', exc_info=True)
        raise e

    except ValueError as e:
        # Log invalid path error
        logging.error(f'Invalid path(s) provided: {str(e)}', exc_info=True)
        raise e

    except Exception as e:
        # Log other errors
        logging.error(f'An error occurred: {str(e)}', exc_info=True)
        raise e



