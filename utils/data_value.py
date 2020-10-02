import numpy as np
import pandas as pd

def get_inf(dataframe):
    return len(dataframe.values[np.isinf(dataframe.values)])

def miss_values_rate(dataframe):
    missing_values_count = dataframe.isnull().sum()
    total_miss_values = missing_values_count.sum()
    total_cells = np.product(dataframe.shape)
    print(total_miss_values)
    print(total_cells)
    print((total_miss_values/total_cells) * 100) 

def nan_quadratic(dataframe):
    return dataframe.interpolate(method='quadratic')

def nan_ffill(dataframe):
    return dataframe.fillna(method='ffill')

def nan_insert_method(dataframe, method='ffill'):
    if method == 'ffill':
        return nan_ffill(dataframe)
    elif method == 'quadratic':
        return nan_quadratic(dataframe)
    else:
        exit(1)

def inf_and_na_drop(dataframe):
    dataframe = dataframe.replace(['Infinity', np.inf, -np.inf], np.nan)
    return dataframe.dropna(axis=0)
    

def inf_drop_insert_nan(dataframe, method='ffill'):
    dataframe = nan_insert_method(dataframe, method)
    return inf_and_na_drop(dataframe)
    
def inf_replace_value_ffill_nan(dataframe, val=9999999.9, method='ffill'):
    dataframe = dataframe.replace(['Infinity',np.inf, -np.inf], val)
    return nan_insert_method(dataframe, method)

def column_trim(dataframe):
    return dataframe.columns.str.strip()

def column_del_one_value(dataframe):
    counts = dataframe.nunique()
    counts = counts.to_dict()
    # record columns to delete
    to_del = [k for k in counts if counts[k] == 1]
    # drop useless columns
    # dataframe.drop(to_del, axis=1, inplace=True)
    return to_del