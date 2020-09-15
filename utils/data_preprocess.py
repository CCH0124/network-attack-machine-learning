import numpy as np
import pandas as pd

def miss_values_rate(dataframe):
    missing_values_count = dataframe.isnull().sum()
    total_miss_values = missing_values_count.sum()
    total_cells = np.product(dataframe.shape)
    print(total_miss_values)
    print(total_cells)
    print((total_miss_values/total_cells) * 100) 