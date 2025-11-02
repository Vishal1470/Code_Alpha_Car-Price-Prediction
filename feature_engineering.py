import pandas as pd

def add_features(df):
    # Example: Create price_per_hp (price divided by horsepower)
    if 'horsepower' in df.columns and 'price' in df.columns:
        df['price_per_hp'] = df['price'] / df['horsepower']
    return df
