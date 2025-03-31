import pandas as pd
from indicators.indicator_utils import add_indicators

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df = add_indicators(df)
    return df

