# imports
import numpy as np
import pandas as pd

def load_raw(path: str) -> pd.DataFrame:
    dataset = pd.read_csv(path, index_col=None)
    dataset = dataset.iloc[:,1:]
    return dataset

def add_return_target(df: pd.DataFrame) -> pd.DataFrame:
    df['Return'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['Target'] = df['Return'].shift(-1)
    df['Target'] = (df['Target'] > 0).astype(int)
    return df

def add_pct_change(df: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    for days in range(1,max_lag): # 46
        df[f'PctChange_{days}days'] = df['Return'].pct_change(periods=days) * 100
        df[f'PctChange_{days}days'] =  df[f'PctChange_{days}days'].replace([np.inf, -np.inf], np.nan) #Replace inf with NaN
        df[f'PctChange_{days}days'] =  df[f'PctChange_{days}days'].fillna(0) # Replace NaN with 0 or another appropriate value
    return df

def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    # Rolling statistics can capture short-term trend and volatillity.
    returns = df['Return']

    for w in windows:
        df[f"roll_mean_{w}"] = returns.rolling(window=w).mean()
        df[f"roll_std_{w}"]  = returns.rolling(window=w).std()
        df[f"mom_{w}"] = returns.diff(w) # momentum

    # Exponential moving average
    df['ema_10'] = returns.ewm(span=10, adjust=False).mean()

    # 10-day RSI -Relative Strength Index
    gain = returns.where(returns > 0, 0)
    loss = -returns.where(returns < 0, 0)
    avg_gain = gain.rolling(10).mean()
    avg_loss = loss.rolling(10).mean()
    rs = avg_gain / avg_loss
    df['RSI_10'] = 100 - (100 / (1 + rs))
    return df

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()

    # The following are not nessessary for our model
    return df.drop(columns=['Close', 'Open', 'High', 'Low'])
    