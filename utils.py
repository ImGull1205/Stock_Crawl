import pandas as pd



def bollinger_bands(series: pd.Series, length: int = 20, *, num_stds: tuple[float, ...] = (2, 0, -2), prefix: str = '') -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/74283044/
    rolling = series.rolling(length)
    bband0 = rolling.mean()
    bband_std = rolling.std(ddof=0)
    return pd.DataFrame({f'{prefix}{num_std}': (bband0 + (bband_std * num_std)) for num_std in num_stds})

def expand_data(data: pd.DataFrame, col= 'Close'):
    
    data_bbands = bollinger_bands(data[col], prefix='bband_')
    for i in range(20):
        data_bbands.iloc[i] = 0
    data_bbands.rename(columns={'bband_2': 'upper_band', 'bband_0': 'mid_band', 'bband_-2': 'lower_band'}, inplace=True)
    data_bbands = data_bbands.drop(columns='mid_band', axis=1)
    data = pd.concat([data, data_bbands], axis=1)
    return data