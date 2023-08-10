import pandas as pd


def normalize_values(max_value, min_value, data):
    series = pd.Series()
    for val in data:
        d = pd.Series(float((val - min_value) / (max_value - min_value)))
        series = series._append(d, ignore_index=True)
    return series
