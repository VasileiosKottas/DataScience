import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# %% Conversion of the class to a dataframe

def load2df(data):
    df_data = pd.DataFrame(data.data, columns=data.feature_names)
    df_target = pd.DataFrame(data.target, columns=['target'])
    df_target_names = pd.DataFrame(data.target_names, columns=['target_names'])
    return df_data, df_target, df_target_names


# %% min - max Normalization:
# df: dataframe i want to Normalize
# D: The space of Normalization (default = [0, 1])
# maxmin_val: It's a 2 x n_columns Metrix that has the min and max we need
def min_max_norm(df, D=[0, 1], maxmin_val=None):
    size = df.shape
    a, b = D
    for i in range(0, size[1]):
        name_col = df.columns[i]
        if maxmin_val == None:
            max_val = df[name_col].max()
            min_val = df[name_col].min()
        else:
            max_val = maxmin_val[0][i]
            min_val = maxmin_val[1][i]
        for j in range(0, size[0]):
            df.loc[j, name_col] = a + (df.loc[j, name_col] - min_val) * (b - a) / (max_val - min_val)


# %% mean Normalization:

def mean_norm(df, maxmin_val=None):
    size = df.shape
    for i in range(0, size[1]):
        name_col = df.columns[i]
        if maxmin_val == None:
            max_val = df[name_col].max()
            min_val = df[name_col].min()
        else:
            max_val = maxmin_val[0][i]
            min_val = maxmin_val[1][i]
        aver_val = df[name_col].mean()
        for j in range(0, size[0]):
            df.loc[j, name_col] = (df.loc[j, name_col] - aver_val) / (max_val - min_val)


# %% mean Normalization:

def Z_score(df):
    size = df.shape
    for i in range(0, size[1]):
        name_col = df.columns[i]
        std_val = df[name_col].std()
        aver_val = df[name_col].mean()
        for j in range(0, size[0]):
            df.loc[j, name_col] = (df.loc[j, name_col] - aver_val) / std_val


data = load_iris()
df_data, df_target, df_target_names = load2df(data)
df_data = df_data[[df_data.columns[0], df_data.columns[1]]].head(20)

# %% Plots scatter
df_data.plot(kind='scatter', x=df_data.columns[0], y=df_data.columns[1])
plt.show()
# min-max
df_data_norm = df_data.copy()
min_max_norm(df_data_norm)
df_data_norm.plot(kind='scatter', x=df_data.columns[0], y=df_data.columns[1])
plt.show()
# mean
df_data_norm = df_data.copy()
mean_norm(df_data_norm)
df_data_norm.plot(kind='scatter', x=df_data.columns[0], y=df_data.columns[1])
plt.show()
# Z_score
df_data_norm = df_data.copy()
Z_score(df_data_norm)
df_data_norm.plot(kind='scatter', x=df_data.columns[0], y=df_data.columns[1])
plt.show()

# %% Plots bar
# min-max
df_data_norm1 = df_data.copy()
min_max_norm(df_data_norm1)
# mean
df_data_norm2 = df_data.copy()
mean_norm(df_data_norm2)
# Z_score
df_data_norm3 = df_data.copy()
Z_score(df_data_norm3)
# df plot bar
data_bar = {'min_max': df_data_norm1[df_data_norm1.columns[0]],
            'mean': df_data_norm2[df_data_norm2.columns[0]],
            'Z_score': df_data_norm3[df_data_norm3.columns[0]]}
df_bar = pd.DataFrame(data_bar)
df_bar.plot(kind='bar')
plt.show()
