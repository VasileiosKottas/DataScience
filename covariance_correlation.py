import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
from Normalization import load2df, min_max_norm


# %% functions
def sort_eigenvalues(e, v):
    for i in range(0, e.shape[0]):
        for j in range(i + 1, e.shape[0]):
            if e[i] < e[j]:
                e[[i, j]] = e[[j, i]]
                v[:, [i, j]] = v[:, [j, i]]


# %%
data = load_breast_cancer()
df_data, df_target, df_target_names = load2df(data)
df_data = df_data

# %%
data_array = df_data.to_numpy().transpose()
covMatrix = np.cov(data_array)
corrMatrix = np.corrcoef(data_array)

# %% normalize
df_data_norm = df_data.copy()
min_max_norm(df_data_norm)
data_array_norm = df_data_norm.to_numpy().transpose()
covMatrix_norm = np.cov(data_array_norm)
corrMatrix_norm = np.corrcoef(data_array_norm)

# %% eigenvalue and eigenvector
e, v = np.linalg.eig(corrMatrix)
sort_eigenvalues(e, v)
e_sum = e.sum()
e_precent = e / e_sum
data_bar = {'eigenvalue': e_precent}
df_bar = pd.DataFrame(data_bar)
df_bar.plot(kind='bar')
plt.show()

A = v @ np.diag(e) @ np.linalg.inv(v)
