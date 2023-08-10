import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("pokemon.csv")

types_Series = df['Type 1']
types_pokemon = []

for type_pok in types_Series:
    if type_pok not in types_pokemon:
        types_pokemon.append(type_pok)
    else:
        continue

types_pokemon.sort()

hash_type = {}
i = 1
for type_pok in types_pokemon:
    hash_type[type_pok] = i
    i += 1

for type_pok in types_pokemon:
    df['Type 1'] = df['Type 1'].replace(type_pok, hash_type[type_pok])
    df['Type 2'] = df['Type 2'].replace(type_pok, hash_type[type_pok])

df['Type 2'] = df['Type 2'].fillna(0)

# %%
corr = df[df.columns[4:13]].corr()

# plot the heatmap
sns.heatmap(corr, cmap="crest")
print(corr)
plt.show()
