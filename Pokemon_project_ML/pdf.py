import pandas as pd
import numpy as np
from functions import normalize_values

df = pd.read_csv("pokemon.csv")

type_gen = df['Generation']

pdf_type_gen = normalize_values(type_gen.max(), type_gen.min(), type_gen)
