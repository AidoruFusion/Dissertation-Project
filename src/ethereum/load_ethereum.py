import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "..", "data", "transaction_dataset.csv")

df_eth = pd.read_csv(data_path)

print("Shape:", df_eth.shape)
print("\nColumns:\n", df_eth.columns)

print("\nCorrect Class Distribution (FLAG column):\n")
print(df_eth["FLAG"].value_counts())