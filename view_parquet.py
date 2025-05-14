import os
import pandas as pd


# Load your dataset (replace with your actual file path)
df = pd.read_parquet('./BLINK/Art_Style/val-00000-of-00001.parquet', engine='pyarrow')

print(df.head().loc[:, df.columns != "image_1"])

