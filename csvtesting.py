import pandas as pd
import numpy as np

# Read the CSV file into a pandas DataFrame
col_names = ['timestamp','power','cadence','label','time_since_last_shift','average_grade','speed']


df = pd.read_csv("iris.csv", skiprows=1, header=None, names=col_names)


# Remove the specified column from the DataFrame
df = df.drop('timestamp', axis=1)


# Write the updated DataFrame to a new CSV file
df.to_csv('iris.csv', index=False)

