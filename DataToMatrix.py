import pandas as pd
import numpy as np

# Reads Excel file
rawData = pd.read_excel("CleanDataSample.xlsx", "Sheet1")

unlabelData = rawData.iloc[2:,1:] 

# Convert DataFrame to numpy array
fullData = unlabelData.to_numpy()

print(rawData.tail())
print(rawData.shape)

print(unlabelData.tail())
print(unlabelData.shape)

print(fullData.shape) # Questions about cleaned data: Seems to have a repeat halfway at row 2450 at value no. 2447, and resets to no. 201

