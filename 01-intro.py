import numpy as np
import pandas as pd

# Question 1. Pandas version
print(pd.__version__)

# Load the data
df = pd.read_csv("/Users/ceren/Desktop/mlzcamp/01-intro/laptops.csv")

# Question 2. Records count
print(df.info())

# Question 3.Laptop brands
print(len(df["Brand"].unique()))

# Question 4. Missing values
print(df.isnull().sum())

# Question 5. Maximum final price
dell_df = df[df["Brand"] == "Dell"]
print(dell_df["Final Price"].max())

# Question 6. Median value of Screen
print(df["Screen"].median())
print(df["Screen"].mode())
fillna_df = df.fillna(df["Screen"].mode())
print(fillna_df["Screen"].median())

# Question 7. Sum of weights
df_innjoo = df[df["Brand"] == "Innjoo"]
print(df_innjoo)
df_innjoo_selected = df_innjoo[["RAM", "Storage", "Screen"]]
print(df_innjoo_selected)
X = df_innjoo_selected.values
print(X)
XTX = X.T.dot(X)
print(XTX)
inv_XTX = np.linalg.inv(XTX)
print(inv_XTX)
y = [1100, 1300, 800, 900, 1000, 1100]
w = (inv_XTX @ X.T) @ y
print(w)
print(sum(w))
