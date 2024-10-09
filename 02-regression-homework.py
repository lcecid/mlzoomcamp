import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv("/Users/ceren/Desktop/mlzcamp/01-intro/laptops.csv")
print(df.head())

# Prepare the data
df.columns = df.columns.str.lower().str.replace(" ", "_")
df = df[['ram', 'storage', 'screen', 'final_price']]
print(df.head())

# EDA
# Question 1
print(df.isnull().sum())

# Question 2
print(np.median(df['ram']))

n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = int(n * 0.6)
print(n, n_val + n_test + n_train)
df_train = df.iloc[: n_train]
df_test = df.iloc[n_train + n_val :]
df_val = df.iloc[n_train : n_train + n_val]

idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)
df_train = df.iloc[idx[: n_train]]
df_test = df.iloc[idx[n_train + n_val :]]
df_val = df.iloc[idx[n_train : n_train + n_val]]
print(df_train.head())
print(df_test.head())
print(df_val.head())

# Question 3
# Fill with 0
df_train_zero = df_train.fillna(0)
df_val_zero = df_val.fillna(0)

X_train_zero = df_train_zero[['ram', 'storage', 'screen']]
y_train_zero = df_train_zero['final_price']

X_val_zero = df_val_zero[['ram', 'storage', 'screen']]
y_val_zero = df_val_zero['final_price']

model_zero = LinearRegression()
model_zero.fit(X_train_zero, y_train_zero)

y_pred_zero = model_zero.predict(X_val_zero)

rmse_zero = np.sqrt(mean_squared_error(y_val_zero, y_pred_zero))

print(f"RMSE with 0: {round(rmse_zero, 2)}")

# Fill with mean
mean_screen = df_train['screen'].mean()
df_train_mean = df_train.fillna(mean_screen)
df_val_mean = df_val.fillna(mean_screen)

X_train_mean = df_train_mean[['ram', 'storage', 'screen']]
y_train_mean = df_train_mean['final_price']

X_val_mean = df_val_mean[['ram', 'storage', 'screen']]
y_val_mean = df_val_mean['final_price']

model_mean = LinearRegression()
model_mean.fit(X_train_mean, y_train_mean)

y_pred_mean = model_mean.predict(X_val_mean)

rmse_mean = np.sqrt(mean_squared_error(y_val_mean, y_pred_mean))

print(f"RMSE with mean: {round(rmse_mean, 2)}")
print("The RMSE is lower when we fill the missing values with the 0 of the screen column.")

# Question 4
def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

for r in [0, 0.01, 1, 0.10, 100]:
    w_0, w = train_linear_regression_reg(X_train_zero, y_train_zero, r=r)
    y_pred_val = w_0 + X_val_zero.dot(w)
    rmse_val = np.round(rmse(y_val_zero, y_pred_val), 2)
    print(r, w_0, rmse_val)

# Question 5
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]

def prepare_X(df, fillna_value=0):
    df = df.fillna(fillna_value)
    X = df.values
    return X

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

df = pd.read_csv("/Users/ceren/Desktop/mlzcamp/01-intro/laptops.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")
df = df[['ram', 'storage', 'screen', 'final_price']]

n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

rmse_list = []
for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    df_shuffled = df.iloc[idx]
    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train:n_train + n_val].copy()
    df_test = df_shuffled.iloc[n_train + n_val:].copy()
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train['final_price'].values
    y_val = df_val['final_price'].values
    y_test = df_test['final_price'].values

    del df_train['final_price']
    del df_val['final_price']
    del df_test['final_price']

    X_train = prepare_X(df_train, fillna_value=0)
    X_val = prepare_X(df_val, fillna_value=0)

    w_0, w = train_linear_regression(X_train, y_train)
    y_pred_val = w_0 + X_val.dot(w)
    rmse_val = np.round(rmse(y_val, y_pred_val), 2)
    rmse_list.append(rmse_val)
    print(seed, w_0, rmse_val)

std_rmse = np.round(np.std(rmse_list), 3)
print(f"Standard deviation of RMSE scores: {std_rmse}")

# Question 6
idx = np.arange(n)
np.random.seed(9)
np.random.shuffle(idx)
df_shuffled = df.iloc[idx]
df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train + n_val].copy()
df_test = df_shuffled.iloc[n_train + n_val:].copy()

df_train_val = pd.concat([df_train, df_val]).reset_index(drop=True)

df_train_val = df_train_val.fillna(0)
df_test = df_test.fillna(0)

X_train_val = df_train_val[['ram', 'storage', 'screen']].values
y_train_val = df_train_val['final_price'].values

X_test = df_test[['ram', 'storage', 'screen']].values
y_test = df_test['final_price'].values

def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]

r = 0.001
w_0, w = train_linear_regression_reg(X_train_val, y_train_val, r=r)
y_pred_test = w_0 + X_test.dot(w)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
rmse_test_rounded = np.round(rmse_test, 2)

print(f"RMSE on test dataset: {rmse_test_rounded}")