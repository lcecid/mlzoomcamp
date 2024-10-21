import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold

data = pd.read_csv("...04-evaluation/bank-full.csv", sep=";")
data.columns = data.columns.str.lower()

print(data.head(10))

data = data[[
    "age", "job", "marital", "education", "balance", "housing", "contact",
    "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"
]]

df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train["y"].map({"yes": 1, "no": 0})
y_val = df_val["y"].map({"yes": 1, "no": 0})
y_test = df_test["y"].map({"yes": 1, "no": 0})

df_train = df_train.drop(columns=["y"])
df_val = df_val.drop(columns=["y"])
df_test = df_test.drop(columns=["y"])

# Define numerical and categorical columns
numerical = ["age", "balance", "day", "duration", "previous", "pdays"]
categorical = ["job", "marital", "education", "housing", "contact", "month", "poutcome"]

print("AUC Scores for Numerical Columns:")
for c in numerical:
    auc = roc_auc_score(y_train, df_train[c])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[c])
    print(f"{c:9}: {auc:.3f}")

columns = categorical + numerical

train_dicts = df_train[columns].to_dict(orient="records")
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
model.fit(X_train, y_train)

val_dicts = df_val[columns].to_dict(orient="records")
X_val = dv.transform(val_dicts)

y_pred = model.predict_proba(X_val)[:, 1]

val_auc = roc_auc_score(y_val, y_pred)
print(f"Validation AUC: {val_auc:.3f}")

def confusion_matrix_dataframe(y_val, y_pred):
    scores = []
    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)
        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()
        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))
    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['p'] = df_scores.tp / (df_scores.tp + df_scores.fp + 1e-10)  # Avoid division by zero
    df_scores['r'] = df_scores.tp / (df_scores.tp + df_scores.fn + 1e-10)  # Avoid division by zero

    df_scores['f1'] = 2 * (df_scores['p'] * df_scores['r']) / (
                df_scores['p'] + df_scores['r'] + 1e-10)  # Avoid division by zero
    return df_scores

df_scores = confusion_matrix_dataframe(y_val, y_pred)

print(df_scores[::10])

plt.figure(figsize=(10, 6))
plt.plot(df_scores.threshold, df_scores.p, label='Precision', color='blue')
plt.plot(df_scores.threshold, df_scores.r, label='Recall', color='orange')
plt.plot(df_scores.threshold, df_scores.f1, label='F1 Score', color='green')
plt.legend()
plt.title('Precision, Recall, and F1 Score vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.grid()
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.show()

# Find the threshold with the maximum F1 Score
max_f1_index = np.argmax(df_scores.f1)
max_f1_score = df_scores.f1[max_f1_index]
optimal_threshold = df_scores.threshold[max_f1_index]

print(f"Maximum F1 Score: {max_f1_score:.3f} at threshold: {optimal_threshold:.2f}")

def train(df_train, y_train, C=1.0):
    dicts = df_train[columns].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)
    return dv, model

def predict(df, dv, model):
    dicts = df[columns].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

scores = []

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    y_train = df_train['y']
    y_val = df_val['y']
    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

mean_auc = np.mean(scores)
std_auc = np.std(scores)

print('Mean AUC: %.3f ± %.3f' % (mean_auc, std_auc))

C_values = [0.000001, 0.001, 1]

results = {}

for C in C_values:
    scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        y_train = df_train['y']  # Correctly reference the target column
        y_val = df_val['y']
        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)
    mean_auc = np.mean(scores)
    std_auc = np.std(scores)
    results[C] = (mean_auc, std_auc)
    print('C=%10s: %.3f ± %.3f' % (C, mean_auc, std_auc))

best_C = min(results.keys(), key=lambda x: (-results[x][0], results[x][1], x))
best_mean, best_std = results[best_C]

print(f'\nBest C: {best_C} with Mean AUC: {best_mean:.3f} ± {best_std:.3f}')