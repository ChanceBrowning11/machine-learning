# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# metrics
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import statistics as stats

# model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform
import xgboost as xgb

# utils
from data_processing import *
from utils import *

# data preprocessing
dataset = load_raw('data.csv')
dataset = add_return_target(dataset)
dataset = add_pct_change(dataset, 46)
dataset = add_rolling_features(dataset, windows=[3,5,10,20])
dataset = finalize(dataset)

# Reindex for Timeseries
dataset = dataset.set_index('Datetime_x')
dataset.index = pd.to_datetime(dataset.index)

# feature engineering
dataset = create_features(dataset)

# train test split
test_size = 0.2
X = dataset.drop(columns=['Target'])
y = (dataset['Target'] > 0).astype(int)

print(y.value_counts())
print(f"% of days where Return > 0%: {(len(y[y == 1])/len(y)*100):.2f}%")

X_train, X_test = X.loc[:'2022-01-01'], X.loc['2022-01-01':]
y_train, y_test = y.loc[:'2022-01-01'], y.loc['2022-01-01':]

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)



# RANDOM SEARCH
# Commented out becuase it takes a long time to run
# xgb_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     tree_method='hist',
#     random_state=42,
# )

# param_dist = {
#     'n_estimators':     randint(200, 1500),  # any int from 200-1500
#     'max_depth':        randint(3, 12),      # 3-12
#     'learning_rate':    uniform(0.001, 0.2), # between 0.001-0.201
#     'subsample':        uniform(0.5, 0.5),   # 0.5-1.0
#     'colsample_bytree': uniform(0.5, 0.5),   # 0.5-1.0
#     'gamma':            uniform(0, 1),       # 0-1
#     'reg_alpha':        uniform(0, 1),       # 0-1
#     'reg_lambda':       uniform(1, 10),      # 1-11
#     'min_child_weight': randint(1, 10),      # any int from 1-10
#     'max_delta_step':   randint(0, 5),       # any int from 0-5
#     'scale_pos_weight': uniform(0.8, 1.2),   # 0.8 - 1.2
# }

# rs = RandomizedSearchCV(
#     xgb_model,
#     param_distributions=param_dist,
#     n_iter=200,
#     cv=tscv,
#     scoring=['accuracy', 'roc_auc'],
#     refit='accuracy',
#     n_jobs=-1,
#     random_state=42,
#     verbose=1
# )
# rs.fit(X_train, y_train)
# print(f"Best Accuracy: {rs.best_score_:.4f}")
# print("Params:", rs.best_params_)

# rs_model = rs.best_estimator_
# search = rs



# GRID SEARCH
# Commented out because it takes a long time to run
# xgb_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     tree_method='hist',
#     random_state=42,
#     early_stopping_rounds=20,
# )

# param_grid = {
#     'learning_rate':    [0.002, 0.005],
#     'n_estimators':     [300, 500],
#     'max_depth':        [5, 6, 7],
#     'min_child_weight': [8], # [3, 5, 7],
#     'gamma':            [0.7, 0.9],
#     'subsample':        [0.6, 0.8],
#     'colsample_bytree': [0.6, 0.7],
#     'reg_alpha':        [0.2], # [0.2, 0.58, 1.0],
#     'reg_lambda':       [3],
#     # drop or tighten scale_pos_weight
#     # 'scale_pos_weight': [1.0, 1.2] 
# }

# # 1) Fit the grid as you already do, then grab the best params
# gs = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     cv=tscv,
#     scoring='accuracy',   # or your custom scorer
#     n_jobs=-1,
#     verbose=1,
# )
# gs.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

# best_params = gs.best_params_
# model = gs.best_estimator_
# print("Best hyperparameters:", best_params)

# RESULTS - after running random search into grid search
# Best hyperparameters: {'colsample_bytree': 0.7, 'gamma': 0.7, 'learning_rate': 0.002, 'max_depth': 7, 'min_child_weight': 8, 'n_estimators': 300, 'reg_alpha': 0.2, 'reg_lambda': 3, 'subsample': 0.6}



# Model with best hyperparameters
best_params =  {'colsample_bytree': 0.7, 'gamma': 0.7, 'learning_rate': 0.002, 'max_depth': 7, 'min_child_weight': 8, 'n_estimators': 300, 'reg_alpha': 0.2, 'reg_lambda': 3, 'subsample': 0.6}

model = xgb.XGBClassifier(
    objective='binary:logistic',
    tree_method='hist',
    random_state=42,
    **best_params
)

model.fit(X_train, y_train)

# Metrics
y_hat = model.predict(X_test)

acc = accuracy_score(y_test, y_hat)
print(f"accuracy: {acc:.2f}")

# Feature importance
fi = pd.DataFrame(data= model.feature_importances_,
                  index= X.columns,
                  columns=['importance'])
fi.sort_values('importance').plot(figsize=(12,20), kind='barh', title='Feature Importance')
plt.show()

# Performance over the years
print('Metrics for model with all features')
gainlist =  model_performance_by_year(dataset, scaler, model)
print(f"Mean: {stats.mean(gainlist):.3f}")
print(f"Median: {stats.median(gainlist):.3f}")

# Dropping features to see if performance improves
dropped_features = fi.sort_values('importance', ascending=False).index[10:]
X_dropped = dataset.drop(columns=dropped_features)

X_train_dropped = X_dropped.loc[:'2022-01-01'].drop(columns=['Target'])
X_test_dropped = X_dropped.loc['2022-01-01':].drop(columns=['Target'])

model.fit(X_train_dropped, y_train)
y_hat_dropped = model.predict(X_test_dropped)
accuracy = accuracy_score(y_test, y_hat)
print(f"Accuracy after using dropped columns: {accuracy}")

fi = pd.DataFrame(data= model.feature_importances_,
                  index= X_train_dropped.columns,
                  columns=['importance'])
fi.sort_values('importance').plot(figsize=(12,12), kind='barh', title='Feature Importance')
plt.show()

print('Metrics for model with less features')
gainlist =  model_performance_by_year(dataset, scaler, model)
print(f"Mean: {stats.mean(gainlist):.3f}")
print(f"Median: {stats.median(gainlist):.3f}")