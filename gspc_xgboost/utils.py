# imports
import pandas as pd
import numpy as np

# metrics
from sklearn.metrics import accuracy_score

YEARS = ['2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01',
         '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01',
         '2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01']

def create_features(df):
  df = df.copy()

  years = df.index.year
  days  = df.index.day
  min_year = 1993
  max_year = 2025
  year_range = max_year - min_year + 1
  year_norm = (years - min_year) / year_range
  
  df['year_sin'] = np.sin(2 * np.pi * year_norm)
  df['year_cos'] = np.cos(2 * np.pi * year_norm)

  month_len = df.index.days_in_month
  day_norm = (days - 1) / month_len
  df['dayofmonth_sin'] = np.sin(2 * np.pi * day_norm)
  df['dayofmonth_cos'] = np.cos(2 * np.pi * day_norm)

  df['dayofweek_sin'] = np.sin(2*np.pi * df.index.dayofweek/7)
  df['dayofweek_cos'] = np.cos(2*np.pi * df.index.dayofweek/7)
  df['month_sin'] = np.sin(2*np.pi * df.index.month/12)
  df['month_cos'] = np.cos(2*np.pi * df.index.month/12)
  df['dayofyear'] = np.sin(2*np.pi * df.index.dayofyear/365)
  df['weekofyear_sin'] = np.sin(2*np.pi * df.index.isocalendar().week/52)
  df['weekofyear_cos'] = np.sin(2*np.pi * df.index.isocalendar().week/52)
  return df

def model_performance_by_year (dataset, scaler, model):
  gainlist = []

  for i in range(len(YEARS)-1):
    start = YEARS[i]
    end = YEARS[i+1]

    X_train_year = dataset.loc[:start, :].drop(columns='Target') 
    X_test_year = dataset.loc[start:end,:].drop(columns='Target')

    y_train_year = dataset.loc[:start, 'Target']
    y_test_year = dataset.loc[start:end, 'Target']

    X_train_year = scaler.fit_transform(X_train_year)
    X_test_year = scaler.transform(X_test_year)

    model.fit(X_train_year, y_train_year)
    y_hat_year = model.predict(X_test_year)
    accuracy = accuracy_score(y_test_year, y_hat_year)
    guess = sum(y_test_year)/len(y_test_year)
    gain = 100 * (accuracy - guess)
    date = YEARS[i]
    print(f"Year {date[:4]} - model accuracy: {accuracy:.4f} - guess accuracy: {guess:.4f} - difference: {gain:.4f}")
    
    gainlist.append(gain)
  return gainlist