import matplotlib.pyplot as plt
import pandas as pd
import csv

df = pd.read_csv('outputs/report_with_gap1.csv')

# x축 -> 모델, y축 -> 지표(MAE, RMSE, R2)
# y_target_word = mae,rmse,r2 /  x_target_word = linear,random,xgb
x_target_word = ['Linear Regression val', 'Random Forest val', 'XGBoost val']
y_target_word = ['val_mae', 'val_rmse', 'val_r2']

