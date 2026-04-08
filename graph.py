import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 알기 쉬운 기준을 위해 mae만 그래프로 생성. 이때 모든 기준, 모델별 입력하기.
df = pd.read_csv('outputs/report_with_gap1.csv')

# 행 필터림
df = df[df['model'].str.contains('val')].copy()
df = df.sort_values(by='Data',ascending=False)
print(df)

x_list = ['PM10 t+1','PM10 t+3','PM10 and Weather t+1','PM10 and Weather t+3']
val = ['val_mae','val_rmse','val_r2']
# 그래프 생성

plt.figure(figsize=(6,4))

for v in val:
    plt.plot(x_list, df[v], marker='o', label=v)

plt.title('Model Performance')
plt.xlabel('Data Condition')
plt.ylabel('Performance Metric')
plt.legend()
plt.grid()
plt.show()