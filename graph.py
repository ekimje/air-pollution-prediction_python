import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 알기 쉬운 기준 비교 위해 MAE, R2 score 만 그래프 생성, 각 Data 별 모델의 막대그래프 생성
df = pd.read_csv('outputs/report_with_gap1.csv')

# 필요 데이터 필터링.
df = df[df['model'].str.contains('val')].copy()
df = df.loc[:,['model','Data','val_mae','val_r2']].copy()
df = df.sort_values(by='Data',ascending=False)
df['model']=df.model.str.split(' ').str[0]
print(df)

# 데이터 중복 제거하여 고유한 데이터 추출
data = df['Data'].unique()
print(data)

w=0.15 # 막대그래프 위치 조정용
nrow = len(data) # 데이터 개수
idx = np.arange(nrow) # x축 위치 설정
print(idx)

# 모델별 Data 순서 맞추기
linear = df[df['model']=='Linear'].sort_values(by='Data',ascending=False)
print(linear)
random = df[df['model']=='Random'].sort_values(by='Data',ascending=False)
xgboost = df[df['model']=='XGBoost'].sort_values(by='Data',ascending=False)

# 그래프 생성
for i in ['val_mae','val_r2']:
    plt.bar(idx-w,linear[i],width=w,label='Linear')
    plt.bar(idx,random[i],width=w,label='Random')
    plt.bar(idx+w,xgboost[i],width=w,label='XGB')
    plt.xticks(idx,data,rotation=30)
    plt.xlabel('Data Contitions')
    plt.ylabel(i)
    plt.legend()
    plt.title(f'{i} by Data and Model')
    plt.show()