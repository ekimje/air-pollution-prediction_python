import pandas as pd
import os
from datetime import datetime, timedelta

# 다운 받은 파일들 excel -> csv 파일 변경
input_files = 'C:/air_pollution_project_python/airKorea_execl'

for root, dirs, files in os.walk(input_files):
    for file in files:
        if file.endswith(".xls"):
            file_path = os.path.join(root, file)
            
            df = pd.read_excel(file_path, header = 3)
            
            filtered_df = df[df['측정망']=='도시대기']
            if "측정망" in filtered_df.columns:
                filtered_df = filtered_df.drop(columns=["측정망"])
            
            filtered_df = filtered_df[~filtered_df['측정소명'].str.contains('강화|백령', na=False)]
            
            columns = ["측정소명"]+[f'{i}시' for i in range(1,25)]
            filtered_df = filtered_df[columns]
            
            save_name = file.replace(".xls",".csv")
            save_path = os.path.join(root,save_name)
            
            filtered_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            
# 데이터 전처리
start_date = datetime(2026,1,1)
file_list =[] # pd.concat

for root, dirs, files in os.walk(input_files):
    csv_file = []
    for fname in files:
        if fname.endswith(".csv"):
            csv_file.append(fname)
            
    csv_file = sorted(csv_file)
    
    if not csv_file:
        continue
    
    for idx, file in enumerate(csv_file):
        file_path = os.path.join(root, file)
        df = pd.read_csv(input)
        
        df["측정소명"] = df["측정소명"].str.replace("[","",regex=False).str.replace("]","",regex=False)
        df["지역"] = df["측정소명"].str.split().str[0]
        
        hour_cols = [f'{i}시' for i in range(1,25)]
        df[hour_cols] = df[hour_cols].apply(pd.to_numeric,errors='coerce')
        columns_mean = df.groupby('지역')[hour_cols].mean().reset_index()
        
        file_melt = columns_mean.melt(
            id_vars = '지역',
            value_vars=hour_cols,
            var_name='hour',
            value_name='PM10')

        file_melt['hour'] = file_melt['hour'].str.replace('시','').astype(int)

        current_date = start_date + timedelta(days=idx)

        file_melt['datetime'] = (pd.to_datetime(current_date)+pd.to_timedelta(file_melt['hour']-1,unit='h'))

        file_melt = file_melt[['지역','datetime','PM10']]

        file_list.append(file_melt)

    pm_df = pd.concat(file_list,ignore_index=True)
    pm_df.to_csv("PM_processed.csv",index=False)
        
# 기상 데이터와 미세 먼지의 공통 데이터 컬럼 맞추기. 지역과 시간.

df =pd.read_csv('C:/air_pollution_project_python/OBS_ASOS_TIM_20260318081505.csv', encoding='cp949')

df=df.drop('지점',axis=1)
print(df.columns)
df = df.rename(columns={'지점명':'지역','일시':'datetime','기온(°C)':'temp','풍속(m/s)':'wind','습도(%)':'humidity'})

pm_df['datetime'] = pd.to_datetime(pm_df['datetime'])
df['datetime'] = pd.to_datetime(df['datetime'])


print(pm_df)
print(df)

df.to_csv('weather_processed.csv',encoding="utf-8-sig")

# 두 데이터 merge. 특정 열을 기준으로 정렬. 지역과 datetime 일치

merge_df = pd.merge(df, pm_df, on=['지역','datetime'],how='inner')
merge_df.to_csv("weather_pm.csv")