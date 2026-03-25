# 미세먼지 + 기상 데이터
import pandas as pd

def PM10_processing():
    data = pd.read_csv('C:\\air_pollution_prediction_python\\data\\weather_pm.csv', encoding='utf-8-sig')
    data['datetime'] = pd.to_datetime(data['datetime'])

    data = data.sort_values(['지역','datetime'].copy())

    columns = ['PM10','PM10','wind','temp','humidity']

    for col in columns:
        data[f'{col}_t-1'] = data.groupby('지역')[col].shift(1)
        data[f'{col}_t-2'] = data.groupby('지역')[col].shift(2)

    data['target_t+1'] = data.groupby('지역')['PM10'].shift(-1)
    data['target_t+3'] = data.groupby('지역')['PM10'].shift(-3)

    data = data.dropna()

    data = pd.get_dummies(data, columns=['지역']) # 지역 더미 변수화

    return data

def split_data(data):
    # 시계열 분리. 한... 2026년 3월 3일까지
    train = data[data['datetime']<'2026-03-03'].copy()
    test = data[data['datetime']>='2026-03-03'].copy()
    
    return train, test

def PM10_data(train, test): 
    x_train = train.drop(['datetime','target_t+1','target_t+3'], axis=1)
    y_train_1 = train['target_t+1']
    y_train_3 = train['target_t+3']

    x_test = test.drop(['datetime','target_t+1','target_t+3'], axis=1)
    y_test_1 = test['target_t+1']
    y_test_3 = test['target_t+3']

    return x_train, y_train_1, y_train_3, x_test, y_test_1, y_test_3