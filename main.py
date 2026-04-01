from data.PM10_processing import PM10_processing, PM10_data, split_data
from data.PM10_and_weather_processing import PM10_and_weather_processing, PM10_and_weather_data, split_data
from model.Linear_model import train_test as linear
from model.Random_model import train_test as random
from model.XGB_model import train_test as xgb
from model.params.rf_params import rf_pm10_params as pm10_params
from model.params.rf_params import rf_pm10_weather_params as pm10_weather_params
from model.params.xgb_params import xgb_pm10_params as xgb_pm10_params
from model.params.xgb_params import xgb_pm10_weather_params as xgb_pm10_weather_params
from report import report
# import data.data as data # excel -> csv 파일 변환
import json

pm10_data = PM10_processing()
train, test = split_data(pm10_data)
x_train, y_train_1, y_train_3, x_test, y_test_1, y_test_3 = PM10_data(train, test)

result=[]

pm10_weather_data = PM10_and_weather_processing()
train_weather, test_weather = split_data(pm10_weather_data)
x_train_weather, y_train_weather_1, y_train_weather_3, x_test_weather, y_test_weather_1, y_test_weather_3 = PM10_and_weather_data(train_weather, test_weather)

linear_pm10_result_1 = linear(x_train, y_train_1)
linear_pm10_result_3 = linear(x_train, y_train_3)

linear_pm10_weather_result_1 = linear(x_train_weather, y_train_weather_1)
linear_pm10_weather_result_3 = linear(x_train_weather, y_train_weather_3)

rf_params1, rf_params3 = pm10_params()
rf_weather_params1, rf_weather_params3 = pm10_weather_params()

random_pm10_result1 = random(x_train, y_train_1, rf_params1)
random_pm10_result3 = random(x_train, y_train_3, rf_params3)
random_pm10_weather_result1 = random(x_train_weather, y_train_weather_1, rf_weather_params1)
random_pm10_weather_result3 = random(x_train_weather, y_train_weather_3, rf_weather_params3)

xgb_params1, xgb_params3 = xgb_pm10_params()
xgb_weather_params1, xgb_weather_params3 = xgb_pm10_weather_params()

xgb_pm10_result1 = xgb(x_train, y_train_1, xgb_params1)
xgb_pm10_result3 = xgb(x_train, y_train_3, xgb_params3)
xgb_pm10_weather_result1 = xgb(x_train_weather, y_train_weather_1, xgb_weather_params1)
xgb_pm10_weather_result3 = xgb(x_train_weather, y_train_weather_3, xgb_weather_params3)

result.append({
    "model":"Linear Regression t+1",
    "Data":"PM10",
    **linear_pm10_result_1
})

result.append({
    "model":"Linear Regression t+3",
    "Data":"PM10",
    **linear_pm10_result_3
})

result.append({
    "model":"Linear Regression t+1",
    "Data":"PM10 and Weather",
    **linear_pm10_weather_result_1
})

result.append({
    "model":"Linear Regression t+3",
    "Data":"PM10 and Weather",
    **linear_pm10_weather_result_3
})

result.append({
    "model":"Random Forest t+1",
    "Data":"PM10",
    **random_pm10_result1
})

result.append({
    "model":"Random Forest t+3",
    "Data":"PM10",
    **random_pm10_result3
})

result.append({
    "model":"Random Forest t+1",
    "Data":"PM10 and Weather",
    **random_pm10_weather_result1
})

result.append({
    "model":"Random Forest t+3",
    "Data":"PM10 and Weather",
    **random_pm10_weather_result3
})

result.append({
    "model":"XGBoost t+1",
    "Data":"PM10",
    **xgb_pm10_result1
})

result.append({
    "model":"XGBoost t+3",
    "Data":"PM10",
    **xgb_pm10_result3
})

result.append({
    "model":"XGBoost t+1",
    "Data":"PM10 and Weather",
    **xgb_pm10_weather_result1
})

result.append({
    "model":"XGBoost t+3    ",
    "Data":"PM10 and Weather",
    **xgb_pm10_weather_result3
})

with open('result2.json','w',encoding = 'utf-8') as file:
    json.dump(result,file,ensure_ascii=False, indent=4)
    
report(result)