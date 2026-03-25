from data.PM10_processing import PM10_processing, PM10_data, split_data
from data.PM10_and_weather_processing import PM10_and_weather_processing, PM10_and_weather_data, split_PM10_and_weather_data
from model.Linear import Linear_model as linear, train_test
from model.Random_PM10 import Random_PM10_model as random, train_test
from model.Random_PM10 import *


pm10_data = PM10_processing()
train, test = split_data(pm10_data)
x_train, y_train_1, y_train_3, x_test, y_test_1, y_test_3 = PM10_data(train, test)

pm10_weather_data = PM10_and_weather_processing()
train_weather, test_weather = split_PM10_and_weather_data(pm10_weather_data)
x_train_weather, y_train_weather_1, y_train_weather_3, x_test_weather, y_test_weather_1, y_test_weather_3 = PM10_and_weather_data(train_weather, test_weather)

linear_pm10_result_1 = linear.train_test(x_train, y_train_1)
linear_pm10_result_3 = linear.train_test(x_train, y_train_3)

linear_pm10_weather_result_1 = linear.train_test(x_train_weather, y_train_weather_1)
linear_pm10_weather_result_3 = linear.train_test(x_train_weather, y_train_weather_3)

# random의 best_params. 이 부분은 기존에 여러번 실행하여 얻은 최적의 파라미터.
best_params1 = {
    'n_estimators': [500],
    'min_samples_split': [5],
    'min_samples_leaf': [3],
    'max_features': [None],
    'max_depth': [8]
}
best_params3 = {
    'n_estimators': [600,700],
    'min_samples_split': [5,10],
    'min_samples_leaf': [5,7],
    'max_features': [None],
    'max_depth': [8,10]
}

random_result_1 = random.train_test(x_train, y_train_1, best_params1)
random_result_3 = random.train_test(x_train, y_train_3, best_params3)

print("1시간 후 PM10 예측 결과:")
result_1 = train_test(x_train, y_train_1)
print("3시간 후 PM10 예측 결과:")
result_3 = train_test(x_train, y_train_3)

print("1시간 후 PM10 예측 결과 (날씨 포함):")
result_weather_1 = train_test(x_train_weather, y_train_weather_1)
print("3시간 후 PM10 예측 결과 (날씨 포함):")
result_weather_3 = train_test(x_train_weather, y_train_weather_3)
