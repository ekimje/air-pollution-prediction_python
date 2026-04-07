from data.PM10_processing import PM10_processing, PM10_data, split_data_PM10
from data.PM10_and_weather_processing import PM10_and_weather_processing, PM10_and_weather_data, split_data_PM10_and_weather
from model.Linear_model import train_test as linear
from model.Random_model import train_test as random
from model.XGB_model import train_test as xgb
from model.params.rf_params import rf_pm10_params as pm10_params
from model.params.rf_params import rf_pm10_weather_params as pm10_weather_params
from model.params.xgb_params import xgb_pm10_params as xgb_pm10_params
from model.params.xgb_params import xgb_pm10_weather_params as xgb_pm10_weather_params
from utils.result_append import result_append as result
from utils.report import save_csv, save_json, save_gap

def run_pipeline():
    pm10_data = PM10_processing()
    train, test = split_data_PM10(pm10_data)
    x_train, y_train_1, y_train_3, x_test, y_test_1, y_test_3 = PM10_data(train, test)

    pm10_weather_data = PM10_and_weather_processing()
    train_weather, test_weather = split_data_PM10_and_weather(pm10_weather_data)
    x_train_weather, y_train_weather_1, y_train_weather_3, x_test_weather, y_test_weather_1, y_test_weather_3 = PM10_and_weather_data(train_weather, test_weather)

    rf_params1, rf_params3 = pm10_params()
    rf_weather_params1, rf_weather_params3 = pm10_weather_params()

    xgb_params1, xgb_params3 = xgb_pm10_params()
    xgb_weather_params1, xgb_weather_params3 = xgb_pm10_weather_params()
    
    linear_result_pred = linear_model(x_train, y_train_1, y_train_3,x_train_weather, y_train_weather_1, y_train_weather_3)
    linear_result_val = linear_model(x_test, y_test_1, y_test_3,x_test_weather, y_test_weather_1, y_test_weather_3)
    random_result_pred = random_model(x_train, y_train_1, y_train_3,x_train_weather, y_train_weather_1, y_train_weather_3, rf_params1, rf_params3, rf_weather_params1, rf_weather_params3)
    random_result_val = random_model(x_test, y_test_1, y_test_3,x_test_weather, y_test_weather_1, y_test_weather_3, rf_params1, rf_params3, rf_weather_params1, rf_weather_params3)
    xgb_result_pred = xgb_model(x_train, y_train_1, y_train_3,x_train_weather, y_train_weather_1, y_train_weather_3, xgb_params1, xgb_params3, xgb_weather_params1, xgb_weather_params3)
    xgb_result_val = xgb_model(x_test, y_test_1, y_test_3,x_test_weather, y_test_weather_1, y_test_weather_3, xgb_params1, xgb_params3, xgb_weather_params1, xgb_weather_params3)

    results = []
    results += result("Linear Regression pred", linear_result_pred)
    results += result("Linear Regression val", linear_result_val)
    results += result("Random Forest pred", random_result_pred)
    results += result("Random Forest val", random_result_val)
    results += result("XGBoost pred", xgb_result_pred)
    results += result("XGBoost val", xgb_result_val)

    save_csv(results)
    save_json(results)
    save_gap(results)
    return results
    
def linear_model(x_pm10, y_pm10_t1, y_pm10_t3, x_pm10_weather, y_pm10_weather_t1, y_pm10_weather_t3):
    linear_pm10_predict_1 = linear(x_pm10, y_pm10_t1)
    linear_pm10_predict_3 = linear(x_pm10, y_pm10_t3)
    linear_pm10_weather_predict_1 = linear(x_pm10_weather, y_pm10_weather_t1)
    linear_pm10_weather_predict_3 = linear(x_pm10_weather, y_pm10_weather_t3)
    
    return{"PM10 t+1":linear_pm10_predict_1, "PM10 t+3":linear_pm10_predict_3, "PM10 and Weather t+1":linear_pm10_weather_predict_1, "PM10 and Weather t+3":linear_pm10_weather_predict_3}

def random_model(x_pm10, y_pm10_t1, y_pm10_t3, x_pm10_weather, y_pm10_weather_t1, y_pm10_weather_t3, rf_params1, rf_params3, rf_weather_params1, rf_weather_params3):
    random_pm10_predict_1 = random(x_pm10, y_pm10_t1, rf_params1)
    random_pm10_predict_3 = random(x_pm10, y_pm10_t3, rf_params3)
    random_pm10_weather_predict_1 = random(x_pm10_weather, y_pm10_weather_t1, rf_weather_params1)
    random_pm10_weather_predict_3 = random(x_pm10_weather, y_pm10_weather_t3, rf_weather_params3)
    
    return{"PM10 t+1":random_pm10_predict_1, "PM10 t+3":random_pm10_predict_3, "PM10 and Weather t+1":random_pm10_weather_predict_1, "PM10 and Weather t+3":random_pm10_weather_predict_3}

def xgb_model(x_pm10, y_pm10_t1, y_pm10_t3, x_pm10_weather, y_pm10_weather_t1, y_pm10_weather_t3, xgb_params1, xgb_params3, xgb_weather_params1, xgb_weather_params3):
    xgb_pm10_predict_1 = xgb(x_pm10, y_pm10_t1, xgb_params1)
    xgb_pm10_predict_3 = xgb(x_pm10, y_pm10_t3, xgb_params3)
    xgb_pm10_weather_predict_1 = xgb(x_pm10_weather, y_pm10_weather_t1, xgb_weather_params1)
    xgb_pm10_weather_predict_3 = xgb(x_pm10_weather, y_pm10_weather_t3, xgb_weather_params3)
    
    return{"PM10 t+1":xgb_pm10_predict_1, "PM10 t+3":xgb_pm10_predict_3, "PM10 and Weather t+1":xgb_pm10_weather_predict_1, "PM10 and Weather t+3":xgb_pm10_weather_predict_3}