from data.PM10_processing import PM10_processing, PM10_data, split_data_PM10
from data.PM10_and_weather_processing import PM10_and_weather_processing, PM10_and_weather_data, split_data_PM10_and_weather
from model.Linear_model import train_test as linear
from model.Random_model import train_test as random
from model.XGB_model import train_test as xgb
from model.params.rf_params import rf_pm10_params, rf_pm10_weather_params
from model.params.xgb_params import xgb_pm10_params,xgb_pm10_weather_params
from utils.result_append import result_append
from utils.report import save_csv, save_json, save_gap

def run_model_pair(train_set,test_set,runner,params=None):
    if params is None:
        train_result = runner(*train_set)
        test_result = runner(*test_set)
    else:
        train_result = runner(*train_set,*params)
        test_result = runner(*test_set,*params)
    return train_result, test_result
    
def linear_model(x_pm10, y_pm10_t1, y_pm10_t3, x_pm10_weather, y_pm10_weather_t1, y_pm10_weather_t3):    
    return{
        "PM10 t+1":linear(x_pm10, y_pm10_t1),
        "PM10 t+3":linear(x_pm10, y_pm10_t3),
        "PM10 and Weather t+1":linear(x_pm10_weather, y_pm10_weather_t1),
        "PM10 and Weather t+3":linear(x_pm10_weather, y_pm10_weather_t3)
    }

def random_model(x_pm10, y_pm10_t1, y_pm10_t3, x_pm10_weather, y_pm10_weather_t1, y_pm10_weather_t3, rf_params1, rf_params3, rf_weather_params1, rf_weather_params3):    
    return{
        "PM10 t+1":random(x_pm10, y_pm10_t1, rf_params1),
        "PM10 t+3":random(x_pm10, y_pm10_t3, rf_params3),
        "PM10 and Weather t+1":random(x_pm10_weather, y_pm10_weather_t1, rf_weather_params1),
        "PM10 and Weather t+3":random(x_pm10_weather, y_pm10_weather_t3, rf_weather_params3)
    }

def xgb_model(x_pm10, y_pm10_t1, y_pm10_t3, x_pm10_weather, y_pm10_weather_t1, y_pm10_weather_t3, xgb_params1, xgb_params3, xgb_weather_params1, xgb_weather_params3):
    return{
        "PM10 t+1":xgb(x_pm10, y_pm10_t1, xgb_params1),
        "PM10 t+3":xgb(x_pm10, y_pm10_t3,xgb_params3),
        "PM10 and Weather t+1":xgb(x_pm10_weather, y_pm10_weather_t1, xgb_weather_params1),
        "PM10 and Weather t+3":xgb(x_pm10_weather, y_pm10_weather_t3, xgb_weather_params3)
    }

def run_pipeline():
    pm10_data = PM10_processing()
    train, test = split_data_PM10(pm10_data)
    pm10_train_test = PM10_data(train, test)

    pm10_weather_data = PM10_and_weather_processing()
    train_weather, test_weather = split_data_PM10_and_weather(pm10_weather_data)
    pm10_weather_train_test= PM10_and_weather_data(train_weather, test_weather)

    train_set = (*pm10_train_test[:3],*pm10_weather_train_test[:3])
    test_set = (*pm10_train_test[3:],*pm10_weather_train_test[3:])
    
    rf_params = (*rf_pm10_params(), *rf_pm10_weather_params())
    xgb_params = (*xgb_pm10_params(), *xgb_pm10_weather_params())
    
    linear_result_pred,linear_result_val = run_model_pair(train_set, test_set, linear_model)
    random_result_pred, random_result_val = run_model_pair(train_set, test_set, random_model, rf_params)
    xgb_result_pred, xgb_result_val = run_model_pair(train_set, test_set, xgb_model, xgb_params)

    results = []
    results += result_append('Linear Regression pred', linear_result_pred)
    results += result_append('Linear Regression val', linear_result_val)
    results += result_append('Random Forest pred', random_result_pred)
    results += result_append('Random Forest val', random_result_val)
    results += result_append('XGBoost pred', xgb_result_pred)
    results += result_append('XGBoost val', xgb_result_val)

    save_csv(results)
    save_json(results)
    save_gap(results)
    return results
