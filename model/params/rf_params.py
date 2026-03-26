# random의 best_params. 이 부분은 기존에 여러번 실행하여 얻은 최적의 파라미터.

def rf_pm10_params():
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
    return best_params1, best_params3

def rf_pm10_weather_params():
    best_params1={
        'n_estimators':[400,450,500],
        'max_depth':[8,9],
        'min_samples_split':[5],
        'min_samples_leaf':[3,7],
        'max_features':[None]
    }

    best_params3={
        'n_estimators':[500],
        'max_depth':[10],
        'min_samples_split':[10],
        'min_samples_leaf':[7],
        'max_features':[None]
    }
    return best_params1, best_params3