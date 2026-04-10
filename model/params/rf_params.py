# random의 best_params. 이 부분은 기존에 여러번 실행하여 얻은 최적의 파라미터.

def rf_pm10_params():
    best_params1 = {
        'n_estimators': [300,500],
        'min_samples_split': [5,10,15],
        'min_samples_leaf': [5,7,10],
        'max_features': ['sqrt', None,0.5],
        'max_depth': [5,6,7]
    }
    best_params3 = {
        'n_estimators': [300,500],
        'min_samples_split': [10,15,20],
        'min_samples_leaf': [7,10,15],
        'max_features': [None],
        'max_depth': [5,6,7,8]
    }
    return best_params1, best_params3

def rf_pm10_weather_params():
    best_params1={
    'n_estimators':[300,500],
    'max_depth':[4,5,6],
    'min_samples_split':[10,15],
    'min_samples_leaf':[7,10,15],
    'max_features':['sqrt',None,0.5]
    }

    best_params3={
        'n_estimators':[300,500],
        'max_depth':[4,5,6,7],
        'min_samples_split':[10,15,20],
        'min_samples_leaf':[10,15],
        'max_features':[None,'sqrt',0.5]
    }
    return best_params1, best_params3