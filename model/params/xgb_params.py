def xgb_pm10_params():
    best_params1 = {
        'n_estimators': [150, 200, 250],
        'learning_rate': [0.1],
        'max_depth': [3],
        'subsample': [0.7],
        'min_child_weight': [5, 7, 10],
        'colsample_bytree': [1.0],
        'gamma': [0.3]
    }

    best_params3 = {
        'n_estimators': [300, 350],
        'learning_rate': [0.05],
        'max_depth': [2, 3],
        'subsample': [0.6],
        'min_child_weight': [12, 13, 14],
        'colsample_bytree': [0.9],
        'gamma': [3],
        'reg_alpha': [0.5, 1],
        'reg_lambda': [1]
    }
    return best_params1, best_params3

def xgb_pm10_weather_params():
    best_params1 = {
        'n_estimators': [200, 300, 400],
        'learning_rate': [0.05, 0.07],
        'max_depth': [3],
        'subsample': [0.8],
        'min_child_weight': [5, 7, 10],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0.3, 1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 3]
    }

    best_params3 = {
        'n_estimators': [200, 300, 700],
        'learning_rate': [0.03, 0.05],
        'max_depth': [2],
        'subsample': [0.6],
        'min_child_weight': [17],
        'colsample_bytree': [0.8],
        'gamma': [3, 5],
        'reg_alpha': [0.1],
        'reg_lambda': [1, 3]
    }
    return best_params1, best_params3