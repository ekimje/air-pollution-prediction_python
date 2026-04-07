def xgb_pm10_params():
    best_params1= {
        'n_estimators': [300],
        'learning_rate': [0.05],
        'max_depth': [5],
        'subsample': [0.6],
        'min_child_weight': [7],
        'colsample_bytree': [0.9],
        'gamma': [3],
        'reg_alpha': [0.5],
        'reg_lambda': [1]
    }

    best_params3 = {
        'n_estimators': [200],
        'max_depth': [5],
        'learning_rate': [0.05],
        'min_child_weight': [9],
        'gamma': [3],
        'subsample': [0.6],
        'colsample_bytree': [0.9],
        'reg_alpha': [1],
        'reg_lambda': [1]
    }
    return best_params1, best_params3

def xgb_pm10_weather_params():
    best_params1 = {
        'n_estimators': [300],
        'learning_rate': [0.05],
        'max_depth': [5],
        'subsample': [0.6],
        'min_child_weight': [7],
        'colsample_bytree': [0.9],
        'gamma': [3],
        'reg_alpha': [0.5],
        'reg_lambda': [1]
    }

    best_params3 = {
        'n_estimators': [200],
        'learning_rate': [0.05],
        'max_depth': [5],
        'subsample': [0.6],
        'min_child_weight': [9],
        'colsample_bytree': [0.9],
        'gamma': [3],
        'reg_alpha': [1],
        'reg_lambda': [1]
    }
    return best_params1, best_params3