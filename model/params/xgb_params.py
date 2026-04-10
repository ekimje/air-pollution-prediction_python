def xgb_pm10_params():
    best_params1 = {
        'n_estimators': [300, 500, 700],
        'learning_rate': [0.03, 0.05],
        'max_depth': [2, 3],
        'subsample': [0.6, 0.7],
        'min_child_weight': [7, 10, 12],
        'colsample_bytree': [0.5, 0.7, 0.8],
        'gamma': [0.5, 1, 2],
        'reg_alpha': [0.1, 0.5, 1],
        'reg_lambda': [3, 5, 10]
    }

    best_params3 = {
            'n_estimators': [300, 500],
        'learning_rate': [0.03, 0.05],
        'max_depth': [2],
        'subsample': [0.5, 0.6],
        'min_child_weight': [12, 15, 18],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [3, 5, 7],
        'reg_alpha': [0.5, 1, 2],
        'reg_lambda': [3, 5, 10]
    }
    return best_params1, best_params3

def xgb_pm10_weather_params():
    param_dist1 = {
        'n_estimators': [300, 500, 700],
        'learning_rate': [0.03, 0.05],
        'max_depth': [2, 3],
        'subsample': [0.6, 0.7],
        'min_child_weight': [10, 15, 20],
        'colsample_bytree': [0.5, 0.6, 0.7],
        'gamma': [1, 2, 3, 5],
        'reg_alpha': [0.5, 1, 2],
        'reg_lambda': [3, 5, 10]
    }

    param_dist3 = {
        'n_estimators': [300, 500],
        'learning_rate': [0.03, 0.05],
        'max_depth': [2],
        'subsample': [0.5, 0.6],
        'min_child_weight': [15, 20],
        'colsample_bytree': [0.5, 0.6, 0.8],
        'gamma': [3, 5, 7],
        'reg_alpha': [0.5, 1, 2],
        'reg_lambda': [3, 5, 10]
    }
    return param_dist1, param_dist3