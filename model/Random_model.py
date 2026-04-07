import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

outer_tscv = TimeSeriesSplit(n_splits=10)
inner_tscv = TimeSeriesSplit(n_splits=3)

def train_test(x_train, y_train,param_dist):
    train_mae, val_mae = [], []
    train_rmse, val_rmse = [], []
    train_r2, val_r2 = [], []
    best_params_list = []
        
    for train_index, val_index in outer_tscv.split(x_train):
        tr_x, val_x = x_train.iloc[train_index], x_train.iloc[val_index]
        tr_y, val_y = y_train.iloc[train_index], y_train.iloc[val_index]

        model = RandomForestRegressor(random_state=42,n_jobs=1)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=20,
            scoring='neg_mean_absolute_error',
            verbose=1,
            n_jobs=1,
            cv=inner_tscv,
            random_state=42
        )
        random_search.fit(tr_x, tr_y)

        best_params_list.append(random_search.best_params_)

        best_model = random_search.best_estimator_

        pred_tr = best_model.predict(tr_x)
        pred_val = best_model.predict(val_x)

        train_mae.append(mean_absolute_error(tr_y, pred_tr))
        val_mae.append(mean_absolute_error(val_y, pred_val))

        train_rmse.append(root_mean_squared_error(tr_y, pred_tr))
        val_rmse.append(root_mean_squared_error(val_y, pred_val))

        train_r2.append(r2_score(tr_y, pred_tr))
        val_r2.append(r2_score(val_y, pred_val))
        
    return {
        "train_mae": np.mean(train_mae),
        "val_mae": np.mean(val_mae),
        "train_rmse": np.mean(train_rmse),
        "val_rmse": np.mean(val_rmse),
        "train_r2": np.mean(train_r2),
        "val_r2": np.mean(val_r2)
    }   