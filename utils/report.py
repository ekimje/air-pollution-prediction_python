#전달받은 딕셔너리를 바탕으로 표 출력
import json
import pandas as pd

def save_csv(result):
    df = pd.DataFrame(result)
    df.to_csv('outputs/report1.csv', index=False)
    print(df)
    
def save_json(result):
    with open('outputs/report1.json', 'w') as f:
        json.dump(result, f, indent=4)
        
def save_gap(result):
    df = pd.DataFrame(result)
    df['mae_gap'] = df['val_mae'] - df['train_mae']
    df['rmse_gap'] = df['val_rmse'] - df['train_rmse']
    df['r2_gap'] = df['train_r2'] - df['val_r2']
    df.to_csv('outputs/report_with_gap1.csv', index=False)