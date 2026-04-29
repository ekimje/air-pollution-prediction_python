#전달받은 딕셔너리를 바탕으로 표 출력
import json
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path('outputs')
REPORT_CSV = OUTPUT_DIR / 'report1.csv'
REPORT_JSON = OUTPUT_DIR / 'report1.json'
REPORT_CAP_CSV = OUTPUT_DIR / 'report_with_gap1.csv'

def _ensure_output_dir()->None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_csv(result):
    _ensure_output_dir()
    df = pd.DataFrame(result)
    df.to_csv(REPORT_CSV, index=False)
    print(df)
    
def save_json(result):
    _ensure_output_dir()
    with open(REPORT_JSON, 'w') as f:
        json.dump(result, f, indent=4)
        
def save_gap(result):
    _ensure_output_dir()
    df = pd.DataFrame(result)
    df['mae_gap'] = df['val_mae'] - df['train_mae']
    df['rmse_gap'] = df['val_rmse'] - df['train_rmse']
    df['r2_gap'] = df['train_r2'] - df['val_r2']
    df.to_csv(REPORT_CAP_CSV, index=False)