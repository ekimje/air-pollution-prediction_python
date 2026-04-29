import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_report(path:str)->pd.DataFrame:
    df = pd.read_csv(path)
    filtered = df[df['model'].str.contains('val')].copy()
    filtered = filtered.loc[:,['model','Data','val_mae','val_r2']].copy() # 특정 컬럼만 선택해서 복사
    filtered = filtered.sort_values(by='Data',ascending=False)
    filtered['model']=filtered['model'].str.split(' ').str[0]
    return filtered

def plot_metrics(df:pd.DataFrame, save_prefix:str | None = None)->None:
    data = df['Data'].unique()
    w=0.15
    idx = np.arange(len(data))
    
    linear = df[df['model']=='Linear'].sort_values(by='Data',ascending=False)
    random = df[df['model']=='Random'].sort_values(by='Data',ascending=False)
    xgboost = df[df['model']=='XGBoost'].sort_values(by='Data',ascending=False)
    for i in ['val_mae','val_r2']:
        plt.bar(idx-w,linear[i],width=w,color = 'gray',label='Linear')
        plt.bar(idx,random[i],width=w,color = 'lightgray',label='Random')
        plt.bar(idx+w,xgboost[i],width=w,color = 'darkgray',label='XGB')
    plt.xticks(idx,data,rotation=30)
    plt.xlabel('Data Contitions')
    plt.ylabel(i)
    plt.legend()
    plt.title(f'{i} by Data and Model')
    if save_prefix:
        plt.savefig(f'{save_prefix}_{i}.png')
    else:
        plt.show()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='validation metrics by model and dataset.')
    parser.add_argument('--report', type=str, default='outputs/report_with_gap1.csv', help='Path to the report CSV file')
    parser.add_argument('--save-prefix', default=None, help='Prefix to save images instead of showing')
    return parser.parse_args()

def main()->None:
    args = parse_args()
    report = load_report(args.report)
    plot_metrics(report, save_prefix=args.save_prefix)

if __name__=='__main__':
    main()