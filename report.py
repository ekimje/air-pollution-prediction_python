#전달받은 딕셔너리를 바탕으로 표 출력
import pandas as pd

def report(result):
    df = pd.DataFrame(result)
    df.to_csv('report2.csv', index=False)
    print(df)