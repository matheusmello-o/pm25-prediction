import pandas as pd 
import numpy as np
import os


CSV_PATH = '../data/raw/PRSA_data_2010.1.1-2014.12.31.csv'
TARGET = 'pm2_5'

DIR_PROCESSED = '../data/processed'



# %%
def load_and_clean() -> pd.DataFrame:

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError (
            f'This file or path does not exits.'
        )
    
    df_local = pd.read_csv(CSV_PATH)
    n_raw = df_local.shape[0]

    # removing null lines
    df_local = df_local.dropna().reset_index(drop = True)

    # low column's name and renaming target
    df_local.columns = [x.lower() for x in df_local.columns]
    df_local.rename(columns = {'pm2.5': TARGET})

    print('Total lines raw: ', n_raw)
    print('Total lines clean: ', df_local.shape[0])

    return df_local




def data_prep(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_local = df_raw.drop(columns = 'No').copy()	

    df_local.rename(columns = {'pm2.5': 'pm2_5'}, inplace = True)

    cols = [x.lower() for x in df_local.columns]
    df_local.columns = cols

    # astype str in the following features
    df_local['year_str'] = df_local.year.astype(str)
    df_local['month_str'] = df_local.month.astype(str)
    df_local['day_str'] = df_local.day.astype(str)
    df_local['hour_str'] = df_local.hour.astype(str)


    df_local['month_str'] = df_local.month_str.str.pad(side = 'left', width = 2, fillchar = '0')
    df_local['day_str'] = df_local.day_str.str.pad(side = 'left', width = 2, fillchar = '0')

    df_local['date'] = df_local.year_str + '-' + df_local.month_str + '-' + df_local.day_str
    df_local['period'] = df_local.year_str + df_local.month_str 

    return df_local.copy()


def main():

    df = load_and_clean()

    return print(df.shape)


if __name__ == '__main__':
    main()