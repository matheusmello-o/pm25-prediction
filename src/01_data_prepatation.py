import pandas as pd 
import numpy as np
import os
import json

CSV_PATH = 'data/raw/PRSA_data_2010.1.1-2014.12.31.csv'
TARGET = 'pm2_5'

DIR_PROCESSED = 'data/processed'
DIR_DF_EVALUATION = 'data/validation'
PROPORTION_VALIDATION_DATASET = 0.3
TRANSFORM_DATE_COLS = False # transform year, month, day to string and create a columns date = year_str + month_str + day_str

def load_and_clean() -> pd.DataFrame:

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError (
            f'This file or path does not exits.'
        )
    
    df_local = pd.read_csv(CSV_PATH)

    n_raw = df_local.shape[0]


    # removing null lines
    df_local = df_local.dropna().reset_index(drop = True)

    # removing No column (No = number of rows)
    df_local.drop(columns = 'No', inplace = True)

    # low column's name and renaming target
    df_local.columns = [x.lower() for x in df_local.columns]
    df_local.rename(columns = {'pm2.5': TARGET}, inplace = True)

    print('Total lines raw: ', n_raw)
    print('Total lines clean: ', df_local.shape[0])

    return df_local



def date_transform_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_local = dataframe.copy()

    # astype str in the following features
    df_local['year_str'] = df_local.year.astype(str)
    df_local['month_str'] = df_local.month.astype(str)
    df_local['day_str'] = df_local.day.astype(str)
    df_local['hour_str'] = df_local.hour.astype(str)


    df_local['month_str'] = df_local.month_str.str.pad(side = 'left', width = 2, fillchar = '0')
    df_local['day_str'] = df_local.day_str.str.pad(side = 'left', width = 2, fillchar = '0')

    df_local['date'] = df_local.year_str + '-' + df_local.month_str + '-' + df_local.day_str
    df_local['period'] = df_local.year_str + df_local.month_str 

    return df_local


def transform_sen_cos(x,  period: int, cos_operation = True) -> pd.DataFrame:
    if cos_operation:
        cos = np.cos(2 * np.pi * x / period)
        return cos
    else:
        sin = np.sin(2 * np.pi * x / period)
        return sin
    

def feature_engineering(df_clean: pd.DataFrame) -> pd.DataFrame:
    df_local = df_clean.copy()	

    if TRANSFORM_DATE_COLS:
        df_local = date_transform_columns(dataframe=df_local)

    # Transform Month and Hour to sen/cos to maintain cyclic characteristics
    df_local['month_cos'] = df_local.month.apply(transform_sen_cos, period = 12)
    df_local['month_sin'] = df_local.month.apply(transform_sen_cos, period = 12, cos_operation = False)

    df_local['hour_cos'] = df_local.hour.apply(transform_sen_cos, period = 24)
    df_local['hour_sin'] = df_local.hour.apply(transform_sen_cos, period = 24, cos_operation = False)

    cbwd = pd.get_dummies(data = df_local.cbwd, prefix = 'cbwd').astype(int)

    binary_cols = list(cbwd.columns)

    continuos_columns = [
        'year', 'dewp', 'temp', 'pres', 'iws', 'is', 'ir',
        'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
    ]

    df_final = pd.concat([df_local[continuos_columns], cbwd, df_local[[TARGET]]], axis = 1)

    return df_final, continuos_columns, binary_cols




def data_split_train_test(df_preprocessed: pd.DataFrame) -> pd.DataFrame:
    df_local = df_preprocessed.copy()

    df_len = df_local.shape[0]
    index_70_per = int(df_len * (1 - PROPORTION_VALIDATION_DATASET))

    # split train and test - method = out-of-time
    train = df_local.iloc[:index_70_per].reset_index(drop = True)
    test = df_local.iloc[index_70_per:].reset_index(drop = True)

    return train, test






def main():

    # 1. Load and Clean dataset
    df_clean = load_and_clean()


    # 2. Preprocessing dataset
    df_preprocessed, continuous_features, binary_features = feature_engineering(df_clean = df_clean)
    print('Step 2: Preprocessing - OK')

    # 3. Train and Test Separation - method out-of-time
    train, test = data_split_train_test(df_preprocessed = df_preprocessed) 
    
    # 4. Saving datasets
    test.to_csv(os.path.join(DIR_DF_EVALUATION, 'validation_dataset.csv'), index = False)
    train.to_csv(os.path.join(DIR_PROCESSED, 'processed_dataset.csv'), index = False)
    print('Step 3: Data Separated and saved - OK')

    # Saving metadata
    with open(os.path.join(DIR_PROCESSED, 'metadata.json'), 'w') as f:
        metadata = {
            'continuos_features': continuous_features,
            'binary_features': binary_features,
            'target_column': TARGET,
            'test_proportion': PROPORTION_VALIDATION_DATASET
        }

        json.dump(metadata, f, indent = 1)
    print('Step 4: Metadata saved - OK')
    
if __name__ == '__main__':
    main()