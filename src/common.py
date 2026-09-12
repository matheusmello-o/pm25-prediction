import pandas as pd
import numpy as np


def data_split_train_test(df_preprocessed: pd.DataFrame, proportion_test: float) -> pd.DataFrame:
    df_local = df_preprocessed.copy()

    df_len = df_local.shape[0]
    index_70_per = int(df_len * (1 - proportion_test))

    # split train and test - method = out-of-time
    train = df_local.iloc[:index_70_per].reset_index(drop = True)
    test = df_local.iloc[index_70_per:].reset_index(drop = True)

    return train, test