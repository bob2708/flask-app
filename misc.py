import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import timedelta

def handle_missing_values(df):
    df_rn = df.dropna()
    min_delta = df_rn.index[1]-df_rn.index[0]

    for i in range(2, len(df_rn.index)):
        if (df_rn.index[i-1]-df_rn.index[i-2])>(df_rn.index[i]-df_rn.index[i-1]):
            min_delta = df_rn.index[i]-df_rn.index[i-1]

    if not False in [int((df_rn.index[i]-df_rn.index[i-1])/min_delta)==(df_rn.index[i]-df_rn.index[i-1])/min_delta for i in range(1, len(df_rn.index))]:
        new_idxs = pd.date_range(start=df_rn.index[0], end=df_rn.index[-1], freq=min_delta)
    df_rn = pd.DataFrame(
        pd.merge(df_rn, 
                 pd.DataFrame(index=new_idxs, columns=df_rn.columns), 
                 how='right', 
                 left_index=True, 
                 right_index=True
                ).iloc[:, :len(df_rn.columns)]
    )
    return df_rn.interpolate(method='time', axis=0).ffill().bfill()
