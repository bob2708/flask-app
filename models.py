import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from elm import Extreme

def timeseries_train_test_split(X, y, test_size):
    
    # Test part start index
    test_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test

def feature_extraction(df, col=0):
    
    data = pd.DataFrame(df.copy())
    data = pd.DataFrame(data.iloc[:, col])
    data.columns = ["y"]

    # Normalizing
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / data_std

    # Add lags 1-24
    for i in range(1, 25):
        data["lag_{}".format(i)] = data.y.shift(i)

    return data, data_mean, data_std
    
def training_models(df, col=0):
    
    data, data_mean, data_std = feature_extraction(df, col=col)

    y = data.dropna().y
    X = data.dropna().drop(["y"], axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    # Models init
    lr = LassoCV()
    xgb = XGBRegressor()
    knn = KNeighborsRegressor(20)
    rf = RandomForestRegressor()
    elm = Extreme()
    ens = LinearRegression()

    # LSTM model
    lstm = Sequential()
    lstm.add(LSTM(50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mse')

    lr.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    elm.fit(X_train, y_train)
    lstm.fit(X_train_reshaped, y_train.values, epochs=100, verbose=0)

    # Ensemble model (stacking with LinearRegression)
    ensemble = pd.DataFrame(y_test)

    for model in [lr, xgb, knn, rf, elm]:
        ensemble[str(model).split('(')[0]]=model.predict(X_test)

    X_ens = ensemble.dropna().drop(["y"], axis=1)
    y_ens = ensemble.dropna().y
    ens.fit(X_ens, y_ens)

    return ([ens, lr, xgb, knn, rf, elm, lstm], (X_train, X_test, y_train, y_test), (data_mean, data_std))

def train_lr_mult(df, target_col):
    y = df.dropna()[df.columns[target_col]]
    X = df.dropna().drop([df.columns[target_col]], axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
    lr = LassoCV()
    lr.fit(X_train, y_train)
    
    return lr, [X_train, X_test, y_train, y_test]


# Calculating predictions
def calc_predicts(data, model, steps):
    main = pd.DataFrame(data.copy())
    idxs=[]
    pylast = main.index[-1].to_pydatetime()
    
    for i in range(steps):
        idxs.append(pylast + datetime.timedelta(days=1+i))
    new = pd.DataFrame(index=idxs, columns=["lag_{}".format(i) for i in range(1, 25)]).astype('float64')
    full = pd.concat([main, new])
    j = steps

    while(j>0):
        for i in range(4):
            if j == 0:
                pdctd = model.predict(full.iloc[-i:].drop(['y'], axis=1))
                full['y'][-i:]=pdctd.reshape((pdctd.shape[0]))
                return full
            first = full.iloc[-j-1].shift(1)
            first[1] = full['y'].dropna()[i-4]
            full.iloc[-j] = first
            j = j - 1
        if j == 0:
            pdctd = model.predict(full.iloc[-j-4:].drop(['y'], axis=1))
            full['y'][-j-4:]=pdctd.reshape((pdctd.shape[0]))
        else:
            pdctd = model.predict(full.iloc[-j-4:-j].drop(['y'], axis=1))
            full['y'][-j-4:-j]=pdctd.reshape((pdctd.shape[0]))
    return full