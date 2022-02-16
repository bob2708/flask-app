import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression

from xgboost import XGBRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """

    # индекс с которого начинается тестовая часть
    test_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test

def feature_extraction(df):
    data = pd.DataFrame(df.copy())
    data.columns = ["y"]
    data = data.astype('int64')

    # Добавления значений лагов 4-24
    for i in range(4, 25):
        data["lag_{}".format(i)] = data.y.shift(i)

    return data
    

def training_models(df):
    # выделение 30% данных для теста
    data = feature_extraction(df)

    y = data.dropna().y
    X = data.dropna().drop(["y"], axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
    
    # обучение моделей
    lr = LinearRegression()
    xgb = XGBRegressor()
    dt = tree.DecisionTreeRegressor()
    rf = RandomForestRegressor()

    lr.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return ([lr, xgb, dt, rf], (X_train, X_test, y_train, y_test))

# Вычисление предсказаний
def calc_predicts(data, model, steps):
    main = pd.DataFrame(data.copy())
    idxs=[]
    pylast = main.index[-1].to_pydatetime()
    
    for i in range(steps):
        idxs.append(pylast + datetime.timedelta(days=1+i))
    new = pd.DataFrame(index=idxs, columns=["lag_{}".format(i) for i in range(4, 25)]).astype('float64')
    full = pd.concat([main, new])
    j = steps

    while(j>0):
        for i in range(4):
            if j == 0:
                pdctd = model.predict(full.iloc[-i:].drop(['y'], axis=1))
                full['y'][-i:]=pdctd
                return full
            first = full.iloc[-j-1].shift(1)
            first[1] = full['y'].dropna()[i-4]
            full.iloc[-j] = first
            j = j - 1
        if j == 0:
            pdctd = model.predict(full.iloc[-j-4:].drop(['y'], axis=1))
            full['y'][-j-4:]=pdctd
        else:
            pdctd = model.predict(full.iloc[-j-4:-j].drop(['y'], axis=1))
            full['y'][-j-4:-j]=pdctd
    return full