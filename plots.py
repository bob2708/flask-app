import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def basic_plot(df, target_col):
    plt.figure(figsize=(11, 5))
    plt.plot(pd.DataFrame(df.iloc[:, target_col]))
    plt.grid(True)
    plt.title(df.columns[target_col])
    plt.savefig('static/basic.png')
    
def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(11,5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig('static/moving_avg.png')

def plotModelResults(
    model, X_train, X_test, y_train, y_test, mean_std,
    tscv, plot_intervals=False, plot_anomalies=False, name=''
):
    """
        Plots modelled vs fact values, prediction intervals and anomalies
    """

    prediction_unscaled = model.predict(X_test)
    prediction2_unscaled = model.predict(X_train)
    
    prediction = prediction_unscaled * mean_std[1][0] + mean_std[0][0]
    prediction2 = prediction2_unscaled * mean_std[1][0] + mean_std[0][0]

    prediction = prediction.reshape((prediction_unscaled.shape[0]))
    prediction2 = prediction2.reshape((prediction2_unscaled.shape[0]))

    X_train = X_train * mean_std[1][0] + mean_std[0][0]
    y_test = y_test * mean_std[1][0] + mean_std[0][0]
    y_train = y_train * mean_std[1][0] + mean_std[0][0]

    plt.figure(figsize=(11, 5))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)

    if plot_intervals and ('keras' not in str(model)) and ('Extreme' not in str(model)):
        cv = cross_val_score(
            model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error"
        )
        mae = cv.mean() * (-1)
        deviation = cv.std()

        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)

        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(y_test))
            anomalies[y_test < lower] = y_test[y_test < lower]
            anomalies[y_test > upper] = y_test[y_test > upper]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

    error = mean_absolute_percentage_error(prediction, y_test)
    error2 = mean_absolute_percentage_error(prediction2, y_train)
    rmse = mean_squared_error(prediction, y_test, squared=False)
    rmse2 = mean_squared_error(prediction2, y_train, squared=False)

    if name:
        plt.title(
            """
            Mean absolute percentage error {1:.2f}%/{0:.2f}%\n 
            Routed mean squared error {2:.2f}/{3:.2f}\n 
            for {4:} (train/test)
            """.format(error, error2, rmse2, rmse, name)
            )
    elif 'Linear' in str(model):
        plt.title(
            """
            Mean absolute percentage error {1:.2f}%/{0:.2f}%\n 
            Routed mean squared error {2:.2f}/{3:.2f}\n 
            for {4:} (train/test)
            """.format(error, error2, rmse2, rmse, 'Models ensemble')
            )
    elif 'sequential' not in str(model):
        plt.title(
            """
            Mean absolute percentage error {1:.2f}%/{0:.2f}%\n 
            Routed mean squared error {2:.2f}/{3:.2f}\n 
            for {4:} (train/test)
            """.format(error, error2, rmse2, rmse, str(model).split('(', 1)[0])
            )
    else:
        plt.title(
            """
            Mean absolute percentage error {1:.2f}%/{0:.2f}%\n
            Routed mean squared error {2:.2f}/{3:.2f} (train/test)
            """.format(error, error2, rmse2, rmse)
            )

    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)

    if name:
        plt.savefig(f'static/{name}_res.png')
    elif 'sequential' in str(model):
        plt.savefig('static/lstm_res.png')
    else:
        plt.savefig('static/{0:}_res.png'.format(str(model).split('(', 1)[0]))
    return (error, error2), (prediction_unscaled, prediction2_unscaled)

def plotCoefficients(model, X_train):
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    plt.figure(figsize=(11, 5))
    coefs.coef.plot(kind="bar")
    plt.grid(True, axis="y")
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles="dashed")
    plt.savefig('static/{0:}_coefs.png'.format(str(model).split('(', 1)[0]))

def plot_ml_predictions(data, model, steps):
    plt.figure(figsize=(11, 5))
    plt.grid(True)
    if 'Linear' in str(model):
        plt.title('Models ensemble')
    elif 'sequential' not in str(model): 
        plt.title(str(model).split('(', 1)[0])
    plt.plot(data)
    plt.axvspan(data.index[-steps], data.index[-1], alpha=0.5, color='silver')
    if 'sequential' in str(model):
        plt.savefig('static/lstm_pred.png')
    else:
        plt.savefig('static/{0:}_pred.png'.format(str(model).split('(', 1)[0]))
