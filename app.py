from flask import Flask, render_template, redirect, url_for, request
from covid import update_covid_data
from plots import plotMovingAverage, plotModelResults, plotCoefficients, plot_ml_predictions, basic_plot, mean_absolute_percentage_error
from misc import handle_missing_values
from models import training_models, calc_predicts, feature_extraction, train_lr_mult
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

prediction_steps = 30
target_col = 0
cur_col = 0
plot_anomalies = False
plot_intervals = False
df = update_covid_data()
df = handle_missing_values(df)
models, data, mean_std = training_models(df)
app = Flask(__name__)

def load_from_file(file):
	global df, models, mult, mult_data, data, target_col, mean_std
	df = pd.read_csv(file, index_col=0, parse_dates=True)
	df = handle_missing_values(df)
	models, data, mean_std = training_models(df, target_col)
	basic_plot(df, target_col)
	if df.shape[1] > 1:
		mult, mult_data = train_lr_mult(df, target_col)

@app.route('/')
def index():
	return redirect(url_for('home'))

@app.route('/view', methods=['GET', 'POST'])
def view():
	global target_col, models, data
	window_size = 14
	max_number = ''
	if len(df.columns) > 1:
		max_number = str(len(df.columns))
		if request.method == 'POST':
			target_col = int(request.form['column']) - 1
			basic_plot(df, target_col)
			window_size = int(request.form['window_size'])
	else:
		if request.method == 'POST':
			window_size = int(request.form['window_size'])

	fig = seasonal_decompose(df.iloc[:, target_col]).plot(resid=False)
	fig.set_figwidth(10)
	fig.set_figheight(10)
	fig.get_axes()[0].set_title('Seasonal/trend decompose')
	plt.savefig('static/decompose.png')

	plotMovingAverage(pd.DataFrame(df.iloc[:, target_col]), window_size, plot_intervals=True, plot_anomalies=True)
	return render_template(
		'view.html', basic_img='static/basic.png', moving_avg_img='static/moving_avg.png', decompose_img = 'static/decompose.png',
		cur_target=target_col+1, max_col=max_number, max_row=(df.shape[0]/5), window_size=window_size
	)

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/load', methods=['GET', 'POST'])
def load():
	if request.method == 'POST':
		f = request.files['time_series']
		f.save('static/time_series.csv')
		load_from_file('static/time_series.csv')
		return redirect(url_for('view'))
	return render_template('load.html')

@app.route('/models', methods=['GET', 'POST'])
def model():
	global models, mult, data, mult_data, plot_anomalies, plot_intervals, cur_col, mean_std
	
	if request.method == 'POST':
		plot_intervals = request.form.get('intervals')
		plot_anomalies = request.form.get('anomalies')
	
	tscv = TimeSeriesSplit(n_splits=5)
	errors = []
	ens_train_df = pd.DataFrame()
	ens_test_df = pd.DataFrame()
	
	# Update column if changed
	if target_col != cur_col:
		cur_col = target_col
		models, data, mean_std = training_models(df, target_col)
		if df.shape[1] > 1:
			mult, mult_data = train_lr_mult(df, target_col)
	
	# Bulid all models (except multivariate and emsemble)
	idx = 0
	for model in models:
		if 'Linear' in str(model):
			ens_idx = idx
			continue
		error, prediction = plotModelResults(
			model, 
			X_train=data[0], X_test=data[1], 
			y_train=data[2], y_test=data[3],
			mean_std = mean_std,
			plot_anomalies=plot_anomalies,
			plot_intervals=plot_intervals,
			tscv=tscv)
		errors.append(error)
		ens_train_df[str(model).split('(')[0]] = prediction[1]
		ens_test_df[str(model).split('(')[0]] = prediction[0]
		idx += 1
	
	# Bulid multivariate model if possible
	if df.shape[1] > 1:
		error, prediction = plotModelResults(
			mult, 
			X_train=mult_data[0], X_test=mult_data[1], 
			y_train=mult_data[2], y_test=mult_data[3],
			mean_std = ([0], [1]),
			plot_anomalies=plot_anomalies,
			plot_intervals=plot_intervals,
			tscv=tscv, name='Multivariate model'
		)

	# Bulid ensemble model
	errors.insert(0, 
		plotModelResults(
			models[ens_idx], 
			X_train=ens_train_df.iloc[:, :-1], X_test=ens_test_df.iloc[:, :-1], 
			y_train=data[2], y_test=data[3], 
			mean_std = mean_std, tscv=tscv
			)[0]
		)

	# Choose the best model
	best_model = models[errors.index(min(errors[:-1]))]
	best_model_idx = models.index(best_model)
	if best_model_idx != 0:
		models[best_model_idx], models[0] = models[0], models[best_model_idx]
	
	return render_template(
		'models.html',
		best_model='static/{0:}_res.png'.format(str(models[0]).split('(', 1)[0]),
		models=['static/{0:}_res.png'.format(str(model).split('(', 1)[0]) for model in models[1:]],
		mult_model='static/Multivariate model_res.png' if df.shape[1]>1 else '',
		checked=[plot_anomalies, plot_intervals]
	)

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
	global prediction_steps, cur_col, target_col, models, mult, mult_data, data, mean_std
	if request.method == 'POST':
		prediction_steps = int(request.form['steps'])

	# Update column if changed
	if target_col != cur_col:
		cur_col = target_col
		models, data, mean_std = training_models(df, target_col)
		if df.shape[1] > 1:
			mult, mult_data = train_lr_mult(df, target_col)

	# Calculate predictions (except multivariate and ensemble)
	col_data, data_mean, data_std = feature_extraction(df, target_col)
	predictions = pd.DataFrame()
	idx, lasso_idx = 0, 0
	for model in models:
		if 'Linear' in str(model):
			ens_idx = idx
			idx += 1
			continue
		elif 'Lasso' in str(model):
			lasso_idx = idx

		full = calc_predicts(col_data, model, prediction_steps)
		prediction = full['y'][-prediction_steps:]*data_std[0]+data_mean[0]
		predictions[str(model).split('(')[0]] = prediction
		df_predicted = pd.DataFrame(df.iloc[:, target_col].append(prediction))
		plot_ml_predictions(df_predicted, model, prediction_steps)
		idx += 1
	
	# Ensemble prediction
	prediction = models[ens_idx].predict(predictions.iloc[:, :-1])
	df_predicted = pd.DataFrame(df.iloc[:, target_col].append(pd.Series(prediction, index=predictions.index)))

	plot_ml_predictions(df_predicted, models[ens_idx], prediction_steps)
	
	# Collect predictions on all columns
	all_predictions = pd.DataFrame()
	for col in range(df.shape[1]):
		col_data, data_mean, data_std = feature_extraction(df, col)
		print(models[lasso_idx])
		print(models)
		full = calc_predicts(col_data, models[lasso_idx], prediction_steps)
		prediction = full['y'][-prediction_steps:]*data_std[0]+data_mean[0]
		all_predictions[df.columns[col]] = prediction

	# Multivariate prediction if possible
	if df.shape[1] > 1:
		all_predictions = all_predictions.drop([all_predictions.columns[target_col]], axis=1)
		mult_pred = mult.predict(all_predictions)
		df_predicted = pd.DataFrame(df.iloc[:, target_col].append(pd.Series(mult_pred, index=predictions.index)))
		plot_ml_predictions(df_predicted, 'Multivariate model()', prediction_steps)

	return render_template(
		'predictions.html',
		best_model='static/{0:}_pred.png'.format(str(models[0]).split('(', 1)[0]),
		models=['static/{0:}_pred.png'.format(str(model).split('(', 1)[0]) for model in models[1:-1]],
		mult_model='static/Multivariate model_pred.png' if df.shape[1]>1 else '',
		steps=prediction_steps,
		max_steps=(df.shape[0]/3)
	)

if __name__ == "__main__":
	app.run('127.0.0.1', 8800, True)