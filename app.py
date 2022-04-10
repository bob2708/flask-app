from flask import Flask, render_template, redirect, url_for, request
from covid import update_covid_data
from plots import plotMovingAverage, plotModelResults, plotCoefficients, plot_ml_predictions, basic_plot
from misc import handle_missing_values
from models import training_models, calc_predicts, feature_extraction
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit

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
	global df, models, data, target_col, mean_std
	df = pd.read_csv(file, index_col=0, parse_dates=True)
	df = handle_missing_values(df)
	models, data, mean_std = training_models(df, target_col)
	basic_plot(df, target_col)

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
			#models, data = training_models(df, target_col)
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
	return render_template('base.html')

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
	global models, data, plot_anomalies, plot_intervals, cur_col, mean_std
	if request.method == 'POST':
		plot_intervals = request.form.get('intervals')
		plot_anomalies = request.form.get('anomalies')
	tscv = TimeSeriesSplit(n_splits=5)
	errors = []
	if target_col != cur_col:
		cur_col = target_col
		models, data, mean_std = training_models(df, target_col)
	for model in models:
		errors.append(plotModelResults(
			model, 
			X_train=data[0], X_test=data[1], 
			y_train=data[2], y_test=data[3],
			mean_std = mean_std,
			plot_anomalies=plot_anomalies,
			plot_intervals=plot_intervals,
			tscv=tscv)
		)
		# plotCoefficients(model, X_train=data[0])

	best_model = models[errors.index(min(errors[:-1]))]
	
	best_model_idx = models.index(best_model)
	if best_model_idx != 0:
		models[best_model_idx], models[0] = models[0], models[best_model_idx]
	
	print(mean_std)

	return render_template(
		'models.html',
		best_model='static/{0:}_res.png'.format(str(models[0]).split('(', 1)[0]),
		models=['static/{0:}_res.png'.format(str(model).split('(', 1)[0]) for model in models[1:]],
		checked=[plot_anomalies, plot_intervals]
	)

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
	global prediction_steps
	if request.method == 'POST':
		prediction_steps = int(request.form['steps'])
	data, data_mean, data_std = feature_extraction(df, target_col)
	for model in models:
		full = calc_predicts(data, model, prediction_steps)
		df_predicted = pd.DataFrame(df.iloc[:, target_col].append(full['y'][-prediction_steps:]*data_std[0]+data_mean[0]))
		print(df_predicted)
		plot_ml_predictions(df_predicted, model, prediction_steps)

	return render_template(
		'predictions.html',
		best_model='static/{0:}_pred.png'.format(str(models[0]).split('(', 1)[0]),
		models=['static/{0:}_pred.png'.format(str(model).split('(', 1)[0]) for model in models[1:-1]],
		steps=prediction_steps,
		max_steps=(df.shape[0]/3)
	)

if __name__ == "__main__":
	app.run('127.0.0.1', 8800, True)