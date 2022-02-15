from flask import Flask, render_template, redirect, url_for, request
from covid import update_covid_data
from plots import plotMovingAverage, plotModelResults, plotCoefficients
from misc import handle_missing_values
from models import training_models

from sklearn.model_selection import TimeSeriesSplit

import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = update_covid_data()
df = handle_missing_values(df)
app = Flask(__name__)

def load_from_file(file):
	global df
	df = pd.read_csv(file, index_col=0, parse_dates=True)
	df = handle_missing_values(df)
	plt.figure(figsize=(11, 5))
	plt.plot(df)
	plt.grid(True)
	plt.title("Moscow daily COVID-19 cases")
	plt.savefig('static/covid.png')

@app.route('/')
def index():
	return redirect(url_for('home'))

@app.route('/view')
def view():
	plotMovingAverage(df, 14, plot_intervals=True, plot_anomalies=True)
	return render_template('view.html', basic_img='static/covid.png', moving_avg_img='static/moving_avg.png')

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
	models, data = training_models(df)
	tscv = TimeSeriesSplit(n_splits=5)
	errors = []
	for model in models:
		errors.append(plotModelResults(
			model, 
			X_train=data[0], X_test=data[1], 
			y_train=data[2], y_test=data[3],
			plot_anomalies=False,
			plot_intervals=True,
			tscv=tscv)
		)
		# plotCoefficients(model, X_train=data[0])
	best_model = models[errors.index(min(errors))]
	best_model_name = str(best_model).split('(', 1)[0]
	
	models.remove(best_model)
	
	return render_template(
		'models.html',
		best_model='static/{0:}_res.png'.format(best_model_name),
		models=['static/{0:}_res.png'.format(str(model).split('(', 1)[0]) for model in models]
	)
	# steps = 90
	# full = calcPredicts(model, steps)
	# predictions = pd.DataFrame(full['y'][-steps:])
	# predictions.columns = ["Russia_cases"]
	# df_lr_predicted = df.append(predictions)

@app.route('/metrics', methods=['GET', 'POST'])
def metric():
	return '123'

if __name__ == "__main__":
	app.run('127.0.0.1', 8800, True)