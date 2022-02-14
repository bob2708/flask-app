from flask import Flask, render_template, redirect, url_for, request
from covid import update_covid_data
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame()
app = Flask(__name__)

update_covid_data()

def load_from_file(file):
	global df
	df = pd.read_csv(file, index_col=0, parse_dates=True)
	plt.figure(figsize=(11, 7))
	plt.plot(df)
	plt.grid(True)
	plt.title("Moscow daily COVID-19 cases")
	plt.savefig('static/covid.png')

@app.route('/')
def index():
	return redirect(url_for('home'))

@app.route('/view')
def view():
	return render_template('view.html', test_img='static/covid.png')

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

if __name__ == "__main__":
	app.run('127.0.0.1', 8800, True)