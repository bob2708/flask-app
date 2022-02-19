import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def update_covid_data():
	# Загрузка глобальных (по странам) и локальных (по городам) данных
	global_data = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", parse_dates=True)
	daily_data = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/12-13-2021.csv")
	global_data_russia = global_data[global_data['Country/Region']=="Russia"]
	processed_dates = np.array([datetime.datetime.strptime(i, "%m/%d/%y") for i in global_data_russia.iloc[:, 4:].keys()])
	total_dates = len(processed_dates)
	
	moscow_dates_having_data = list()
	moscow_daily_data = list()
	with open("static/moscow_daily_cases.txt", 'r') as f:
		test = f.readline()
		moscow_daily_data = list(map(int, test.split(' ')))
		test = f.readline()
		moscow_dates_having_data = [datetime.datetime.strptime(x, "%m-%d-%Y") for x in test.split(' ')]

	if moscow_dates_having_data[-1] < processed_dates[-1]:
		print('Updating COVID data...', end='')
		dates_to_add = list()
		cases_to_add = list()
		for date in processed_dates[np.where(processed_dates == moscow_dates_having_data[-1])[0][0]+1:]:
			search_date = f"{str(date.month).zfill(2)}-{str(date.day).zfill(2)}-{date.year}"
			url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/"+search_date+".csv"
			daily_data = pd.read_csv(url)
			if 'Province_State' in daily_data.keys():
				daily_cases = daily_data[daily_data['Province_State']=="Moscow"]["Confirmed"].values
				if len(daily_cases) == 1:
					cases_to_add.append(daily_cases[0]) 
					dates_to_add.append(date)
		moscow_daily_data+=cases_to_add
		moscow_dates_having_data+=dates_to_add
		with open("static/moscow_daily_cases.txt", 'w') as f:
			f.write(" ".join(str(x) for x in moscow_daily_data))
			f.write('\n')
			f.write(" ".join(x.date().strftime("%m-%d-%Y") for x in moscow_dates_having_data))
		print('done!')

	df = pd.read_csv('static/moscow_daily_cases.txt', sep=' ').T
	df = pd.DataFrame(df.index.values, index=[datetime.datetime.strptime(idx[0], "%m-%d-%Y") for idx in df.values], columns=['Daily Cases']).astype(float)

	daily_change = list()
	for i in range(1, len(df)):
		daily_change.append(df.iloc[i]-df.iloc[i-1])

	plt.figure(figsize=(11, 5))
	plt.plot(df.index[1:], daily_change)
	plt.grid(True)
	plt.title("Moscow daily cases")
	plt.savefig('static/basic.png')
	return pd.DataFrame(daily_change, index=df.index[1:])