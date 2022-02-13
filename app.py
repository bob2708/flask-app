from flask import Flask, render_template
import datetime
import matplotlib.pyplot as plt
import pandas as pd

def build_model(data):
	pass

app = Flask(__name__)

# df = pd.DataFrame([1, 2, 3, 4], columns=['test'])
df = pd.read_csv('C:\\Users\\Vova\\Desktop\\moscow_daily_cases.txt', sep=' ').T
df = pd.DataFrame(df.index.values, index=[datetime.datetime.strptime(idx[0], "%m-%d-%Y") for idx in df.values], columns=['Daily Cases']).astype(float)

daily_change = list()
for i in range(1, len(df)):
    daily_change.append(df.iloc[i]-df.iloc[i-1])

plt.figure(figsize=(11, 7))
plt.plot(df.index[1:], daily_change)
plt.grid(True)
plt.title("Moscow daily cases")
plt.savefig('static/test.png')

@app.route('/')
def hello_world():
	return render_template('index.html', test_img='static/test.png')

if __name__ == "__main__":
	app.run('127.0.0.1', 8800, True)