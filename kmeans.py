import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

root = 'data'
def read_data(root, filename):
	csv_path = os.path.join(root, filename)
	df = pd.read_csv(csv_path)
	X = df.iloc[:, 0:-1].drop('Id', axis=1)
	y = df.iloc[:, -1]
	return X, y

def plot(X, y):
	COLORS = {'Iris-setosa':'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}
	fig, ax = plt.subplots()
	for i in range(len(X['SepalLengthCm'])):
		ax.scatter(X['SepalLengthCm'][i], X['SepalWidthCm'][i], color=COLORS[y[i]])	
	ax.set_title('Iris dataset')
	ax.set_xlabel('SepalLengthCm')	
	ax.set_ylabel('SepalWidthCm')
	plt.show()

X, y = read_data(root, 'iris.csv')
# fit(X.values, 3)