import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

enecoder = preprocessing.LabelEncoder()

root = 'data'
def read_data(root, filename):
    csv_path = os.path.join(root, filename)
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 0:-1].drop('Id', axis=1)
    y = df.iloc[:, -1]
    return X, y

def plot(X, y, centroids):
    COLORS = {'Iris-setosa':'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}
    fig, ax = plt.subplots()
    for i in range(len(X['SepalLengthCm'])):
        ax.scatter(X['SepalLengthCm'][i], X['SepalWidthCm'][i], color=COLORS[y[i]]) 
    ax.set_title('Iris dataset')
    ax.set_xlabel('SepalLengthCm')  
    ax.set_ylabel('SepalWidthCm')

    _color = ''
    _marker = ''
    for centroid in centroids:
        if centroid == 0:
            _color = 'g'
            _marker = '.'
        elif centroid == 0:
            _color = 'r'
            _marker = 'o'
        else:
            _color = 'b'
            _marker = 'v'
        plt.scatter(centroids[centroid][0], centroids[centroid][1], marker=_marker, color=_color)
    plt.show()
    

def fit(X, k, max_iters=300, tol=1e-4):
    centroids = {}
    classes = {}

    # init centroids
    for i in range(k):
        centroids[i] = X[i]
    for i in range(max_iters):
        for i in range(k):
            classes[i] = []
        for x in X:
            distances = [np.linalg.norm(x - centroids[centroid]) for centroid in centroids]
            nearest = np.argmin(distances)
            classes[nearest].append(x)

        previous = dict(centroids)
        for c in classes:
            centroids[c] = np.average(classes[c], axis=0)

        isOptimal = False
        for centroid in centroids:
            original_centroid = previous[centroid]
            current_centroid = centroids[centroid]
            if sum((current_centroid - original_centroid) / original_centroid) * 100 > tol:
                isOptimal = True
        if isOptimal:
            break
    return centroids

def predict(centroids, X_test):
    nearests = []
    for x in X_test:   
        distances= [np.linalg.norm(x - centroids[centroid]) for centroid in centroids]
        nearest = np.argmin(distances)
        nearests.append(nearest)
    # labels = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
    return nearests

X, y = read_data(root, 'iris.csv')
y1 = y
y = enecoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
centroids = fit(X_train.values, 3)
plot(X, y, centroids)
print(centroids)
# predict = predict(centroids, X_test.values)
# print(y.shape)
# print(np.sum(np.where(predict == y_test)))

from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train.values)
print(kmeans.cluster_centers_)
# predict = kmeans.predict(X_test)
# print(np.sum(np.where(predict == y_test)))