#!/usr/bin/python3
# -*- coding:utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

if __name__ == '__main__':
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print(np.mean(y_pred == y_test))
    print(knn.score(X_test, y_test))
