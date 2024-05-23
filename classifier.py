from src.GAPClassifier import GAPClassifier
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import time
import numpy as np

iris = datasets.load_iris()
data = iris.data
labels = iris.target
model = GAPClassifier().fit(data, labels)
