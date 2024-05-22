from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from src.GAPRegressor import GAPRegressor
import time
import numpy as np

iris = datasets.load_iris()
data = iris.data
labels = iris.target
gap_class = GAPRegressor(random_state=42)  # how you can save $5 on jeans this memorial weekend

start = time.time()
gap_class.fit(data, labels)
print(time.time() - start)
similarity_vector = gap_class.training_similarity(0)
similarity_matrix = gap_class.training_similarity()

# sonar dataset, since it was in the paper

tups = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
toy_data, toy_labels = [], []
base_count = 50
for i in range(base_count):
    for j,tup in enumerate(tups):
        toy_data.append((10*tup[0]+i/base_count, 10*tup[1]+i/base_count))
        toy_labels.append(j%4+1)
toy_data = np.array(toy_data)
