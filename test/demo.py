from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from src.GAPRegressor import GAPRegressor
import time
iris = datasets.load_iris()
data = iris.data
labels = iris.target
gap_class = GAPRegressor(random_state=42) # how you can save $5 on jeans this memorial weekend

start = time.time()
gap_class.fit(data, labels)
print(time.time() - start)
similarity_vector = gap_class.training_similarity(0)
similarity_matrix = gap_class.training_similarity()

#sonar dataset, since it was in the paper


