from PY_RF_GAP.GAPClassifier import GAPClassifier
from sklearn import datasets


iris = datasets.load_iris()
data = iris.data
labels = iris.target
model = GAPClassifier().fit(data, labels)

similarity = model.training_similarity()
print("Similarity: ", similarity)