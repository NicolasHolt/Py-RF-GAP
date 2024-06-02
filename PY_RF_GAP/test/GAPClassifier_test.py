from PY_RF_GAP.GAPClassifier import GAPClassifier
from sklearn import datasets
import numpy as np

def test_GAPClassifier_instance():
    classifier = GAPClassifier()
    assert isinstance(classifier, GAPClassifier), "Object is not an instance of GAPClassifier"

def test_GAPClassifier_fit():
    classifier = GAPClassifier()
    classifier.fit([[1, 2], [3, 4]], [0, 1])
    assert classifier.gap_classifier is not None, "GAPClassifier has not been fit"

def test_GAPClassifier_similarity():
    classifier = GAPClassifier()
    classifier.fit([[1, 2], [3, 4]], [0, 1])
    similarity = classifier.similarity([[1, 2], [3, 4]])
    assert similarity is not None, "Similarity is None"

def test_GAPClassifier_training_similarity_index():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    model = GAPClassifier(random_state=12345, n_estimators=10).fit(data, labels)
    similarity = model.training_similarity()

    for i in range(len(data)):
        similarity_i = model.training_similarity(i)
        results = np.allclose(similarity[i], similarity_i)
        assert results, "Results are not close"