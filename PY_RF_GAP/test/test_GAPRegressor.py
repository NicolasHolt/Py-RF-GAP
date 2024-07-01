import unittest
from PY_RF_GAP.GAPRegressor import GAPRegressor
from sklearn import datasets
import numpy as np

class TestGAPClassifier(unittest.TestCase):

    def test_GAPClassifier_instance(self):
        classifier = GAPRegressor()
        self.assertIsInstance(classifier, GAPRegressor, "Object is not an instance of GAPClassifier")

    def test_GAPClassifier_fit(self):
        classifier = GAPRegressor()
        classifier.fit([[1, 2], [3, 4]], [0, 1])
        self.assertIsNotNone(classifier.gap_classifier, "GAPClassifier has not been fit")

    def test_GAPClassifier_similarity(self):
        classifier = GAPRegressor()
        classifier.fit([[1, 2], [3, 4]], [0, 1])
        similarity = classifier.similarity([[1, 2], [3, 4]])
        self.assertIsNotNone(similarity, "Similarity is None")

    def test_GAPClassifier_training_similarity_index(self):
        iris = datasets.load_iris()
        data = iris.data
        labels = iris.target
        model = GAPRegressor(random_state=12345, n_estimators=10).fit(data, labels)
        similarity = model.training_similarity()

        for i in range(len(data)):
            similarity_i = model.training_similarity(i)
            self.assertTrue(np.allclose(similarity[i], similarity_i), "Results are not close")

if __name__ == '__main__':
    unittest.main()