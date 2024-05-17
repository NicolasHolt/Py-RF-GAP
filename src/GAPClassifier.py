from sklearn.ensemble import RandomForestClassifier
from GAPSimilarity import GAPSimilarity
from numpy.typing import ArrayLike

class GAPClassifier(RandomForestClassifier):
    def __init___(self, *args, **kwargs):
        RandomForestClassifier.__init__(*args, **kwargs)

    def fit(self, X: ArrayLike,
            y: ArrayLike,
            sample_weight: ArrayLike | None = None):
        self._training_data_ = X
        super().fit(X, y)
        self.gap_classifier = GAPSimilarity(self, X)
        self.__leaf_builder(X)
        return self