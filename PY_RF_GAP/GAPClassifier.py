from sklearn.ensemble import RandomForestClassifier
from PY_RF_GAP.GAPSimilarity import GAPSimilarity
from numpy.typing import ArrayLike

class GAPClassifier(RandomForestClassifier):
    def __init___(self, *args, **kwargs):
        RandomForestClassifier.__init__(*args, **kwargs)
        self.gap_classifier = None

    def fit(self, X: ArrayLike,
            y: ArrayLike,
            sample_weight: ArrayLike | None = None):
        super().fit(X, y, sample_weight)
        self.gap_classifier = GAPSimilarity(self, X)
        return self
    
    def similarity(self, X: ArrayLike):
        if self.gap_classifier is None:
            raise ValueError("Model has not been fit")
        return self.gap_classifier.similarity(X)
    
    def training_similarity(self, index_i: int | None = None):
        if self.gap_classifier is None:
            raise ValueError("Model has not been fit")
        return self.gap_classifier.training_similarity(index_i)