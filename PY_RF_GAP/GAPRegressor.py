from sklearn.ensemble import RandomForestRegressor
from PY_RF_GAP.GAPSimilarity import GAPSimilarity
from numpy.typing import ArrayLike


class GAPRegressor(RandomForestRegressor):
    def __init___(self, *args, **kwargs):
        RandomForestRegressor.__init__(*args, **kwargs)
        self.gap_classifier = None

    def fit(self, X: ArrayLike,
            y: ArrayLike,
            sample_weight: ArrayLike | None = None):
        """
        Fit the GAP classifier to the given data.

        Parameters
        ----------
        X : ArrayLike
            The input data to fit the model to.

        y : ArrayLike
            The target labels to fit the model to.

        sample_weight : ArrayLike | None
            The sample weights to apply to the training data.
        
        Returns
        -------
        GAPClassifier
            The fitted GAP classifier.
        """
        super().fit(X, y, sample_weight)
        self.gap_classifier = GAPSimilarity(self, X)
        return self

    def similarity(self, X: ArrayLike):
        """
        Calculate the similarity scores of the given data using the trained GAP classifier.

        Parameters
        ----------
        X : ArrayLike
            The input data for which to calculate the similarity score. 
            This should not be an empty array.

        Raises
        ------
        ValueError
            If the GAP classifier has not been fit before calling this method.

        Returns
        -------
        ArrayLike
            The similarity score calculated by the GAP classifier.
        """
        if self.gap_classifier is None:
            raise ValueError("Model has not been fit")
        return self.gap_classifier.similarity(X)

    def training_similarity(self, index_i: int | None = None):
        """
        Calculate the similarity scores of either the training data point at index `index_i` 
        or the entire training dataset if `index_i` is `None`.

        Parameters
        ----------
        index_i : int | None
            The input index for which to calculate the similarity score.
            If `None`, the similarity score for the entire training dataset is calculated.

        Raises
        ------
        ValueError
            If the GAP classifier has not been fit before calling this method.

        Returns
        -------
        ArrayLike
            The similarity score calculated by the GAP classifier.
        """
        if self.gap_classifier is None:
            raise ValueError("Model has not been fit")
        return self.gap_classifier.training_similarity(index_i)