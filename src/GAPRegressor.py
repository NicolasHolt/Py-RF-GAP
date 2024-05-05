from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from numpy.typing import ArrayLike

class GAPRegressor(RandomForestRegressor):

    def __init___(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    #Were you trying to import a specific type for X? The types in sklearn are usually just for understanding of
    # what can be passed in and not actual types. ArrayLike is a valid Numpy type though
    def fit(self, X: MatrixLike | ArrayLike,
    y: MatrixLike | ArrayLike,
    sample_weight: ArrayLike | None = None):
        super().fit(X, y)
        self.do_stuff(X, y)
        return self
    
    def do_stuff(self, X, y):
        estimator_computed = []

        for index, estimator in enumerate(super().estimators_):
            # get the samples used for training
            # create a set of the samples used for training
            # count the number of times each sample is used and put into a dictionary
            samples = dict(Counter(super().estimators_samples_[index]))
            estimator_data = {
                "estimator_samples_set": set(samples.keys()),
                "estimator_samples_count": samples,
                "leaves": [{"set_": set(), "length_": 0}] * estimator.tree_.n_leaves
            }

            # X has shape (n_samples, n_features)
            to_compute = list(estimator_data["estimator_samples_set"])
            X[to_compute]
            results = each.apply(for the points in the list(estimator_samples_set))

            zip(list(estimator_samples_set), results)

            for sample, leaf in zip(list(estimator_samples_set), results):
                leaves[leaf]["set_"].add(sample)
                leaves[leaf]["length_"] += estimator_samples_count[sample]

            estimator_computed.append(estimator_data)
            print(each)