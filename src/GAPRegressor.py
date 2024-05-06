from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from numpy.typing import ArrayLike
import numpy as np

class GAPRegressor(RandomForestRegressor):

    def __init___(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X: ArrayLike,
            y: ArrayLike,
            sample_weight: ArrayLike | None = None):
        self._training_data_ = X
        super().fit(X, y)
        self.__leaf_builder(X)
        return self
    
    def __leaf_builder(self, X: ArrayLike):
        # a list that contains the in bag samples for each tree and their corresponding leaves
        estimator_computed = []

        # TODO: Verify that we are correctly computing the number of samples... pandas indexing is weird
        # dictionary that maps from sample index to trees OOB 
        oob_trees = {k: set() for k in range(np.shape(X)[0])}
        itb_trees = {k: set() for k in range(np.shape(X)[0])}
        
        # set of all samples
        samples_set = set(range(np.shape(X)[0]))
        self._num_samples_ = np.shape(X)[0]

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
            oob_trees[index] = samples_set - estimator_data["estimator_samples_set"]
            itb_trees[index] = estimator_data["estimator_samples_set"]

            # X has shape (n_samples, n_features)
            to_compute = list(estimator_data["estimator_samples_set"])
            in_bag_samples = X[to_compute]
            leaf_indicies = estimator.apply(in_bag_samples)

            for i, j in zip(to_compute, leaf_indicies):
                estimator_data["leaves"][j]["set_"].add(i)
                estimator_data["leaves"][j]["length_"] += samples[i]

            estimator_computed.append(estimator_data)

        self._estimator_computed_ = estimator_computed
        self._oob_trees_ = oob_trees
        self._itb_trees_ = itb_trees

    def similarity(self, X: ArrayLike):
        # X is gonna be n x m_features
        # Return a matrix of shape n x s where s is the number of samples in the training set
        factor = 1 / self._num_samples_
        result = np.zeros((np.shape(X)[0], self._num_samples_))

        for index, estimator_data in enumerate(self._estimator_computed_):
            leaf_indicies = super().estimators_[index].apply(X)

            for sample_index in estimator_data["estimator_samples_set"]:
                # sample_index provides the j value in the range of [0, s)
                c_j = estimator_data["estimator_samples_count"][sample_index]
                for i, leaf_index in enumerate(leaf_indicies):
                    # i provides the i value in the range of [0, n)
                    if sample_index in estimator_data["leaves"][leaf_index]["set_"]:
                        result[i, sample_index] += c_j / estimator_data["leaves"][leaf_index]["length_"]
                        
        return result * factor
                    

    def training_similarity(self, index_i: int, index_j: int | None = None):
        
        def get_similarity(index_i: int, index_j: int):
            if index_i == index_j:
                return 1
            oob_trees_i = self._oob_trees_[index_i]
            acc = 0
            if index_j is not None:
                # iterate over trees where sample index_i is OOB and sample index_j is in bag
                for oob_tree in oob_trees_i.intersection(self._itb_trees_[index_j]):
                    estimator_data = self._estimator_computed_[oob_tree]

                    # get the leaf index for sample index_i
                    leaf_index = super().estimators_[oob_tree].apply(self._training_data_[index_i])

                    if index_j not in estimator_data["leaves"][leaf_index]["set_"]:
                        continue

                    c_j = estimator_data["estimator_samples_count"][index_j]
                    m_cardinality = estimator_data["leaves"][leaf_index]["length_"]

                    acc += c_j / m_cardinality

                    
            return acc * (1 / len(oob_trees_i))
        
        if index_j is not None:
            return [get_similarity(index_i, index_j)]
        
        return [get_similarity(index_i, j) for j in range(len(self._num_samples_))]
