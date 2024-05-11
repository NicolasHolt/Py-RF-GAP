from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from numpy.typing import ArrayLike
import numpy as np
from collections import defaultdict
from copy import deepcopy


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
        # a list of matrices which encode the multiplicity of each sample and size of each leaf for each tree
        tree_matrices = []
        inner_dict_structure = {"leaf_size_": 0}

        # TODO: Verify that we are correctly computing the number of samples... pandas indexing is weird

        self._num_samples_ = np.shape(X)[0]
        self._max_leaf_count_ = 0
        for estimator in super().estimators_:
            self._max_leaf_count_ = max(estimator.tree_.n_leaves, self._max_leaf_count_)

        for index, estimator in enumerate(super().estimators_):
            # get the samples used for training
            # create a set of the samples used for training
            # count the number of times each sample is used and put into a dictionary
            samples_count = dict(Counter(super().estimators_samples_[index]))

            # X has shape (n_samples, n_features)
            in_bag_sample_ids = list(samples_count.keys())
            in_bag_samples = X[in_bag_sample_ids]
            leaf_indices = estimator.apply(in_bag_samples)


            leaf_attributes = defaultdict(lambda: deepcopy(inner_dict_structure))
            sample_to_leaf = {sample: leaf_index for sample, leaf_index in zip(in_bag_samples, leaf_indices)}
            for sample, leaf_index in sample_to_leaf.items():
                leaf_attributes[leaf_index]['leaf_size_'] += samples_count[sample]

            for i, v in enumerate(leaf_attributes.values()):
                v['id'] = i

            tree_matrices.append(
                np.array(
                    np.concatenate(
                        [np.eye(1, self._max_leaf_count_, leaf_attributes[sample_to_leaf[i]]['id'])
                         * samples_count[i] / leaf_attributes[sample_to_leaf[i]]['leaf_size_']
                         if i in samples_count.keys() else np.zeros((1, self._max_leaf_count_))
                         for i in range(self._num_samples_)], axis=0
                    )
                )
            )
        self._ensemble_tensor_ = np.dstack(tuple(tree_matrices))

    def similarity(self, X: ArrayLike):
        # X is gonna be n x m_features
        # Return a matrix of shape n x s where s is the number of samples in the training set
        factor = 1 / self._num_samples_
        result = np.zeros((np.shape(X)[0], self._num_samples_))

        for index, estimator_data in enumerate(self._tree_dict_list_):
            leaf_indices = super().estimators_[index].apply(X)

            for sample_index in estimator_data["tree_samples_set"]:
                # sample_index provides the j value in the range of [0, s)
                c_j = estimator_data["tree_sample_count_dict"][sample_index]
                for i, leaf_index in enumerate(leaf_indices):
                    # i provides the i value in the range of [0, n)
                    if sample_index in estimator_data["leaves_dict"][leaf_index]["leaf_set_"]:
                        result[i, sample_index] += c_j / estimator_data["leaves_dict"][leaf_index]["leaf_size_"]

        # todo similarity = np.einsum('lkm,nmk->ln', INPUT, TREE)

        return result * factor

    def training_similarity(self, index_i: int, index_j: int | None = None):

        def get_similarity(index_i: int, index_j: int):
            if index_i == index_j:
                return 1
            oob_trees_i = self._oob_trees_[index_i]
            acc = 0
            # iterate over trees where sample index_i is OOB and sample index_j is in bag
            for oob_tree in oob_trees_i.intersection(self._itb_trees_[index_j]):
                estimator_data = self._tree_dict_list_[oob_tree]

                # get the leaf index for sample index_i
                leaf_index = super().estimators_[oob_tree].apply(self._training_data_[index_i])

                if index_j not in estimator_data["leaves_dict"][leaf_index]["leaf_set_"]:
                    continue

                c_j = estimator_data["tree_sample_count_dict"][index_j]
                m_cardinality = estimator_data["leaves_dict"][leaf_index]["leaf_size_"]

                acc += c_j / m_cardinality

            return acc * (1 / len(oob_trees_i))

        if index_j is not None:
            return [get_similarity(index_i, index_j)]

        return [get_similarity(index_i, j) for j in range(len(self._num_samples_))]

