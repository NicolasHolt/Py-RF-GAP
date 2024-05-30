from collections import Counter
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from numpy.typing import ArrayLike
import numpy as np
from collections import defaultdict
from copy import deepcopy

class GAPSimilarity:
    
    def __init__(self, random_forest: RandomForestClassifier | RandomForestRegressor, X: ArrayLike):
        self._training_data_ = X
        self.random_forest = random_forest
        self.__leaf_builder(X)

    def __leaf_builder(self, X: ArrayLike):
        # a list of matrices which encode the multiplicity of each sample and size of each leaf for each tree
        tree_matrices = []
        inner_dict_structure = {"leaf_size_": 0}
        out_of_bag_matrix = []

        # TODO: Verify that we are correctly computing the number of samples... pandas indexing is weird

        self._num_samples_ = np.shape(X)[0]
        self._max_leaf_count_ = 0
        self._samples_range_list_ = np.arange(self._num_samples_)
        self._leaf_to_matrix_ = []
        for estimator in self.random_forest.estimators_:
            self._max_leaf_count_ = max(estimator.tree_.n_leaves, self._max_leaf_count_)

        for index, estimator in enumerate(self.random_forest.estimators_):
            # get the samples used for training
            # create a set of the samples used for training
            # count the number of times each sample is used and put into a dictionary
            samples_count = dict(Counter(self.random_forest.estimators_samples_[index]))

            # X has shape (n_samples, n_features)
            in_bag_sample_ids = list(samples_count.keys())
            in_bag_samples = X[in_bag_sample_ids]
            leaf_indices = estimator.apply(in_bag_samples)

            out_of_bag_matrix.append(np.isin(self._samples_range_list_, in_bag_sample_ids, invert=True).astype(int))

            leaf_attributes = defaultdict(lambda: deepcopy(inner_dict_structure))
            sample_to_leaf = {sample: leaf_index for sample, leaf_index in zip(in_bag_sample_ids, leaf_indices)}
            for sample, leaf_index in sample_to_leaf.items():
                leaf_attributes[leaf_index]['leaf_size_'] += samples_count[sample]

            for i, v in enumerate(leaf_attributes.values()):
                v['id'] = i
            self._leaf_to_matrix_.append(leaf_attributes)

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
        # todo mayhaps remove leaf_size, such woah
        self._ensemble_tensor_ = np.dstack(tuple(tree_matrices))
        self._out_of_bag_matrix_ = np.array(out_of_bag_matrix)

    def similarity(self, X: ArrayLike):
        factor = len(self.random_forest.estimators_)
        mapped_leaves = self.random_forest.apply(X)
        tree_matrices = []
        for k in range(factor):
            tree_matrices.append(
                np.array(
                    np.concatenate(
                        [np.eye(1, self._max_leaf_count_, self._leaf_to_matrix_[k][mapped_leaves[i, k]]['id'])
                         for i in range(mapped_leaves.shape[0])], axis=0)
                )
            )
        applied_tensor = np.dstack(tuple(tree_matrices))

        similarity_unweighted = np.einsum('lmk,nmk->ln', applied_tensor, self._ensemble_tensor_)
        return np.divide(similarity_unweighted, factor)


    def training_similarity(self, index_i: int | None = None):
        oob_mat = self._out_of_bag_matrix_[:, index_i].reshape(len(self._out_of_bag_matrix_),1) if index_i is not None else self._out_of_bag_matrix_
        mapped_leaves = self.random_forest.apply([self._training_data_[index_i, :]]) if index_i is not None else self.random_forest.apply(self._training_data_)
        tree_matrices = []
        for k in range(len(self.random_forest.estimators_)):
            tree_matrices.append(
                np.array(
                    np.concatenate(
                        [np.eye(1, self._max_leaf_count_, self._leaf_to_matrix_[k][mapped_leaves[i, k]]['id'])
                         for i in range(mapped_leaves.shape[0])], axis=0)
                    )
            )
        training_tensor = np.dstack(tuple(tree_matrices))

        intermediate_tensor = np.einsum('lmk,nmk->lnk', training_tensor, self._ensemble_tensor_)
        similarity_unweighted = np.einsum('lnk,kl->ln', intermediate_tensor, oob_mat)
        similarities = np.divide(similarity_unweighted, np.sum(oob_mat, axis=0))
        return np.add(similarities, np.eye(1, len(self._training_data_))) if index_i is not None else np.add(similarities, np.diag(np.ones(len(self._training_data_))))