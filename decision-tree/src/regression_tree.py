import math
from src.math_utils import load_data, zero_mean, scale_and_shift, compute_mean_square_error
import matplotlib.pyplot as plt
import numpy as np

##########################
# regression tree
##########################


class RegressionTree:
    """
    data_set: the data set in this node
    mean_square_error: the entropy of this node

    left: left child
    right: right child
    feature_index: indicate which feature should I use for split
    threshold: indicate the threshold of feature value used for split
    mse_drop: the drop of mean square error after split

    label: indicate the predicate result if this node is leaf node
    is_leaf: indicate this node is leaf node or not
    """
    def __init__(self, data_set, sum_of_mean_square_error, mean_square_error, mean):
        self.data_set = data_set
        self.sum_of_mse = sum_of_mean_square_error
        self.mean_square_error = mean_square_error
        self.mean = mean

        # data generate from "split" call
        self.max_mse_drop = None
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None

        # data generate from "set_as_leaf" call
        self.is_leaf = False

    def split(self):
        """
        split the node by using the max information gain
        :param tree_node:
        :return: the information gain of this split
        """
        max_mse_drop = 0
        feature_index_of_max_mse_drop = None
        threshold_of_max_mse_drop = None
        left_node_of_max_mse_drop = None
        right_node_of_max_mse_drop = None

        data_set = self.data_set
        feature_length = len(data_set[0]) - 1

        for feature_index in range(feature_length):
            all_thresholds = sorted([data_point[feature_index] for data_point in data_set])
            for idx, threshold in enumerate(all_thresholds):
                left_data_set = []
                right_data_set = []
                # skip duplicate threshold
                if idx > 0 and threshold == all_thresholds[idx - 1]:
                    continue

                for data_point in data_set:
                    if data_point[feature_index] < threshold:
                        left_data_set.append(data_point)
                    else:
                        right_data_set.append(data_point)

                left_sum_of_mean_square_error, left_mean_square_error, left_mean = compute_mean_square_error(left_data_set)
                right_sum_of_mean_square_error, right_mean_square_error, right_mean = compute_mean_square_error(right_data_set)
                current_mse_drop = self.mean_square_error - left_sum_of_mean_square_error / len(data_set) \
                                   - right_sum_of_mean_square_error / len(data_set)

                if current_mse_drop >= max_mse_drop:
                    max_mse_drop = current_mse_drop
                    feature_index_of_max_mse_drop = feature_index
                    threshold_of_max_mse_drop = threshold
                    left_node_of_max_mse_drop = RegressionTree(left_data_set,
                                                               left_sum_of_mean_square_error,
                                                               left_mean_square_error,
                                                               left_mean)
                    right_node_of_max_mse_drop = RegressionTree(right_data_set,
                                                                right_sum_of_mean_square_error,
                                                                right_mean_square_error,
                                                                right_mean)

        self.feature_index = feature_index_of_max_mse_drop
        self.threshold = threshold_of_max_mse_drop
        self.max_mse_drop = max_mse_drop
        self.left = left_node_of_max_mse_drop
        self.right = right_node_of_max_mse_drop
        print("max_mse_drop: %f" % max_mse_drop)
        print("total: %d" %(len(data_set)))
        print("left: %d" %(len(left_node_of_max_mse_drop.data_set)))
        print("right: %d" %(len(right_node_of_max_mse_drop.data_set)))
        print()
        return self

    def set_as_leaf(self):
        self.is_leaf = True

    def predict(self, data_point):

        if self.is_leaf:
            return self.mean

        if data_point[self.feature_index] < self.threshold:
            return self.left.predict(data_point)

        return self.right.predict(data_point)


##########################
# build regression tree
##########################
MIN_NUM_DATA_SET = 5
MINIMUM_MSE_DROP = 0.1


def build_regression_tree(tree_node, max_depth=None, min_mse_drop=None):
    """
    :param tree_node:
    :param max_depth: optional, max_depth of the tree
    :return:
    """
    if not can_split(tree_node, max_depth):
        tree_node.set_as_leaf()
        return tree_node

    tree_node.split()

    # if max information gain is less than minimum information gain, stop split
    if (min_mse_drop is not None and tree_node.max_mse_drop < min_mse_drop) \
            or (tree_node.max_mse_drop < MINIMUM_MSE_DROP):
        tree_node.left = None
        tree_node.right = None
        tree_node.set_as_leaf()
    else:
        tree_node.left = build_regression_tree(tree_node.left, None if max_depth is None else max_depth - 1, min_mse_drop)
        tree_node.right = build_regression_tree(tree_node.right, None if max_depth is None else max_depth - 1, min_mse_drop)
    return tree_node


def can_split(tree_node, max_depth):
    """
    split requirements:
        1) the number of data points in each node should be more than MIN_NUM_DATA_SET,
        the value can be adjusted when initiating the program by using percentage of total
        number of data points
        2) if the max_depth is not None, the max_depth has to be equal or more than 1

    :param tree_node:
    :param max_depth: the depth of current tree node
    :return: true if we can still split
    """
    if not tree_node:
        return tree_node

    data_set = tree_node.data_set

    if not data_set:
        return False

    if len(data_set) < MIN_NUM_DATA_SET:
        return False

    if max_depth is not None and max_depth <= 1:
        return False

    return True


##########################
# measure overfitting
##########################
class Overfitting:

    def __init__(self, training_data_location, testing_data_location):
        """
        data delimiter is "\s+"
        :param training_data_set: location of training data set
        :param testing_data_set: location of training data set
        """
        training_data_set = load_data(training_data_location, "\s+")
        self.training_data_set = zero_mean(training_data_set)

        testing_data_set = load_data(testing_data_location, "\s+")
        self.testing_data_set = zero_mean(testing_data_set)

        sum_of_mean_square_error, mean_square_error, mean = compute_mean_square_error(training_data_set)
        self.sum_of_mean_square_error = sum_of_mean_square_error
        self.mean_square_error = mean_square_error
        self.mean = mean

    def measure_overfit(self, min_mse_drop):
        """
        Build regression tree model using training data. Run the model and test against both training data and testing data.
        return mse result of training data and testing data
        :param min_mse_drop: the minimum mean square error drop requirement has to meet, in order for the current node to split
        :return: mse_of_training_data, mse_of_testing_data
        """
        regression_tree = RegressionTree(self.training_data_set, self.sum_of_mean_square_error, self.mean_square_error, self.mean)
        regression_tree = build_regression_tree(regression_tree, None, min_mse_drop)

        training_data_set = self.training_data_set
        testing_data_set = self.testing_data_set
        label_index = len(training_data_set[0]) - 1

        sum_of_error_train = 0.0

        for data_point in training_data_set:
            predication = regression_tree.predict(data_point)
            real_result = data_point[label_index]
            sum_of_error_train += math.pow((real_result - predication), 2)

        mse_train = sum_of_error_train / len(training_data_set)

        sum_of_error_test = 0.0

        for data_point in testing_data_set:
            predication = regression_tree.predict(data_point)
            real_result = data_point[label_index]
            sum_of_error_test += math.pow((real_result - predication), 2)

        mse_test = sum_of_error_test / len(testing_data_set)

        return mse_train, mse_test

    def plot(self, start_mse_drop, end_mse_drop, step):
        min_mse_drops = np.arange(start_mse_drop, end_mse_drop, step)
        train_mean_square_errors = []
        test_mean_square_errors = []

        for min_mse_drop in np.nditer(min_mse_drops):
            train_mean_square_error, test_mean_square_error = self.measure_overfit(min_mse_drop)
            train_mean_square_errors.append(train_mean_square_error)
            test_mean_square_errors.append(test_mean_square_error)
            print("train data mse array: " + str(train_mean_square_errors))
            print("test data mse array: " + str(test_mean_square_errors))

        plt.plot(min_mse_drops, np.array(train_mean_square_errors), min_mse_drops, np.array(test_mean_square_errors))
        plt.xlabel("minimum mean square error drop")
        plt.ylabel("training data mse and testing data mse")
        plt.show()

if __name__ == "__main__":
    training_data_location = "../data/housing_train.txt"
    testing_data_location = "../data/housing_test.txt"
    overfit = Overfitting(training_data_location, testing_data_location)
    overfit.plot(0.0, 20, 0.5)


    """
    #sample code for building regression tree:
    
    training_data_set = load_data("../data/housing_train.txt", "\s+")
    training_data_set = zero_mean(training_data_set)

    testing_data_set = load_data("../data/housing_test.txt", "\s+")
    testing_data_set = zero_mean(testing_data_set)

    label_index = len(training_data_set[0]) - 1

    sum_of_mean_square_error, mean_square_error, mean = compute_mean_square_error(training_data_set)
    regression_tree = RegressionTree(training_data_set, sum_of_mean_square_error, mean_square_error, mean)
    regression_tree = build_regression_tree(regression_tree)

    sum_of_error_train = 0.0

    for data_point in training_data_set:
        predication = regression_tree.predict(data_point)
        real_result = data_point[label_index]
        sum_of_error_train += math.pow((real_result - predication), 2)

    mse_train = sum_of_error_train / len(training_data_set)
    print("mse of training data: " + str(mse_train))

    sum_of_error_test = 0.0
    for data_point in testing_data_set:
        predication = regression_tree.predict(data_point)
        real_result = data_point[label_index]
        sum_of_error_test += math.pow((real_result - predication), 2)

    mse = sum_of_error_test / len(testing_data_set)
    print("mse of testing data: " + str(mse))
    """










