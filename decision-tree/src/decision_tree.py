from functools import reduce
import numpy as np
from src.math_utils import load_data, zero_mean, scale_and_shift, compute_entropy, compute_binary_entropy
import matplotlib.pyplot as plt
##########################
# decision tree
##########################
SPAM = "spam"
NON_SPAM = "non-spam"


class DecisionTreeNode:
    """
    data_set: the data set in this node
    entropy: the entropy of this node

    left: left child
    right: right child
    feature_index: indicate which feature should I use for split
    threshold: indicate the threshold of feature value used for split
    max_information_gain:

    label: indicate the predicate result if this node is leaf node
    is_leaf: indicate this node is leaf node or not
    """
    def __init__(self, data_set, entropy):
        self.data_set = data_set
        self.entropy = entropy

        # data generate from "split" call
        self.max_information_gain = None
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None

        # data generate from "set_as_leaf" call
        self.label = None
        self.is_leaf = False

    def split(self):
        """
        split the node by using the max information gain
        :param tree_node:
        :return: the information gain of this split
        """
        max_information_gain = 0
        feature_index_of_max_ig = None
        threshold_of_max_ig = None
        left_node_of_max_ig = None
        right_node_of_max_ig = None

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

                num_of_spam_in_left = 0
                num_of_spam_in_right = 0
                for data_point in data_set:
                    if data_point[feature_index] < threshold:
                        left_data_set.append(data_point)

                        if data_point[label_index] == 1.0:
                            num_of_spam_in_left += 1
                    else:
                        right_data_set.append(data_point)

                        if data_point[label_index] == 1.0:
                            num_of_spam_in_right += 1

                left_entropy = compute_binary_entropy(num_of_spam_in_left / len(left_data_set))
                right_entropy = compute_binary_entropy(num_of_spam_in_right / len(right_data_set))
                current_information_gain = self.entropy - len(left_data_set) * 1.0 * left_entropy / len(data_set) \
                                           - len(right_data_set) * 1.0 * right_entropy / len(data_set)

                if current_information_gain >= max_information_gain:
                    max_information_gain = current_information_gain
                    feature_index_of_max_ig = feature_index
                    threshold_of_max_ig = threshold
                    left_node_of_max_ig = DecisionTreeNode(left_data_set, left_entropy)
                    right_node_of_max_ig = DecisionTreeNode(right_data_set, right_entropy)

        self.feature_index = feature_index_of_max_ig
        self.threshold = threshold_of_max_ig
        self.max_information_gain = max_information_gain
        self.left = left_node_of_max_ig
        self.right = right_node_of_max_ig
        print("max_information_gain: %f" % max_information_gain)
        print("total: %d" %(len(data_set)))
        print("left: %d" %(len(left_node_of_max_ig.data_set)))
        print("right: %d" %(len(right_node_of_max_ig.data_set)))
        print()
        return self

    def set_as_leaf(self):
        self.is_leaf = True
        self.label = self.compute_label_of_leaf_node()

    def compute_label_of_leaf_node(self):
        """
        :param tree_node:
        :return: "spam" if the more than 0.5 of the data_points in data set is spam, "non-spam" otherwise
        """
        data_set = self.data_set

        if not data_set:
            return None

        if len(data_set[0]) <= 2:
            return None

        num_of_spam = 0
        label_index = len(data_set[0]) - 1

        for data_point in data_set:
            if data_point[label_index] == 1:
                num_of_spam += 1

        ratio_of_spam = num_of_spam * 1.0 / len(data_set)

        if ratio_of_spam > 0.5:
            return SPAM
        else:
            return NON_SPAM

    def predict(self, data_point):

        if self.is_leaf:
            return self.label

        if data_point[self.feature_index] < self.threshold:
            return self.left.predict(data_point)

        return self.right.predict(data_point)


##########################
# build decision tree
##########################
MIN_NUM_DATA_SET = 5
MINIMUM_INFORMATION_GAIN = 0.01


def build_decision_tree(tree_node, max_depth=None, min_information_gain=None):
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
    if (min_information_gain is not None and tree_node.max_information_gain < min_information_gain) or \
                    tree_node.max_information_gain < MINIMUM_INFORMATION_GAIN:
        tree_node.left = None
        tree_node.right = None
        tree_node.set_as_leaf()
    else:
        tree_node.left = build_decision_tree(tree_node.left, None if max_depth is None else max_depth - 1, min_information_gain)
        tree_node.right = build_decision_tree(tree_node.right, None if max_depth is None else max_depth - 1, min_information_gain)
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


def split_data_set(data_set, i, k):
    """
    split the data set into k folds, use ith folds as testing data and the other k-1 folds
    of data as training data
    :param data_set:
    :param i: start from 0
    :param k:
    :return:
    """
    num_of_data_points = len(data_set)
    num_of_data_in_fold = int(num_of_data_points / k)
    start_index_of_ith_fold = num_of_data_in_fold * i
    end_index_of_ith_fold = start_index_of_ith_fold + num_of_data_in_fold

    testing_data_set = data_set[start_index_of_ith_fold: end_index_of_ith_fold]
    training_data_set = data_set[0:start_index_of_ith_fold] + data_set[end_index_of_ith_fold:]
    return training_data_set, testing_data_set


if __name__ == "__main__":
    data_set = load_data("../data/spambase.data.txt")
    data_set = zero_mean(data_set)

    label_index = len(data_set[0]) - 1

    min_information_gains = np.arange(0.5, 10.5, 1)
    avg_error_rates = []
    for min_information_gain in np.nditer(min_information_gains):
        error_rates = []
        for i in range(10):
            training_data_set, testing_data_set = split_data_set(data_set, i, 10)
            entropy = compute_entropy(training_data_set)
            decision_tree = DecisionTreeNode(training_data_set, entropy)
            decision_tree = build_decision_tree(decision_tree, None, min_information_gain)
            num_of_error = 0
            num_of_data = len(testing_data_set)

            for data_point in testing_data_set:
                predication = decision_tree.predict(data_point)
                real_result = NON_SPAM if data_point[label_index] == 0.0 else SPAM
                if predication != real_result:
                    num_of_error += 1

            error_rates.append(num_of_error * 1.0 / num_of_data)
        print("error rates: " + str(error_rates))
        avg_error_rate = reduce((lambda x, y: x + y), error_rates) * 1.0 / len(error_rates)
        avg_error_rates.append(avg_error_rate)
        print("average error rates: %f" % avg_error_rate)

    plt.plot(min_information_gains, np.array(avg_error_rates))
    plt.xlabel("minimum information gain")
    plt.ylabel("avg error rate")
    plt.show()