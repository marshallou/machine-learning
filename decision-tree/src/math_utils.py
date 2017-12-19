import math
import re
from functools import reduce

########################
# Math helper functions
########################


def compute_entropy(data_set):
    """
    :param data_set: multi-array where rows are data point and column are features. The last column is label.
    :return: entropy of the data set
    """
    if (not data_set) or (not data_set[0]):
        return 0

    label_index = len(data_set[0]) - 1
    label_count_map = {}

    for data_point in data_set:
        label = data_point[label_index]

        if label not in label_count_map:
            label_count_map[label] = 0
        else:
            label_count_map[label] = label_count_map[label] + 1

    res = 0
    for label in label_count_map:
        p = label_count_map[label] / len(data_set)
        if p == 0:
            res += 0
        else:
            res -= p * math.log2(p)


    return res


def compute_binary_entropy(p):
    return - p * math.log(p) - (1 - p) * math.log(1 - p)


def compute_mean_label(data_set):
    label_index = len(data_set[0]) - 1
    sum = reduce((lambda x, y: x + y),
                 [data_point[label_index] for data_point in data_set])

    return sum / len(data_set)


def compute_mean_square_error(data_set, mean=None):
    if not data_set:
        return 0.0, 0.0, 0.0

    label_index = len(data_set[0]) - 1

    if mean is None:
        mean = compute_mean_label(data_set)

    sum = reduce((lambda x, y: x + math.pow((y - mean), 2)),
                 [data_point[label_index] for data_point in data_set], 0)

    return sum, sum / len(data_set), mean


def scale_and_shift(data_set):
    """
    :param data_set: multi-array where rows are data point and column are features. The last column is label
    :return: normalized data using scale and shift. The label will stay unchanged
    """
    if not data_set:
        return data_set

    feature_length = len(data_set[0]) - 1

    if feature_length <= 0:
        return data_set

    max_features = data_set[0][0:feature_length]
    min_features = max_features[:]

    for data_point in data_set:
        max_features = [data_point[i] if max_features[i] < data_point[i] else max_features[i]
                        for i in range(feature_length)]
        min_features = [data_point[i] if min_features[i] > data_point[i] else min_features[i]
                        for i in range(feature_length)]

    for data_point in data_set:
        for i in range(feature_length):
            data_point[i] = (data_point[i] - min_features[i]) * 1.0 / (max_features[i] - min_features[1])
    return data_set


def mean_features(data_set):
    """
    :param data_set:
    :return: return mean of each feature of the data set as an array
    """
    if not data_set:
        return []

    feature_length = len(data_set[0]) - 1

    if feature_length <= 0:
        return []

    mean = [0.0] * feature_length
    for data_point in data_set:
        mean = [mean[i] + data_point[i] for i in range(feature_length)]

    return [mean[i] / len(data_set) for i in range(feature_length)]


def variance_features(data_set):
    """
    :param data_set:
    :return: return variance of each feature of the data set as an array
    """
    if len(data_set) <= 1:
        return []

    feature_length = len(data_set[0]) - 1

    if feature_length <= 0:
        return []

    mean = mean_features(data_set)
    variance = [0.0] * feature_length

    for data_point in data_set:
        variance = [variance[i] + math.pow((data_point[i] - mean[i]), 2)
                           for i in range(feature_length)]
    return [variance[i] / (len(data_set) - 1) for i in range(feature_length)]


def stand_dev_features(data_set):
    """
    :param data_set:
    :return: return standard deviation of each feature of data set as an array
    """
    if not data_set:
        return []

    feature_length = len(data_set[0]) - 1

    if feature_length <= 0:
        return []

    variance = variance_features(data_set)
    return [math.sqrt(variance[i]) for i in range(feature_length)]


def zero_mean(data_set):
    """
    :param data_set: multi-array where rows are data point and column are features. The last column is label
    :return: normalized data using zero mean. The label will stay unchanged
    """
    if len(data_set) <= 1:
        return data_set

    feature_length = len(data_set[0]) - 1

    if feature_length <= 0:
        return data_set

    mean = mean_features(data_set)
    standard_deviation = stand_dev_features(data_set)

    for data_point in data_set:
        for i in range(feature_length):
            data_point[i] = (data_point[i] - mean[i]) / standard_deviation[i]

    return data_set

##########################
# Other helper functions
##########################


def load_data(file_name, delimiter=None):
    """
    :param file_name:
    :return: two dimension array. The data within is converted to float
    """
    with open(file_name) as file:
        content = file.readlines()
    data_set = []

    for line in content:
        if delimiter is None:
            data_point = list(map(float, line.strip().split(",")))
        else:
            data_point = list(map(float, re.split(delimiter, line.strip())))
        data_set.append(data_point)
    return data_set


def print_data(data_set):
    if not data_set:
        return data_set

    data_length = len(data_set[0])

    if data_length <= 0:
        return data_set

    file = open("../data/test.txt", 'w')
    for data_point in data_set:
        for i in range(data_length):
            print(data_point[i], end=' ')
            print(data_point[i], end=' ', file=file)
        print()
        print(file=file)
    file.close()


if __name__ == "__main__":
    data_set = load_data("../data/spambase.data.txt")
    data_set = zero_mean(data_set)
    print_data(data_set)
