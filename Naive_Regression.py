import pandas as pd
import numpy as np
import collections


def minkovski_distance(x, y, p = 2):
    squared_errors = [(xi - yi)**p for xi, yi in zip(x, y)]
    return sum(squared_errors)**(1/p)


def get_k_nearest_neighbors (k, x, X):
    neighbors = []
    removed_self = False

    for index in range(len(X)):
        xi = X[index]
        
        if (removed_self == False) and (np.array_equal(x, xi)):
            removed_self = True
            continue

        neighbors.append({'distance': minkovski_distance(x, xi), 'index': index})

    return sorted(neighbors, key=lambda elem: elem['distance'])[:k]


def kernel (r, kernel_type = 'triangular'):
    if kernel_type == 'uniform':
        return max(1/2 - (abs(r) // 1), 0)

    if kernel_type == 'triangular':
        return max((1-abs(r)), 0)

    if kernel_type == 'epanechnikov':
        return max((3/4 * (1-r**2)), 0)


def smooth_kernel (h, x, dataset, kernel_type = 'triangular'):
    weighted_class_sum = 0
    kernels_sum = 0

    for item in dataset:
        weighted_class_sum += item[-1] * kernel(minkovski_distance(item[:-1], x)/h, kernel_type)
        kernels_sum += kernel(minkovski_distance(item[:-1], x)/h, kernel_type)

    return weighted_class_sum/kernels_sum


data_csv = './data/glass.csv'

dataframe = pd.read_csv(data_csv)

class_mapping = {name: index for index, name in enumerate(sorted(list(set(dataframe.Type))))}

dataframe = dataframe.replace({'Type': class_mapping})

np_dataset = np.array(dataframe)

features = np_dataset[:, :-1]
labels = np_dataset[:, -1]

h = 5
correct = 0

for index in range (len(np_dataset)):
    res = round(smooth_kernel(h, features[index], np_dataset, 'epanechnikov'))
    y = labels[index]

    if res == y:
        correct += 1

print(correct)