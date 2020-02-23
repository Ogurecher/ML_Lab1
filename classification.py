import importlib

from kernel import kernel
from distance import distance


def classify_object(features, labels, test_features, distance_function_type, kernel_function_type, window_type, window_parameter):
    distances_and_labels = []

    for index in range(len(features)):
        distances_and_labels.append({'distance': distance(test_features, features[index], distance_function_type), 'label': labels[index]})

    distances_and_labels = sorted(distances_and_labels, key=lambda k: k['distance'])

    if window_type == 'variable':
        window_radius = distances_and_labels[window_parameter]['distance'] \
            if distances_and_labels[window_parameter-1]['distance'] < distances_and_labels[window_parameter]['distance'] \
            else distances_and_labels[window_parameter-1]['distance'] + 0.000001
    else:
        window_radius = window_parameter

    weighted_class_sum = 0
    kernels_sum = 0

    for index in range(len(features)):
        kernel_value = kernel(distances_and_labels[index]['distance']/window_radius if window_radius != 0 else 0, kernel_function_type)
        weighted_class_sum += distances_and_labels[index]['label'] * kernel_value
        kernels_sum += kernel_value

    predicted_value = weighted_class_sum / kernels_sum if kernels_sum != 0 else weighted_class_sum

    return predicted_value
