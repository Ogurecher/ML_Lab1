import numpy as np
from sklearn.metrics import f1_score

import parameters
from classification import classify_object

def find_best_parameters(features, labels, num_classes):
    results = []

    for distance_function in parameters.distance_functions:
        for kernel_function in parameters.kernel_functions:
            for window_type in parameters.window_types:
                for window_parameter in parameters.window_widths if window_type == 'fixed' else parameters.window_neighbours:
                    predicted_labels = []

                    for index in range(len(features)):
                        predicted_value = classify_object(features[np.arange(len(features)) != index],
                                                          labels[np.arange(len(labels)) != index], features[index],
                                                          distance_function,
                                                          kernel_function, window_type, window_parameter)

                        predicted_label = round(predicted_value)
                        predicted_labels.append(predicted_label)

                    f_score = f1_score(labels, predicted_labels, labels=[i for i in range(num_classes)],
                                       average='weighted')
                    results.append(
                        {'f_score': f_score, 'distance_function': distance_function, 'kernel_function': kernel_function,
                         'window_type': window_type, 'window_parameter': window_parameter})

    sorted_results = sorted(results, key=lambda configuration: configuration['f_score'], reverse=True)

    return sorted_results

def find_best_parameters_one_hot(features, labels, encoded_labels, num_classes):
    results = []

    for distance_function in parameters.distance_functions:
        for kernel_function in parameters.kernel_functions:
            for window_type in parameters.window_types:
                for window_parameter in parameters.window_widths if window_type == 'fixed' else parameters.window_neighbours:
                    predicted_labels = []

                    for index in range(len(features)):

                            predicted_label_values = []

                            for label_index in range(len(encoded_labels[index])):
                                predicted_label_value = classify_object(features[np.arange(len(features)) != index],
                                                                  encoded_labels[np.arange(len(encoded_labels)) != index][:, label_index], features[index],
                                                                  parameters.best_combination['distance_function'],
                                                                  parameters.best_combination['kernel_function'],
                                                                  parameters.best_combination['window_type'],
                                                                  window_parameter)

                                predicted_label_values.append(predicted_label_value)

                            predicted_value = [i for i, j in enumerate(predicted_label_values) if j == max(predicted_label_values)][0]
                            predicted_label = round(predicted_value)
                            predicted_labels.append(predicted_label)

                    f_score = f1_score(labels, predicted_labels, labels=[i for i in range(num_classes)],
                                       average='weighted')
                    results.append(
                        {'f_score': f_score, 'distance_function': distance_function, 'kernel_function': kernel_function,
                         'window_type': window_type, 'window_parameter': window_parameter})

    sorted_results = sorted(results, key=lambda configuration: configuration['f_score'], reverse=True)

    return sorted_results