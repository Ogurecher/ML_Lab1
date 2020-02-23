import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from find_best_parameters import find_best_parameters_one_hot
from classification import classify_object
from prepare_data import prepare_data, one_hot_encode
import parameters

data_csv = './data/glass.csv'

dataset, features, labels, num_classes = prepare_data(data_csv)
encoded_labels = one_hot_encode(labels)

#print(find_best_parameters_one_hot(features, labels, encoded_labels, num_classes))

f_scores = []

for window_parameter in parameters.window_neighbours:
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
    f_scores.append({'f_score': f_score, 'window_parameter': window_parameter})

print(f_scores)

plt.plot([point['window_parameter'] for point in f_scores],
         [point['f_score'] for point in f_scores])
plt.xlabel('nearest neighbours')
plt.ylabel('f-score')
plt.show()
