import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from find_best_parameters import find_best_parameters
from classification import classify_object
from prepare_data import prepare_data
import parameters

data_csv = './data/glass.csv'

dataset, features, labels, num_classes = prepare_data(data_csv)

#print(find_best_parameters(features, labels, num_classes))

f_scores = []

for window_parameter in parameters.window_neighbours:
    predicted_labels = []

    for index in range(len(features)):
        predicted_value = classify_object(features[np.arange(len(features)) != index],
                                          labels[np.arange(len(labels)) != index], features[index],
                                          parameters.best_combination['distance_function'],
                                          parameters.best_combination['kernel_function'],
                                          parameters.best_combination['window_type'],
                                          window_parameter)

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
