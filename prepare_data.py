import pandas as pd
import numpy as np
from sklearn import preprocessing


def normalize(dataset):
    return preprocessing.normalize(dataset, axis=0)

def prepare_data(filename):
    dataframe = pd.read_csv(filename)

    class_mapping = {name: index for index, name in enumerate(sorted(list(set(dataframe.Type))))}
    num_classes = len(set(dataframe.Type))

    dataframe = dataframe.replace({'Type': class_mapping})

    np_dataset = np.array(dataframe)
    normalized_features = normalize(np_dataset[:, :-1])
    labels = np_dataset[:, -1]

    return np_dataset, normalized_features, labels, num_classes

def one_hot_encode(labels):
    pd_labels = pd.Series(labels)
    
    return np.array(pd.get_dummies(pd_labels))
