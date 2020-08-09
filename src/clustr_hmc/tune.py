
import random
import numpy as np


def aggregate_by_leaf_node(data):
    '''
    Creates a dictionary of data indices for every leaf node
    :param data: Input data in format [(sku title, [[labels]])]
    :return: dictionary
    '''
    leaf_nodes_to_index = dict()
    for i, x in enumerate(data):
        for y in x[1]:
            if y[-1] in leaf_nodes_to_index:
                leaf_nodes_to_index[y[-1]].append(i)
            else:
                leaf_nodes_to_index[y[-1]] = [i]
    return leaf_nodes_to_index


def sample_leaf_nodes(leaf_to_index, train_size=0.8):
    '''
    Samples each leaf node separately using the train_size
    :param leaf_to_index: the dictionary with keys as leaf nodes and list of indices
    :param train_size: fraction of train
    :return: train and test indices
    '''
    train = []
    complete = []
    for k,v in leaf_to_index.items():
        ids = random.sample(v, int(np.floor(len(v)*train_size)))
        train.extend(ids)
        complete.extend(v)
    train = list(set(train))
    test = list(set(complete) - set(train))
    return train,test


def generate_train_test(data):
    '''
    creates train and test data
    :param data: input data
    :return: train data, test data
    '''
    label_index_dict = aggregate_by_leaf_node(data)
    train_id, test_id = sample_leaf_nodes(label_index_dict)
    train_data = [data[i] for i in train_id]
    test_data = [data[i] for i in test_id]
    return train_data, test_data

