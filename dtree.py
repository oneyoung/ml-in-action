# desicion tree
import math


def entroy(dataset):
    '''
    get Shannon Entroy of dataset
    higher Entroy means more chaos

    params:
        dataset: NxM array data, the last element of each row is label

    return Shannon Entroy
    '''
    total = len(dataset)
    counts = {}
    for vec in dataset:
        label = vec[-1]
        counts[label] = counts.get(label, 0) + 1
    ent = 0
    for label, count in counts.items():
        prob = float(count)/total
        ent -= prob*math.log(prob, 2)
    return ent


def split_dataset(dataset, index, value):
    '''
    split dataset
    filter out data vec with given index and value

    params:
        dataset: dataset to be processed
        index: index of vec to check
        value: target value to match

    return
    '''
    ret_data = []
    for vec in dataset:
        if vec[index] == value:
            # pop up value at index
            # then we can reduce complexity by 1 axes
            reduced_vec = vec[:index]
            reduced_vec.extend(vec[index+1:])
            ret_data.append(reduced_vec)
    return ret_data


def best_feature_split(dataset):
    '''
    choose best feature to split

    params:
        dataset: input dataset, the last member of vec is label

    return: the index of best feature for split in vec
    '''
    feature_num = len(dataset[0]) - 1  # exclude label
    base_entroy = entroy(dataset)
    best_index = -1
    best_info_gain = 0
    for i in range(feature_num):
        # all possible values at index i
        values = set([vec[i] for vec in dataset])
        ientroy = 0  # total entroy of this split
        for v in values:
            sub_set = split_dataset(dataset, i, v)
            # the length of sub_set is how many time value=v appears
            # so probility is len(sub_set)/ (total samples)
            prob = len(sub_set)/float(len(dataset))
            ientroy += entroy(sub_set)
        info_gain = base_entroy - ientroy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_index = i
    return best_index
