# desicion tree
import math


def entroy(dataset):
    '''
    get Shannon Entroy of dataset

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
