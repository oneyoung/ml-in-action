import numpy


def classify(x, data, labels, k):
    '''
    kNN: k Nearest Neighbors

    Input:
        x: vector to compare to existing dataset (1xN)
        data: size m data set of known vectors (NxM)
        labels: 'data' set labels (1xM vector)
        k: number of neighbors to use for comparison (should be an odd number)

    Output:     the most popular class label
    '''
    m = data.shape[0]  # size of data: M
    diff = numpy.tile(x, (m, 1)) - data  # extend 'x' to 'N*M', and get diff
    dists = ((diff**2).sum(axis=1))**0.5  # Euler formula for distance
    # argsort, index[0] is min num index, index[-1] is max num index
    index = dists.argsort()
    weight_map = {}
    for i in range(k):
        label = labels[index[i]]  # get Nth nearest label
        weight_map[label] = weight_map.get(label, 0) + 1
    # return the label with highest weight
    max_key = None
    for key in weight_map:
        if weight_map.get(max_key, 0) < weight_map[key]:
            max_key = key
    return max_key


def normalize(data):
    '''
    normalize numpy array data to range [0, 1]

    Input:
        data: size m data set of known vectors (NxM)
    Return:
        normalized data (NxM)
    '''
    maxs = data.max(0)
    mins = data.min(0)
    ranges = maxs - mins
    norm = numpy.zeros(data.shape)
    m = data.shape[0]  # M size
    norm = data - numpy.tile(mins, (m, 1))
    norm = norm/numpy.tile(ranges, (m, 1))
    return norm
