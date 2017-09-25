import numpy


def sigmod(x):
    return 1.0/(1 + numpy.exp(-x))


def grad_ascent(data, labels):
    '''
    grad ascent

    params:
        data - input NxM data matrix (N - row, M - column)
        labels - N array of label, should be values like int/float
    '''
    # conver input into matrix
    data_mat = numpy.mat(data)
    label_mat = numpy.mat(labels).transpose()  # Nx1 matrix
    n, m = data_mat.shape

    alpha = 0.001  # 1 step length
    max_cycle = 500  # max iter cycle
    weights = numpy.ones((m, 1))  # Mx1 matrix of 1
    for _ in range(max_cycle):
        # sigmod(z)
        # z = w0*x0 + w1*x1 + w2*x2 ... + wM*xM
        # matrix dot mutiple
        h = sigmod(data_mat.dot(weights))  # Nx1 sample of z, then do sigmod
        err = label_mat - h  # in ideal situation, should be Nx1 zeros
        # grad ascent alg: w := w + alpha * delta(f(w))
        weights = weights + alpha*data_mat.transpose().dot(err)
    return numpy.array(weights)


def stoc_grad_ascent0(data_input, labels):
    '''
    stochastic gradient ascent
    '''
    data = numpy.array(data_input)
    n, m = numpy.shape(data)
    alpha = 0.01
    weights = numpy.ones(m)
    for i in xrange(n):
        h = sigmod(sum([data[i]*weights]))  # just value, not matrix
        err = labels[i] - h
        weights = weights + alpha*data[i]*err
    return weights


def stoc_grad_ascent(data_input, labels, iter=150):
    '''
    stochastic gradient ascent, improved version
    '''
    data = numpy.array(data_input)
    n, m = numpy.shape(data)
    alpha = 0.01
    weights = numpy.ones(m)
    for j in xrange(iter):  # iterator mutiple loops for better result
        data_indexes = range(n)  # array of indexes to data
        for i in range(n):
            alpha = 4/(1.0 + i + j) + 0.01  # use dymanic alpha
            # random pick an sample from data
            rand_index = int(numpy.random.uniform(0, len(data_indexes)))
            h = sigmod(sum([data[rand_index]*weights]))
            err = labels[rand_index] - h
            weights = weights + alpha*err*data[rand_index]
            del(data_indexes[rand_index])  # avoid use this val in next loop
    return weights
