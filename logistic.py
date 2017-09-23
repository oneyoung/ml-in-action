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
    return weights
