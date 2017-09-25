import os
import logging
import numpy
import logistic
from .base import TestBase


class LogisticTest(TestBase):
    def test_grad_ascent(self):
        data, labels = self.load_data('testSet.txt')
        result = logistic.grad_ascent(data, labels)
        logging.info(result)
        self.assertEqual(len(result), 3)
        self.plot(data, labels, result)

    def test_stoc_grad_ascent0(self):
        data, labels = self.load_data('testSet.txt')
        result = logistic.stoc_grad_ascent0(data, labels)
        logging.info(result)
        self.plot(data, labels, result)

    def test_stoc_grad_ascent(self):
        data, labels = self.load_data('testSet.txt')
        result = logistic.stoc_grad_ascent(data, labels)
        logging.info(result)
        self.plot(data, labels, result)

    def load_data(self, fname):
        folder = self.get_file('logistic')
        fpath = os.path.join(folder, fname)
        data = []
        labels = []
        with open(fpath) as fp:
            for l in fp.xreadlines():
                parts = l.strip().split()
                # we extend a x0 here
                data.append([1.0, float(parts[0]), float(parts[1])])
                labels.append(int(parts[2]))
        return data, labels

    def plot(self, dataMat, labelMat, weights):  # noqa
        import matplotlib.pyplot as plt
        dataArr = numpy.array(dataMat)
        n = numpy.shape(dataArr)[0]
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in range(n):
            if int(labelMat[i]) == 1:
                xcord1.append(dataArr[i, 1])
                ycord1.append(dataArr[i, 2])
            else:
                xcord2.append(dataArr[i, 1])
                ycord2.append(dataArr[i, 2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='green')
        x = numpy.arange(-3.0, 3.0, 0.1)
        y = (-weights[0]-weights[1]*x)/weights[2]
        ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
