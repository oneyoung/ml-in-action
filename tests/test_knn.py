import logging
import numpy
from kNN import classify, normalize
from .base import TestBase


class KNNTest(TestBase):
    @staticmethod
    def test_dataset():
        group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']
        x = numpy.array([1, 0.5])
        result = classify(x, group, labels, 3)
        logging.info(result)
