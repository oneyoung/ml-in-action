import logging
import itertools
import numpy
from kNN import classify, normalize
from .base import TestBase


class KNNTest(TestBase):
    @staticmethod
    def test_dataset():
        group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']
        x = numpy.array([1, 0.5])
        result = classify(x, normalize(group), labels, 3)
        logging.info(result)

    def test_dating_testset(self):
        for i in range(2, 20):
            ret = self.do_txt_training('knn/datingTestSet.txt', i)
            logging.info('k = %d, Error Rate: %f', i, ret)

    def test_dating_testset2(self):
        for i in range(2, 20):
            ret = self.do_txt_training('knn/datingTestSet2.txt', i)
            logging.info('k = %d, Error Rate: %f', i, ret)

    def do_txt_training(self, txt, k):
        dataset, labels = self.txt2dataset(txt)
        dataset = normalize(dataset)
        # 90% for training, 10% for verify
        index = int(0.9*len(labels))
        training_set = dataset[:index]
        training_lables = labels[:index]
        ref_set = dataset[index:]
        ref_labels = labels[index:]
        # start testing
        errno = 0
        for (x, label) in itertools.izip(ref_set, ref_labels):
            result = classify(x, training_set, training_lables, k)
            msg = 'Data: %s, label: %s, result: %s' % (x, label, result)
            logging.debug(msg)
            if result != label:
                errno += 1
        return float(errno)/len(ref_labels)

    def txt2dataset(self, fname):
        src = self.get_file(fname)
        fp = open(src)
        labels = []
        dataset = []
        for l in fp.read().splitlines():
            parts = l.split()
            dataset.append([float(p) for p in parts[:3]])
            labels.append(parts[-1])
        return numpy.array(dataset), labels
