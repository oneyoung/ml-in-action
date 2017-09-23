import os
import logging
from .base import TestBase
import logistic


class LogisticTest(TestBase):
    def test_grad_ascent(self):
        data, labels = self.load_data('testSet.txt')
        result = logistic.grad_ascent(data, labels)
        logging.info(result)
        self.assertEqual(len(result), 3)

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
