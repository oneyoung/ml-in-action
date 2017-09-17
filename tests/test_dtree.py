import logging
from .base import TestBase
import dtree


def create_dataset():
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    return dataset


class DtreeTest(TestBase):
    def test_entroy(self):
        dataset = create_dataset()
        entroy = dtree.entroy(dataset)
        logging.info(entroy)
        self.assertTrue(0.97 < entroy < 0.98)

    def test_best_feature_split(self):
        dataset = create_dataset()
        index = dtree.best_feature_split(dataset)
        self.assertEqual(index, 0)
