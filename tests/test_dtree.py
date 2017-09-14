import logging
from .base import TestBase
import dtree


class DtreeTest(TestBase):
    def test_entroy(self):
        dataset = [
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']
        ]
        entroy = dtree.entroy(dataset)
        logging.info(entroy)
        self.assertTrue(0.97 < entroy < 0.98)
