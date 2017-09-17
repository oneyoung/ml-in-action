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
    labels = ['no surfacing', 'flippers']
    return dataset, labels


class DtreeTest(TestBase):
    def test_entroy(self):
        dataset, labels = create_dataset()
        entroy = dtree.entroy(dataset)
        logging.info(entroy)
        self.assertTrue(0.97 < entroy < 0.98)

    def test_best_feature_split(self):
        dataset, labels = create_dataset()
        index = dtree.best_feature_split(dataset)
        self.assertEqual(index, 0)

    def test_create_tree(self):
        dataset, labels = create_dataset()
        tree = dtree.create_tree(dataset, labels)
        logging.info(tree)
        expected = {'no surfacing':
                    {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        self.assertDictEqual(tree, expected)

    def test_classify(self):
        fpath = self.get_file('dtree/lenses.txt')
        dataset = []
        with open(fpath) as fp:
            for l in fp.readlines():
                dataset.append(l.strip().split('\t'))
        labels = ['age', 'prescript', 'astigmatic', 'tearRate']
        tree = dtree.create_tree(dataset, labels)
        logging.info(tree)
        test_vec = ['pre', 'myope', 'no', 'normal']
        result = dtree.classify(tree, labels, test_vec)
        logging.info('result: ' + result)
        self.assertEquals(result, 'soft')
