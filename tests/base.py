import os
import logging
from unittest import TestCase


logging.basicConfig(format='%(filename)s-%(funcName)s: %(message)s ',
                    level=logging.INFO)


class TestBase(TestCase):
    @staticmethod
    def get_file(path):
        files_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 'files'))
        return os.path.join(files_dir, path)
