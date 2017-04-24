"""
Saurabh Mahajan
sm6921

Testing for various functions of parallel_sorter
"""


import unittest
import numpy as np
import parallel_sorter

class Tests(unittest.TestCase):

    def setUp(self):
        pass

    def test_generate_nums(self):
        self.assertEqual(len(parallel_sorter.generate_numbers(500)), 500)

    def test_split_data(self):
        self.assertEqual(parallel_sorter.split_data([5, 8, 3, 1, 4, 7], 3), [[1], [5, 3, 4], [8, 7]])

    def test_create_sort(self):
        self.assertEqual(len(parallel_sorter.parallel_sort(100)), 100)

if __name__ == '__main__':
    unittest.main()
