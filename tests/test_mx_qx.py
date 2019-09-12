import unittest

import numpy as np

from src.calculations.mortality_tables import qx2mx, mx2qx


class TestQx2Mx(unittest.TestCase):
    def test_small_remains_small(self):
        self.assertTrue(all(np.equal(qx2mx(np.array([0])), np.array([0]))))

    def test_large_remains_large(self):
        self.assertTrue(qx2mx(np.array([0.99999]))[0] > 10)


class TestMx2Qx(unittest.TestCase):
    def test_small_remains_small(self):
        self.assertTrue(all(np.equal(mx2qx(np.array([0])), np.array([0]))))

    def test_large_remains_large(self):
        self.assertTrue(mx2qx(np.array([99]))[0] > 0.999)
