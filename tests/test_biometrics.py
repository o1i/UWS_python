import unittest

from src.calculations.biometrics import *


class TestBiometrics(unittest.TestCase):

    def test_xy(self):
        self.assertAlmostEqual(xy(1), 1, 5)
        self.assertTrue(np.array_equal(xy(np.array([0, -1, 1000])), np.array([0, -1, 1000])))
        self.assertTrue(np.array_equal(xy(pd.Series([0, -1, 1000])), np.array([0, -1, 1000])))

    def test_yx(self):
        self.assertAlmostEqual(yx(1), -3, 5)
        self.assertTrue(np.array_equal(yx(np.array([0, -1, 1000])), np.array([-4, -5, 996])))
        self.assertTrue(np.array_equal(yx(pd.Series([0, -1, 1000])), np.array([-4, -5, 996])))

    def test_hx(self):
        self.assertAlmostEqual(hx(1), 0, 5)
        self.assertTrue(np.allclose(hx(np.array([0, 1000, 65])), np.array([0, 0, 0.7955])))
        self.assertTrue(np.allclose(hx(pd.Series([0, 1000, 65])), np.array([0, 0, 0.7955])))

    def test_hy(self):
        self.assertAlmostEqual(hy(1), 0.6, 5)
        self.assertTrue(np.allclose(hy(np.array([0, 1000, 65])), np.array([0.6, 0, 0.458333])))
        self.assertTrue(np.allclose(hy(pd.Series([0, 1000, 65])), np.array([0.6, 0, 0.458333])))

    def test_kx(self):
        self.assertAlmostEqual(kx(1), 10, 5)
        self.assertTrue(np.allclose(kx(np.array([0, 1000, 65])), np.array([10, 25,  10])))
        self.assertTrue(np.allclose(kx(pd.Series([0, 1000, 65])), np.array([10, 25,  10])))

    def test_ky(self):
        self.assertAlmostEqual(ky(1), 10, 5)
        self.assertTrue(np.allclose(ky(np.array([0, 1000, 65])), np.array([10, 25,  20])))
        self.assertTrue(np.allclose(ky(pd.Series([0, 1000, 65])), np.array([10, 25,  20])))

    def test_hkx(self):
        self.assertAlmostEqual(hkx(1), 0.84, 5)
        self.assertTrue(np.allclose(hkx(np.array([0, 1000, 65])), np.array([0.85, 0,  0.2])))
        self.assertTrue(np.allclose(hkx(pd.Series([0, 1000, 65])), np.array([0.85, 0,  0.2])))

    def test_hky(self):
        self.assertAlmostEqual(hky(1), 0.74, 5)
        self.assertTrue(np.allclose(hky(np.array([0, 1000, 65])), np.array([0.75, 0,  0.1])))
        self.assertTrue(np.allclose(hky(pd.Series([0, 1000, 65])), np.array([0.75, 0,  0.1])))
