import numpy as np
import pandas as pd

import unittest

from src.calculations.present_values import make_v


class TestPer2Gen(unittest.TestCase):

    def setUp(self) -> None:
        self.qx = pd.DataFrame(np.zeros([4, 5]), index=[30, 31, 32, 33], columns=list(range(2020, 2025)))
        self.i = pd.Series(np.arange(0, 1, step=0.1), index=range(2020, 2030))

    def test_correct_numbers(self):
        v_mat = make_v(self.i, self.qx)
        self.assertEqual(v_mat.shape, self.qx.shape)
        self.assertEqual(v_mat.iloc[0, 0], 1)
        self.assertEqual(v_mat.iloc[3, 4], 1 / 1.4)

