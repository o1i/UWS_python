import numpy as np
import pandas as pd

import unittest

from src.calculations.present_values import make_v


class TestMakeV(unittest.TestCase):

    def setUp(self) -> None:
        self.qx = pd.DataFrame(np.zeros([4, 5]), index=[30, 31, 32, 33], columns=list(range(1980, 1985)))
        self.i = pd.Series(np.arange(0, 0.9, step=0.1), index=range(2010, 2019))

    def test_correct_numbers(self):
        v_mat = make_v(self.i, self.qx)
        self.assertEqual(v_mat.shape, self.qx.shape)
        self.assertEqual(v_mat.loc[30, 1980], 1 / (1 + self.i[2010]))
        self.assertEqual(v_mat.loc[32, 1984], 1 / (1 + self.i[2016]))
