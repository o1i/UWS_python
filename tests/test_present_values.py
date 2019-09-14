import unittest

import numpy as np
import pandas as pd

from src.calculations.present_values import äx


class TestPresentValues(unittest.TestCase):

    def setUp(self) -> None:
        self.qx_zero = pd.DataFrame(np.zeros([3, 3]), columns=[2019, 2020, 2021], index=[30, 31, 32])
        self.qx_ones = pd.DataFrame(np.ones([3, 3]), columns=[2019, 2020, 2021], index=[30, 31, 32])
        self.i_zero = pd.Series(np.zeros([3]), index=[2019, 2020, 2021])
        self.i_ones = pd.Series(np.ones([3]), index=[2019, 2020, 2021])

    def test_zero_interest(self):
        values = äx(self.qx_zero, self.i_zero, k=0, m=1)
        self.assertEqual(values.iloc[2, 0], 2)
        self.assertEqual(values.iloc[0, 0], 4)
        values = äx(self.qx_ones, self.i_zero, k=0, m=1)
        self.assertEqual(values.iloc[2, 0], 1)
        self.assertEqual(values.iloc[0, 0], 1)

    def test_interest_one(self):
        values = äx(self.qx_zero, self.i_ones, k=0, m=1)
        self.assertEqual(values.iloc[2, 0], 1.5)
        values = äx(self.qx_ones, self.i_ones, k=0, m=1)
        self.assertEqual(values.iloc[2, 0], 1)
        self.assertEqual(values.iloc[0, 0], 1)