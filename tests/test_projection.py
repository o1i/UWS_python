import unittest

import numpy as np
import pandas as pd

from src.calculations.mortality_tables import project_lc


class TestLcProjection(unittest.TestCase):

    def setUp(self) -> None:
        self.qx = pd.DataFrame(np.array([1, 0.0001, 0.8, 0.0001]).reshape(2, 2),
                               columns=[2019, 2020], index=[30, 31])
        self.lc = {
            "x": [30, 31],
            "y": [2019, 2020],
            "beta2": [1, 0.5],  # age 31 only gets half the reduction
            "kappa2": np.log(np.array([1, 0.5]))  # halving mx every year
        }

    def test_project_lc(self):
        projected = project_lc(self.qx, self.lc, factor=2, n_years=2)  # doubling the intensity of decrease
        self.assertAlmostEqual(projected.loc[30, 2021], 0.000025, 7)
        self.assertAlmostEqual(projected.loc[31, 2022], 0.000025, 7)
