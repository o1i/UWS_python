import numpy as np
import pandas as pd

import unittest

from src.calculations.mortality_tables import per2gen


class TestPer2Gen(unittest.TestCase):

    def setUp(self) -> None:
        self.qx = pd.DataFrame(np.arange(0, 20).reshape(4, 5), index=[30, 31, 32, 33], columns=list(range(2020, 2025)))

    def test_per2gen(self):
        self.assertTrue(per2gen(self.qx).equals(pd.DataFrame(np.array([[0, 1], [6, 7], [12, 13], [18, 19]]),
                                                             index=[30, 31, 32, 33], columns=[1990, 1991])))
