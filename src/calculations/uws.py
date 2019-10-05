import logging
from typing import Callable

import pandas as pd

from src.calculations.biometrics import xy, yx, hx, hy
from src.calculations.present_values import äx, äk, äxw


logger = logging.getLogger(__name__)


def name_logger(func: Callable):
    def wrapped_func(*args, **kwargs):
        logger.debug("Running " + func.__name__)
        func(*args, **kwargs)
    return wrapped_func


class Uws:
    def __init__(self, qx: pd.DataFrame, qy: pd.DataFrame, i: pd.Series, km: dict, plan: dict):
        self._qx = qx
        self._qy = qy
        self._i = i
        self._km = km
        self._plan = plan

        self.calc_äx()
        self.calc_äy()
        self.calc_kx()
        self.calc_ky()
        self.calc_äxw()
        self.calc_äyw()
        self.calc_uws_x()
        self.calc_uws_y()

    def calc_äx(self):
        self._äx = äx(self._qx, self._i, self._km["m"], self._km["k"])

    def calc_äy(self):
        self._äy = äx(self._qy, self._i, self._km["m"], self._km["k"])

    def calc_kx(self):
        self._kx = äk(self._qx, self._i, self._km["k"])

    def calc_ky(self):
        self._ky = äk(self._qy, self._i, self._km["k"])

    def calc_äxw(self):
        self._äxw = äxw(self._qx, self.äy, self._i, yx, hx, self._km["k"])

    def calc_äyw(self):
        self._äyw = äxw(self._qy, self.äx, self._i, xy, hy, self._km["k"])

    def calc_uws_x(self):
        self._uws_x = 1 / (self._äx + self._äxw * self.plan["wx"] + self._kx * self.plan["kx"])

    def calc_uws_y(self):
        self._uws_y = 1 / (self._äy + self._äyw * self.plan["wy"] + self._ky * self.plan["ky"])

    @property
    def äxw(self):
        return self._äxw

    @äxw.setter
    @name_logger
    def äxw(self, value: pd.DataFrame):
        self._äxw = value
        self.calc_uws_x()

    @property
    def äyw(self):
        return self._äyw

    @äyw.setter
    @name_logger
    def äyw(self, value: pd.DataFrame):
        self._äyw = value
        self.calc_uws_y()

    @property
    def kx(self):
        return self._kx

    @kx.setter
    @name_logger
    def kx(self, value: pd.DataFrame):
        self._kx = value

    @property
    def ky(self):
        return self._ky

    @ky.setter
    @name_logger
    def ky(self, value: pd.DataFrame):
        self._ky = value

    @property
    def äx(self):
        return self._äx

    @äx.setter
    @name_logger
    def äx(self, value: pd.DataFrame):
        self._äx = value
        self.calc_äyw()

    @property
    def äy(self):
        return self._äy

    @äy.setter
    @name_logger
    def äy(self, value: pd.DataFrame):
        self._äy = value
        self.calc_äxw()

    @property
    def qx(self):
        return self._qx

    @qx.setter
    @name_logger
    def qx(self, value: pd.DataFrame):
        self._qx = value
        self.calc_äx()

    @property
    def qy(self):
        return self._qy

    @qy.setter
    @name_logger
    def qy(self, value: pd.DataFrame):
        self._qy = value
        self.calc_äy()

    @property
    def i(self):
        return self._i

    @i.setter
    @name_logger
    def i(self, value: pd.Series):
        self._i = value
        self.calc_äx()
        self.calc_äy()
        self.calc_kx()
        self.calc_ky()

    @property
    def plan(self):
        return self._plan

    @plan.setter
    def plan(self, value: dict):
        self._plan = value

    @property
    def uws_x(self):
        return self._uws_x

    @property
    def uws_y(self):
        return self._uws_y



