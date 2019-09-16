from typing import Callable

import numpy as np
import pandas as pd


def make_v(i: pd.Series, qx: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the discount matrix for the years appearing as columns in qx

    Args:
        i: a series with years as index, covering all years appearing in qx
        qx: a data frame with years as columns

    Returns:
        A data frame the same size as qx with the (annual) discount numbers
    """
    assert all(qx.columns.isin(i.index)), "interest rate time range must cover life table time range"
    v = 1 / (1 + i[qx.columns])
    v_mat = pd.concat([v] * qx.shape[0], axis=1).transpose()
    v_mat.index = qx.index
    return v_mat


def äx(qx: pd.DataFrame, i: pd.Series, m: int = 4, k: float = 0.02) -> pd.DataFrame:
    """
    Present value of annuity until death, given a mortality table and interest rate curve.

    Args:
        qx: (period) life table
        i: interest rate series (annual values)
        m: number of yearly payments of the annuity
        k: cost factor

    Returns:

    """
    out = qx * 0
    v_mat = make_v(i, qx)
    last_age = qx.index[-1]
    out.loc[last_age] = 1 + v_mat.loc[last_age] * (1 - qx.loc[last_age]) * 1  # Last age initialisation
    for age in qx.index[-2::-1]:  # Iterate over other ages
        out.loc[age] = 1 + v_mat.loc[age] * (1 - qx.loc[age]) * out.loc[age + 1]
    return (out - (m - 1) / (2 * m)) * (1 + k)  # consider costs and payment frequency


def äk(qx: pd.DataFrame, i: pd.Series, k: float = 0.02):
    """
    Present value of an orphan's pension, assuming no mortality of the orphan.
    kx and zx are extrapolations from the numbers given in koller.

    Args:
        qx: life table to be used
        i: fixed interest rate
        k: costs to be applied

    Returns:
        value of the orphan's pensions

    """
    def kx(x):
        """Number of orphans for deaths at age x"""
        return np.minimum(0.24, np.maximum(0, np.exp(-(x - 60)/4) * 0.24))

    def zx(x):
        """Age of orphans for death at age x"""
        return np.maximum(17.4, np.minimum(25, (x-60)/2 + 17.4))

    def äz(z, i):
        """PV of orphans pension at age z, given interest rate i"""
        # Positive rates
        out = -(np.power(1 + i, z - 24) - 1) / np.log(1 + i)
        # zero rates
        out.loc[i == 0] = 24 - z
        return out

    out = qx * 0
    v_mat = make_v(i, qx)
    max_age = qx.index[-1]
    out.loc[max_age] = qx.loc[max_age] * np.power(v_mat.loc[max_age], 0.5) * kx(max_age) * \
                       äz(zx(max_age), 1/v_mat.loc[max_age])
    for age in qx.index[:-1]:
        out.loc[age, ] = qx.loc[age] * np.power(v_mat.loc[age], 0.5) * kx(age) * \
                         äz(zx(age), 1 / v_mat.loc[age, ] - 1) + \
                         (1 - qx.loc[age]) * v_mat.loc[age] * out.loc[age + 1]
    out = out * (1 + k)
    return out


def äxw(qx: pd.DataFrame, äyw: pd.DataFrame, i: pd.Series, yx: Callable[[int], int],
        hx: Callable[[int], int], m: int = 4, k: float = 0.02):
    """
    PV of widows/widowers pensions

    Args:
        qx: Generational life table of the person whose life is insured
        äyw: PV of annuity of the beneficiary
        i: interest rates
        yx: function calculating the average age of the spouse for deaths at age x
        hx: function calculating probability of being married at death at age x
        m: under year frequency of payment
        k: costs

    Returns:

    """
    out = qx * 0
    v_mat = make_v(i, qx)
    omega = qx.index[-1]

    def get_d(x: pd.Index) -> pd.Index:
        return yx(x) - x

    def get_indices(i: int, d: int) -> (np.ndarray, np.ndarray):
        """Get the dislocation of a row by diagonally shifting it top-right or bottom-left (move in generation table)"""
        rows = (np.ones([qx.shape[1]]) * np.maximum(0, np.minimum(qx.shape[0], i + d))).astype(int)
        cols = np.maximum(0, np.minimum(qx.shape[0] - 1, np.arange(0, qx.shape[1]) - d))
        return rows, cols

    out.loc[omega] = qx.loc[omega] * v_mat.loc[omega] * hx(omega) * äyw.values[get_indices(qx.index.get_loc(omega),
                                                                                           get_d(omega))]
    for age in qx.index[-2::-1]:
        out.loc[age] = (1 - qx.loc[age]) * v_mat.loc[age] * out.loc[age + 1] + \
                       qx.loc[age] * v_mat.loc[age] * hx(age) * äyw.values[get_indices(qx.index.get_loc(age),
                                                                                       get_d(age))]

    out = (out - (m - 1) / (2 * m)) * (1 + k)
    return out
