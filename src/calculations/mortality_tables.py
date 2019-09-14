import numpy as np
import pandas as pd
from scipy.stats import linregress


def project_lc(qx: pd.DataFrame, lc: dict, factor: float = 1.0, n_years: int = 50) -> pd.DataFrame:
    """
    Projects a life table according to the lee-carter model for n_years years

    Args:
        qx: life table. int-index on ages (rows) and year columns (int)
        lc: dict with the parameters of the lee-carter model
        factor: multiplicative factor for mortality improvement
        n_years: number of years to project

    Returns:
        qx extended by n_years columns with projected probabilities of mortality

    """
    assert(qx.shape[0] == len(lc["x"]))
    lastyear = qx.columns[-1]
    mx = qx2mx(qx[lastyear].values).reshape([-1, 1])
    slope = linregress(np.array(lc["y"]), lc["kappa2"]).slope * factor
    add = mx2qx(mx * np.exp(np.matmul(np.maximum(0, np.array(lc["beta2"]).reshape([-1, 1])),
                                      np.arange(1, n_years + 1).reshape([1, -1]) * slope)))
    add_df = pd.DataFrame(add, columns=lastyear + np.arange(1, n_years + 1), index=qx.index)
    return pd.concat([qx, add_df], axis=1)


def per2gen(qx: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a period table to a generational life table. Only takes the
    complete diagonal from the period table. Mortalities of generations that
    appear in the period table that are to young or old to have all ages in the
    table are discarded.

    Args:
        qx: period life table (age on rows, years on columns

    Returns:
        Generational life tables for the generations for which all ages are observed
    """
    assert(qx.shape[1] > qx.shape[0])
    n_gen = qx.shape[1] - qx.shape[0] + 1
    vals = np.stack([np.diagonal(qx, i) for i in range(0, n_gen)], axis=1)
    gens = qx.columns[0] - qx.index[0] + np.arange(0, n_gen)
    return pd.DataFrame(vals, index=qx.index, columns=gens)


def mx2qx(mx: np.ndarray) -> np.ndarray:
    """Converts force of mortality to probability of death"""
    return 1-np.exp(-mx)


def qx2mx(qx: np.ndarray) -> np.ndarray:
    """Converts probabilities of death to forces of mortality"""
    return -np.log(1-qx)