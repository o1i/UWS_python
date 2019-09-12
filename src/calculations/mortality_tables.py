import numpy as np
import pandas as pd
from scipy.stats import linregress


def project_lc(qx, lc, factor=1, n_years=50):
    assert(qx.shape[0] == len(lc["x"]))
    lastyear = qx.columns[-1]
    mx = qx2mx(qx[lastyear].values).reshape([-1, 1])
    slope = linregress(np.array(lc["y"]), lc["kappa2"]).slope * factor
    add = mx2qx(mx * np.exp(np.matmul(np.maximum(0, np.array(lc["beta2"]).reshape([-1, 1])),
                                      np.arange(1, n_years + 1).reshape([1, -1]) * slope)))
    add_df = pd.DataFrame(add, columns=lastyear + np.arange(1, n_years + 1), index=qx.index)
    return pd.concat([qx, add_df], axis=1)


def mx2qx(mx: np.ndarray) -> np.ndarray:
    """Converts force of mortality to probability of death"""
    return 1-np.exp(-mx)


def qx2mx(qx: np.ndarray) -> np.ndarray:
    """Converts probabilities of death to forces of mortality"""
    return -np.log(1-qx)