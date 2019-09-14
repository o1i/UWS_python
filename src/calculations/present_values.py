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
