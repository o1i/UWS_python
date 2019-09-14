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
