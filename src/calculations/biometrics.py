"""
Biometric functions
Sources:
https://www.actuaries.ch/fr/downloads/aid!b4ae4834-66cd-464b-bd27-1497194efc96/id!101/Koller_LV_2013.pdf
https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/publikationen.assetdetail.350506.html
https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/publikationen.assetdetail.2347880.html
"""

from typing import Union

import numpy as np
import pandas as pd


def xy(x: Union[int, np.ndarray, pd.Series, pd.Index]) -> Union[int, np.ndarray, pd.Series, pd.Index]:
    """
    Age of widower at the point of death of the wife

    Formula of KT95 was not available. Koller script suggests it's close to identity after 60


    Args:
        x(np.ndarray): Age of the deceased wife

    Returns:
        np.ndarray: Age of the widower

    """
    return x


def yx(x: Union[int, np.ndarray, pd.Series, pd.Index]) -> Union[int, np.ndarray, pd.Series, pd.Index]:
    """
    As xy, but with reversed roles

    Args:
        x(np.ndarray): Age of the husband at time of death

    Returns:
        np.ndarray: Age of widow
    """
    return x - 4


def hx(x: Union[int, np.ndarray, pd.Series, pd.Index]) -> Union[float, np.ndarray, pd.Series, pd.Index]:
    """
    Probability of leaving a widow eligible for a widow's annuity if a man dies at age y.

    Curve is chosen to resemble Koller p.90 (above the age of 60 below which the probability of dying is small)

    Args:
        x(np.ndarray): Numeric vector with the age of the man who dies.

    Returns:
        (np.ndarray) The probability of leaving behind a widow eligible for a pension.
    """
    return np.maximum(0, np.minimum(0.85, (x - 102) * (x - 22) * -5e-4))


def hy(x: Union[int, np.ndarray, pd.Series, pd.Index]) -> Union[float, np.ndarray, pd.Series, pd.Index]:
    """
    Probability of leaving a widower eligible for a widower's annuity if a woman dies at age y.

    Curve is chosen to resemble Koller p.90 (above the age of 60 below which the probability of dying is small)

    Args:
        x(int): Numeric vector with the age of the woman who dies.

    Returns:
        np.ndarray: The probability of leaving behind a widower eligible for a pension.
    """
    return np.maximum(0, np.minimum(0.6, (-0.55/30) * x + 1.65))


def kx(x:  Union[int, np.ndarray, pd.Series, pd.Index]) -> Union[int, np.ndarray, pd.Series, pd.Index]:
    """
    Average age of a child getting orphan benefits, given that the father's death generates an orphan.

    Args:
        x(np.ndarray): Numeric vector with the age of death of the father

    Returns:
        (np.ndarray): The average age of the orphan
    """
    return np.maximum(10, np.minimum(25, 10 + (x-65)))


def ky(x: Union[int, np.ndarray, pd.Series, pd.Index]) -> Union[int, np.ndarray, pd.Series, pd.Index]:
    """
    Average age of a child getting orphan benefits, given that the mother's death generates an orphan.

    Args:
        x(np.ndarray): Numeric vector with the age of death of the mother

    Returns:
        (np.ndarray): The average age of the orphan
    """
    return np.maximum(10, np.minimum(25, 20 + (x-65)))


def hkx(x: Union[int, np.ndarray, pd.Series, pd.Index]) -> Union[float, np.ndarray, pd.Series, pd.Index]:
    """
    Expected number of orphans for a man, given there are any

    Args:
        x(np.ndarray): Numeric vector with the age of death of the father

    Returns:
        (np.ndarray): The average number of orphans
    """
    return np.maximum(0, (85 - x) / 100)


def hky(x: Union[int, np.ndarray, pd.Series, pd.Index]) -> Union[float, np.ndarray, pd.Series, pd.Index]:
    """
    Expected number of orphans for a woman, given there are any

    Args:
        x(np.ndarray): Numeric vector with the age of death of the mother

    Returns:
        (np.ndarray): The average number of orphans
    """
    return np.maximum(0, (75 - x) / 100)
