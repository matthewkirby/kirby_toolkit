import numpy as np


def jackknife_array(arr):
    """Return the remove 1 jackknife mean and variance

    Parameters
    ----------
    arr : array_like
        The list of data to be jackknifed

    Returns
    -------
    jkvar : float
        The jackknifed variance
    """
    narr = len(arr)

    xbar_i = np.array([np.mean(np.delete(arr, i)) for i in range(narr)])
    xbar = np.mean(xbar_i)
    jkvar = (narr-1)*np.mean((xbar_i-xbar)*(xbar_i-xbar))

    return jkvar

