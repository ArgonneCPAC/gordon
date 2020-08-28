"""Weighted histograms using triweight kernels."""
from numba import vectorize


@vectorize
def tw_kern(x, m, h):
    """Triweight kernel.

    Parameters
    ----------
    x : array-like or scalar
        The value at which to evaluate the kernel.

    m : array-like or scalar
        The mean of the kernel.

    h : array-like or scalar
        The approximate 1-sigma width of the kernel.

    Returns
    -------
    kern : array-like or scalar
        The value of the kernel.

    """
    z = (x - m) / h
    if z < -3 or z > 3:
        return 0
    else:
        return 35 / 96 * (1 - (z / 3) ** 2) ** 3 / h


@vectorize
def tw_cuml_kern(x, m, h):
    """CDF of the triweight kernel.

    Parameters
    ----------
    x : array-like or scalar
        The value at which to evaluate the kernel.

    m : array-like or scalar
        The mean of the kernel.

    h : array-like or scalar
        The approximate 1-sigma width of the kernel.

    Returns
    -------
    kern_cdf : array-like or scalar
        The value of the kernel CDF.

    """
    y = (x - m) / h
    if y < -3:
        return 0
    elif y > 3:
        return 1
    else:
        val = (
            -5 * y ** 7 / 69984
            + 7 * y ** 5 / 2592
            - 35 * y ** 3 / 864
            + 35 * y / 96
            + 1 / 2
        )
        return val


@vectorize
def tw_bin_kern(m, h, L, H):
    """Integrated bin weight for the triweight kernel.

    Parameters
    ----------
    m : array-like or scalar
        The value at which to evaluate the kernel.

    h : array-like or scalar
        The approximate 1-sigma width of the kernel.

    L : array-like or scalar
        The lower bin limit.

    H : array-like or scalar
        The upper bin limit.

    Returns
    -------
    bin_prob : array-like or scalar
        The value of the kernel integrated over the bin.

    """
    return tw_cuml_kern(H, m, h) - tw_cuml_kern(L, m, h)
