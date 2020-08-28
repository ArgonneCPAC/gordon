"""Functions calculating various weighted histograms.

A Gaussian PDF is dropped onto each data point.

The gaussian_weighted_histogram function adds up the
total probability that each Gaussian contributes to each bin.

The gaussian_weighted_profile function adds up the value of some other quantity
that each Gaussian-weighted point contributes to each bin.

"""
import numpy as np
from math import erf as math_erf
from math import sqrt as math_sqrt
from numba import njit
from .triweight_kernels import tw_kern as tw_kern
from .triweight_kernels import tw_cuml_kern as tw_cuml_kern

__all__ = (
    "gaussian_weighted_histogram",
    "gaussian_weighted_profile",
    "triweighted_kernel_histogram",
    "triweighted_kernel_profile",
    "triweighted_kernel_histogram_with_derivs",
)


@njit
def triweighted_kernel_histogram_with_derivs(log10mstar, log10mstar_jac, bins, sigma):
    """Calculate a triweight-kernel weighted histogram of log10(M*).

    Parameters
    ----------
    log10mstar : ndarray, shape (n_halos, )
    log10mstar_jac : ndarray, shape (n_halos, n_params)
    sigma : float
    bins : ndarray, shape (n_bin_edges, )

    Returns
    -------
    hist : ndarray, shape (n_bin_edges-1, )
    hist_jac : ndarray, shape (n_bin_edges-1, n_params)
    """
    n_bins = bins.shape[0] - 1
    n_params = log10mstar_jac.shape[1]
    n_data = log10mstar.shape[0]

    hist = np.zeros(n_bins, dtype=np.float64)
    hist_jac = np.zeros((n_bins, n_params), dtype=np.float64)

    for i in range(n_data):
        x = log10mstar[i]
        last_cdf = tw_cuml_kern(bins[0], x, sigma)
        last_cdf_deriv = tw_kern(bins[0], x, sigma)

        for j in range(n_bins):
            new_cdf = tw_cuml_kern(bins[j + 1], x, sigma)
            new_cdf_deriv = tw_kern(bins[j + 1], x, sigma)

            # get the hist weight
            weight = new_cdf - last_cdf
            hist[j] += weight

            # do the derivs
            for k in range(n_params):
                hist_jac[j, k] += (new_cdf_deriv - last_cdf_deriv) * log10mstar_jac[
                    i, k
                ]

            last_cdf = new_cdf
            last_cdf_deriv = new_cdf_deriv

    return hist, hist_jac


def gaussian_weighted_histogram(data, scatter, bins):
    """Sum the contribution of Gaussian-weighted data to each bin.

    Parameters
    ----------
    data : ndarray of shape (nhalos, )

    scatter : float or ndarray of shape (nhalos, )

    bins : ndarray of shape (nbins, )

    Returns
    -------
    weighted_hist : ndarray of shape (nbins-1, )

    """
    data, scatter = _get_1d_arrays(data, scatter)
    bins = np.atleast_1d(bins).astype("f8")
    nbins = bins.shape[0]
    weighted_hist = np.zeros(nbins - 1).astype("f8")

    _numba_gw_hist(data, scatter, bins, weighted_hist)
    return weighted_hist


def triweighted_kernel_histogram(data, scatter, bins):
    """Sum the contribution of triweighted-kernel data to each bin.

    Parameters
    ----------
    data : ndarray of shape (nhalos, )

    scatter : float or ndarray of shape (nhalos, )

    bins : ndarray of shape (nbins, )

    Returns
    -------
    weighted_hist : ndarray of shape (nbins-1, )

    """
    data, scatter = _get_1d_arrays(data, scatter)
    bins = np.atleast_1d(bins).astype("f8")
    nbins = bins.shape[0]
    weighted_hist = np.zeros(nbins - 1).astype("f8")

    _numba_tw_hist(data, scatter, bins, weighted_hist)
    return weighted_hist


def gaussian_weighted_profile(data_to_bin, data_to_sum, scatter, bins):
    """Sum data_to_sum in each bin using Gaussian-weighted bin assignment.

    Parameters
    ----------
    data_to_bin : ndarray of shape (nhalos, )

    data_to_sum : ndarray of shape (nhalos, )

    scatter : float or ndarray of shape (nhalos, )

    bins : ndarray of shape (nbins, )

    Returns
    -------
    weighted_prof : ndarray of shape (nbins-1, )
        Sum of data_to_sum in each bin

    weighted_hist : ndarray of shape (nbins-1, )
        Gaussian-weighted sum of data_to_sum in each bin

    """
    data_to_bin, data_to_sum, scatter = _get_1d_arrays(
        data_to_bin, data_to_sum, scatter
    )
    bins = np.atleast_1d(bins).astype("f8")
    nbins = bins.shape[0]
    weighted_prof = np.zeros(nbins - 1).astype("f8")
    weighted_hist = np.zeros(nbins - 1).astype("f8")

    _numba_gw_prof(
        data_to_bin, data_to_sum, scatter, bins, weighted_prof, weighted_hist
    )

    return weighted_prof, weighted_hist


def triweighted_kernel_profile(data_to_bin, data_to_sum, scatter, bins):
    """Sum data_to_sum in each bin using Gaussian-weighted bin assignment.

    Parameters
    ----------
    data_to_bin : ndarray of shape (nhalos, )

    data_to_sum : ndarray of shape (nhalos, )

    scatter : float or ndarray of shape (nhalos, )

    bins : ndarray of shape (nbins, )

    Returns
    -------
    weighted_prof : ndarray of shape (nbins-1, )
        Sum of data_to_sum in each bin

    weighted_hist : ndarray of shape (nbins-1, )
        Gaussian-weighted sum of data_to_sum in each bin

    """
    data_to_bin, data_to_sum, scatter = _get_1d_arrays(
        data_to_bin, data_to_sum, scatter
    )
    bins = np.atleast_1d(bins).astype("f8")
    nbins = bins.shape[0]
    weighted_prof = np.zeros(nbins - 1).astype("f8")
    weighted_hist = np.zeros(nbins - 1).astype("f8")

    _numba_tw_prof(
        data_to_bin, data_to_sum, scatter, bins, weighted_prof, weighted_hist
    )

    return weighted_prof, weighted_hist


@njit
def _numba_gw_hist(data, scatter, bins, khist):
    """Numba kernel for the Gaussian-weighted histogram.

    Parameters
    ----------
    data : ndarray of shape (ndata, )

    scatter : ndarray of shape (ndata, )

    bins : ndarray of shape (nbins, )

    khist : ndarray of shape (nbins-1, )
        Empty array used to store the weighted histogram

    """
    ndata = len(data)
    nbins = len(bins)
    bot = bins[0]
    sqrt2 = math_sqrt(2)

    for i in range(ndata):
        x = data[i]
        scale = scatter[i]

        z = (x - bot) / scale / sqrt2
        last_cdf = 0.5 * (1.0 + math_erf(z))
        for j in range(1, nbins):
            bin_edge = bins[j]
            z = (x - bin_edge) / scale / sqrt2
            new_cdf = 0.5 * (1.0 + math_erf(z))
            weight = last_cdf - new_cdf
            khist[j - 1] += weight
            last_cdf = new_cdf


@njit
def _numba_tw_hist(data, scatter, bins, khist):
    """Numba kernel for the triweighted kernel histogram.

    Parameters
    ----------
    data : ndarray of shape (ndata, )

    scatter : ndarray of shape (ndata, )

    bins : ndarray of shape (nbins, )

    khist : ndarray of shape (nbins-1, )
        Empty array used to store the weighted histogram

    """
    ndata = len(data)
    nbins = len(bins)
    bot = bins[0]

    for i in range(ndata):
        x = data[i]
        scale = scatter[i]

        last_cdf = tw_cuml_kern(x, bot, scale)
        for j in range(1, nbins):
            bin_edge = bins[j]
            new_cdf = tw_cuml_kern(x, bin_edge, scale)
            weight = last_cdf - new_cdf
            khist[j - 1] += weight
            last_cdf = new_cdf


@njit
def _numba_gw_prof(data_to_bin, data_to_sum, scatter, bins, khist, whist):
    """Numba kernel for the Gaussian-weighted profile.

    Parameters
    ----------
    data_to_bin : ndarray of shape (ndata, )

    data_to_sum : ndarray of shape (ndata, )

    scatter : ndarray of shape (ndata, )

    bins : ndarray of shape (nbins, )

    khist : ndarray of shape (nbins-1, )
        Empty array used to store the weighted profile

    whist : ndarray of shape (nbins-1, )
        Empty array used to store the weighted histogram

    """
    ndata = len(data_to_bin)
    nbins = len(bins)
    bot = bins[0]
    sqrt2 = math_sqrt(2)

    for i in range(ndata):
        x = data_to_bin[i]
        d = data_to_sum[i]
        scale = scatter[i]

        z = (x - bot) / scale / sqrt2
        last_cdf = 0.5 * (1.0 + math_erf(z))
        for j in range(1, nbins):
            bin_edge = bins[j]
            z = (x - bin_edge) / scale / sqrt2
            new_cdf = 0.5 * (1.0 + math_erf(z))
            weight = last_cdf - new_cdf
            khist[j - 1] += weight * d
            whist[j - 1] += weight
            last_cdf = new_cdf


@njit
def _numba_tw_prof(data_to_bin, data_to_sum, scatter, bins, khist, whist):
    """Numba kernel for the Gaussian-weighted profile.

    Parameters
    ----------
    data_to_bin : ndarray of shape (ndata, )

    data_to_sum : ndarray of shape (ndata, )

    scatter : ndarray of shape (ndata, )

    bins : ndarray of shape (nbins, )

    khist : ndarray of shape (nbins-1, )
        Empty array used to store the weighted profile

    whist : ndarray of shape (nbins-1, )
        Empty array used to store the weighted histogram

    """
    ndata = len(data_to_bin)
    nbins = len(bins)
    bot = bins[0]

    for i in range(ndata):
        x = data_to_bin[i]
        d = data_to_sum[i]
        scale = scatter[i]

        last_cdf = tw_cuml_kern(x, bot, scale)
        for j in range(1, nbins):
            bin_edge = bins[j]
            new_cdf = tw_cuml_kern(x, bin_edge, scale)
            weight = last_cdf - new_cdf
            khist[j - 1] += weight * d
            whist[j - 1] += weight
            last_cdf = new_cdf


def _get_1d_arrays(*args):
    """Return a list of ndarrays of the same length."""
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
