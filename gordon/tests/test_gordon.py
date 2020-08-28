"""
"""
import numpy as np
from ..sigmoid_smhm import logsm_from_logmhalo_jax, DEFAULT_PARAM_VALUES
from ..kernel_weighted_hist import triweighted_kernel_histogram_with_derivs as twhist

LOGM = np.linspace(8, 15, 500)
PARAMS = np.array(list(DEFAULT_PARAM_VALUES.values()))


def test_logsm_from_logmhalo_evaluates():
    logsm = logsm_from_logmhalo_jax(LOGM, PARAMS)
    assert np.all(np.isfinite(logsm))


def test_kernel_hist_evaluates():
    logsm = np.array(logsm_from_logmhalo_jax(LOGM, PARAMS))
    logsm_bins = np.linspace(10, 11, 10)
    n_halos, n_params = logsm.size, len(PARAMS)
    sigma = 0.25
    jac = np.random.uniform(0, 1, size=n_halos * n_params).reshape((n_halos, n_params))

    h, h_jac = twhist(logsm, np.array(jac), logsm_bins, sigma)
    assert h.shape == (logsm_bins.size - 1,)
    assert h_jac.shape == (logsm_bins.size - 1, n_params)
