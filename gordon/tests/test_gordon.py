"""
"""
import os
import numpy as np
from jax import numpy as jax_np
from jax import grad as jax_grad
from jax import vmap as jax_vmap
from jax import jit as jax_jit
from jax import local_devices
from jax import local_device_count
from jax import device_put
from jax.debug import visualize_array_sharding
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from ..sigmoid_smhm import logsm_from_logmhalo_jax, DEFAULT_PARAM_VALUES
from ..kernel_weighted_hist import triweighted_kernel_histogram_with_derivs as twhist
from ..sigmoid_smhm import _logsm_from_logmhalo_jax_kern


NUM_DEVICES = int(os.environ.get("DEVICES_NM",  1))     # Number of XPUs to be used 
NM = int(os.environ.get("GORDON_NM", 500))

if NUM_DEVICES > local_device_count():
  print("Error: More device("+str(NUM_DEVICES)+") was requested than supported("+str(local_device_count)+")")
  exit(-1)

if NM % local_device_count():
  print("Error: GORDON_NM value : "+str(NM)+" cannot be evenly splitted into: "+str(local_device_count())+" devices")
  exit(-1)

# allocate data on CPU
LOGM = np.linspace(8, 15, NM)
PARAMS = np.array(list(DEFAULT_PARAM_VALUES.values()))

# distribute/replicate data on devices
sharding = PositionalSharding(mesh_utils.create_device_mesh((NUM_DEVICES,),devices=local_devices()[0:NUM_DEVICES]))
LOGM_DEVICES = device_put(LOGM,sharding.reshape(NUM_DEVICES,))
PARAMS_DEVICES = device_put(PARAMS,sharding.replicate()) 
visualize_array_sharding(LOGM_DEVICES)

def test_logsm_from_logmhalo_evaluates():

    logsm = logsm_from_logmhalo_jax(LOGM_DEVICES, PARAMS_DEVICES)
    assert np.all(np.isfinite(logsm))


def test_kernel_hist_evaluates():
    logsm = np.array(logsm_from_logmhalo_jax(LOGM_DEVICES, PARAMS_DEVICES))
    logsm_bins = np.linspace(10, 11, 10)
    n_halos, n_params = logsm.size, len(PARAMS)
    sigma = 0.25
    jac = np.random.uniform(0, 1, size=n_halos * n_params).reshape((n_halos, n_params))

    h, h_jac = twhist(logsm, np.array(jac), logsm_bins, sigma)
    assert h.shape == (logsm_bins.size - 1,)
    assert h_jac.shape == (logsm_bins.size - 1, n_params)


def test_kernel_weighted_hist_of_model_gradients():
    """Compute model gradients with jax and propagate through histogram with numba."""

    logsm = np.array(logsm_from_logmhalo_jax(LOGM_DEVICES, PARAMS_DEVICES))
    logsm_bins = np.linspace(10, 11, 10)
    scatter = 0.25
    _gradfunc = jax_grad(_logsm_from_logmhalo_jax_kern, argnums=1)
    gradfunc = jax_jit(jax_vmap(_gradfunc, in_axes=(0, None)))
    smhm_jac = np.array(gradfunc(LOGM_DEVICES, PARAMS_DEVICES))
    assert np.shape(smhm_jac) == (LOGM.size, len(PARAMS))

    h, h_jac = twhist(logsm, smhm_jac, logsm_bins, scatter)
