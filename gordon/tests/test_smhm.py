"""
"""
import os
import numpy as np
from ..sigmoid_smhm import logsm_from_logmhalo

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DDRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_smhm_correctness():
    X = np.loadtxt(os.path.join(DDRN, "logsm_from_logmhalo_testing.dat"))
    logmhalo = X[:, 0]
    logsm_correct = X[:, 1]
    logsm = logsm_from_logmhalo(logmhalo)
    assert np.allclose(logsm, logsm_correct, atol=0.001)
