"""
This is a benchmark to test a performance in data parallel mode with MPI

"""

import numpy as np
import time

# Need to fix python paths to load main module
import os
import sys

thefile_path = os.path.abspath(os.path.dirname(__file__))
gordon_module_path = os.path.normpath(os.path.join(thefile_path, "../../"))
sys.path.append(gordon_module_path)

from gordon.sigmoid_smhm import _logsm_from_logmhalo_jax_kern, DEFAULT_PARAM_VALUES
from gordon.tests.utils import Perf_data, print_line, print_test_parameters, print_mpi_info


def do_calculation(data, params):
    """
    Main calculation function uses input data initialized early
    """

    # this function uses Numpy to do a calculation
    # Host side only
    result = _logsm_from_logmhalo_jax_kern(data, params)

    return result


def data_initialize(data_size_global):
    """
    Initialize test input data
    """

    data = np.linspace(0, 1, data_size_global)

    return data


def distribute_parameters(params_global, comm):
    """
    Copy same parameters list to all other ranks in a communicator
    """

    params_dup = None

    if comm.Get_rank() == 0:
        params_dup = [params_global for _ in range(comm.Get_size())]
    params_local = comm.scatter(params_dup)

    return params_local


def distribute_data(data_global, comm):
    """
    Copy single 1-D array from root rank to all other.
    com.scatter send a chank of the main array to particular rank
    """

    data_global_splitted = None
    if comm.Get_rank() == 0:
        data_global_splitted = np.array_split(data_global, comm.Get_size())

    data = comm.scatter(data_global_splitted)

    return data


def collect_results(mpi_comm, data_local_results):
    """
    Collect data from all ranks into root.
    Each rank send it's data to root and put it into python list
    Each python list of values put into common python list

    Returns
    -------
    full_data_results : list[list[]]
        On rank root: A python list of python lists of values collected from corresponded rank.
        On other ranks: NoneType
    """

    full_data_results = mpi_comm.gather(data_local_results)

    return full_data_results


if __name__ == "__main__":
    # number of iteration for each measured function
    experiment_iterations = 1

    # Ctreate performance data storage
    perf = Perf_data()

    # main "problem" data size in elements count
    data_global_size = 1024 * 1024 * 1024

    # get parameters which are required by "calculation" function
    params_global = list(DEFAULT_PARAM_VALUES.values())

    print_line(f" Run reference data collection ({experiment_iterations} iterations)")
    for _ in range(experiment_iterations):
        perf.time_start()
        data_global_input = data_initialize(data_global_size)
        perf.time_stop("data_initialize_ref")

        perf.time_start()
        result_reference = do_calculation(data_global_input, params_global)
        perf.time_stop("do_calculation_ref")

    # print performance numbers collected at this point
    perf.data_print()

    print_line(" Run the test with MPI ")
    try:
        from mpi4py import MPI
    except ImportError as err:
        print_test_parameters(data_global_size, DEFAULT_PARAM_VALUES)
        print(f"GORDON_TEST_MPI: Can't load mpi4py module with error:\n{err}")
        raise

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print_test_parameters(data_global_size, DEFAULT_PARAM_VALUES)
    print_mpi_info(MPI)

    for _ in range(experiment_iterations):
        perf.time_start()
        params_local = distribute_parameters(params_global, comm)
        perf.time_stop("distribute_parameters")

        perf.time_start()
        data_local_input = distribute_data(data_global_input, comm)
        perf.time_stop("distribute_data")

        perf.time_start()
        data_local_result = do_calculation(data_local_input, params_local)
        perf.time_stop("do_calculation")

        perf.time_start()
        full_data_results_list = collect_results(comm, data_local_result)
        perf.time_stop("collect_results")

    # analize and validate calculation results
    if rank == 0:
        result = np.concatenate(full_data_results_list)

        # Validate test results
        validation_result = np.allclose(result, result_reference)
        if validation_result:
            print(f"Validation PASSED")
        else:
            print(f"Validation FAILED")
            np.testing.assert_allclose(result, result_reference)

        # print performance data collected early
        perf.data_print()

        print(f"Test finished")
