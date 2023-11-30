"""
This is an utilities used in tests
This file should not contains any tests or benchmarks

"""

import time
import statistics

# number of symbols in width of output
console_output_width = 120


def print_line(msg=""):
    head_str_fill = "="
    print(f"{msg.center(console_output_width, head_str_fill)}")


def print_test_parameters(size, params):
    print_line(" Gordon Test ")
    print(f"Problem size (count): {size}")
    print(f"Parameters: {params}")
    print_line()


def print_mpi_info(lib_MPI):
    """
    Each rank prints information to common output
    Mixed output is expected
    """

    comm = lib_MPI.COMM_WORLD
    comm_size = comm.Get_size()
    # comm_topo = comm.Get_topology()
    rank = comm.Get_rank()

    if rank == 0:
        print_line(" Gordon Test MPI ")
        print(f"Version: {lib_MPI.Get_version()}")
        print(f"Vendor: {lib_MPI.get_vendor()}")
        print(f"Extended:\n{lib_MPI.Get_library_version()}")
        print_line()
    print_line(f"Communicator size {comm_size}, rank {rank} on {lib_MPI.Get_processor_name()}")


class Perf_data:
    storage = dict()
    timer = 0

    def _get_time(self):
        return time.monotonic_ns()
        # return time.monotonic() # for compatibuility

    def time_start(self):
        self.timer = self._get_time()

    def time_stop(self, key):
        if key not in self.storage.keys():
            self.storage[key] = list()

        time_mark = self._get_time()
        self.storage[key].append(time_mark - self.timer)

    def data_print(self):
        column_width = 25
        print_line(" Performance results ")
        for key, value_list_ns in self.storage.items():
            # assume performance scores are in nanoseconds
            value_list_sec = [val_ns / 1.0e09 for val_ns in value_list_ns]

            name_str = key.rjust(column_width)
            tmin = min(value_list_sec)
            tmax = max(value_list_sec)
            tmedian = statistics.median(value_list_sec)  # gives the most reproducable performance value
            print(f"{name_str}: {value_list_sec}, min={tmin:.2e}, max={tmax:.2e}, median={tmedian:.2e}")
