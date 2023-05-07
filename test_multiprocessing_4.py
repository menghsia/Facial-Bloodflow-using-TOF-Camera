import multiprocessing as mp
import numpy as np
import time
import ctypes

def worker_function(i, shared_arr):
    print(f"{i}: Worker starting...")

    # Modify the shared array in-place
    shared_arr[i] = i * 2

    # Wait randomly between 2 and 5 seconds
    time.sleep(np.random.randint(2, 5))

    print(f"{i}: Worker exiting...")

if __name__ == '__main__':
    # # Create a shared array with 10 elements of type double
    # arr_size = 10
    # arr_type = ctypes.c_double * arr_size
    # shared_arr = mp.Array(arr_type, arr_size, lock=False)

    # Create a Manager object and a shared array with 10 elements of type double
    manager = mp.Manager()
    arr_size = 40
    shared_arr = manager.list([0.0] * arr_size)

    print("Before:")
    print(shared_arr)

    # Create a pool of 16 worker processes
    with mp.Pool(processes=16) as pool:
        # Call worker_function() for each index in the shared array
        # pool.starmap(worker_function, [(i, shared_arr) for i in range(arr_size)])

        results = []

        for i in range(arr_size):
            results.append(pool.apply_async(worker_function, args=(i, shared_arr)))

        # results = [pool.apply_async(worker_function, args=(i, shared_arr)) for i in range(arr_size)]

        for r in results:
            r.wait()


    # Print the contents of the shared array
    print("After:")
    print(shared_arr)
