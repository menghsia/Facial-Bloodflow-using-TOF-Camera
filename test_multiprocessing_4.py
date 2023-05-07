import multiprocessing as mp
import numpy as np
import time
import ctypes

def worker_function(i, shared_arr):
    print(f"{i}: Worker starting...")

    # Modify the shared array in-place
    shared_arr[i] = i * 2

    # # Wait randomly between 2 and 5 seconds
    # time.sleep(np.random.randint(2, 5))

    # Wait 3 seconds
    time.sleep(3)

    print(f"{i}: Worker exiting...")

if __name__ == '__main__':
    # num_ROIs = 7
    # num_frames = 40

    # # Create a shared array with 10 elements of type double
    # arr_size = 10
    # arr_type = ctypes.c_double * arr_size
    # shared_arr = mp.Array(arr_type, arr_size, lock=False)

    # Create a Manager object
    manager = mp.Manager()

    # Create a shared array with 10 elements of type double
    arr_size = 40
    shared_buffer = manager.list([0.0] * arr_size)

    # # Create a shared memory buffer using the Manager object
    # shared_buffer = manager.Array('h', num_ROIs * num_frames)

    print("Before:")
    print(shared_buffer)

    start_time = time.time()

    # # Create a pool of 16 worker processes
    # with mp.Pool(processes=16) as pool:
    #     # Call worker_function() for each index in the shared array
    #     # pool.starmap(worker_function, [(i, shared_arr) for i in range(arr_size)])

    #     results = []

    #     # results = [pool.apply_async(worker_function, args=(i, shared_arr)) for i in range(arr_size)]

    #     # for i in range(arr_size):
    #     #     results.append(pool.apply_async(worker_function, args=(i, shared_arr)))

    #     # for r in results:
    #     #     r.wait()

    #     # Queue each task using apply_async()
    #     results = [pool.apply_async(worker_function, args=(i, shared_arr)) for i in range(arr_size)]

    #     # # Start executing tasks as soon as they become available using imap_unordered()
    #     # for r in pool.imap_unordered(lambda x: x.get(), results):
    #     #     pass

    #     # Wait for all processes to finish
    #     pool.close()
    #     pool.join()






    # Create a pool of 16 worker processes
    num_threads = 16
    pool = mp.Pool(processes=num_threads)

    results = []

    # Queue each task using apply_async()
    results = [pool.apply_async(worker_function, args=(i, shared_buffer)) for i in range(arr_size)]

    # # Start executing tasks as soon as they become available using imap_unordered()
    # for r in pool.imap_unordered(lambda x: x.get(), results):
    #     pass

    # Wait for all processes to finish
    pool.close()
    pool.join()

    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time}")


    # Print the contents of the shared array
    print("After:")
    print(shared_buffer)
