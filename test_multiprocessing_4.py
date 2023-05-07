import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import ctypes

# def worker_function(i, shared_arr):
#     print(f"{i}: Worker starting...")

#     # Modify the shared array in-place
#     shared_arr[i] = i * 2

#     # # Wait randomly between 2 and 5 seconds
#     # time.sleep(np.random.randint(2, 5))

#     # Wait 3 seconds
#     time.sleep(3)

#     print(f"{i}: Worker exiting...")

# Access and modify the shared array from multiple processes
def worker_function(shared_buffer, frame_num, shared_array_shape, shared_array_dtype):
    print(f"{frame_num}: Worker starting...")

    # shared_np_arr = np.frombuffer(shared_buffer.get_obj(), dtype=np.int16).reshape(num_ROIs, num_frames)
    shared_np_arr = np.ndarray(shape=shared_array_shape, dtype=shared_array_dtype, buffer=shared_buffer.buf)
    
    shared_np_arr[:, frame_num] = frame_num + 1

    # print(f"{frame_num}: shared_np_arr:")
    # print(shared_np_arr)

    print(f"{frame_num}: Worker exiting...")

if __name__ == '__main__':
    num_ROIs = 7
    num_frames = 40

    # # Create a shared array with 10 elements of type double
    # arr_size = 10
    # arr_type = ctypes.c_double * arr_size
    # shared_arr = mp.Array(arr_type, arr_size, lock=False)

    # Create a Manager object
    manager = mp.Manager()

    # Create a shared array with 10 elements of type double
    # arr_size = 40
    # shared_buffer = manager.list([0.0] * arr_size)

    # Create a shared memory buffer with num_ROIs*num_frames elements of type int16
    # shared_buffer = mp.Array('h', num_ROIs * num_frames, lock=False)

    # Use Manager object to create a shared memory buffer with num_ROIs*num_frames
    # elements of type int16 that can be passed to pool worker processes
    # shared_buffer = manager.Array('h', num_ROIs * num_frames)
    # shared_buffer = manager.list([0] * num_ROIs * num_frames)
    # shared_buffer = manager.Array('h', [0] * num_ROIs * num_frames)

    # Create a shared array using multiprocessing.Manager
    # my_array = manager.Array('i', [1, 2, 3])

    shared_array_shape = (num_ROIs, num_frames)
    shared_array_dtype = np.float16

    mem_size = int(np.dtype(shared_array_dtype).itemsize * np.prod(shared_array_shape))
    shared_buffer = shared_memory.SharedMemory(create=True, size=mem_size, name="shared_buffer")
    shared_np_arr = np.ndarray(shape=shared_array_shape, dtype=shared_array_dtype, buffer=shared_buffer.buf)

    # Create a NumPy array that uses my_array as the memory buffer
    # my_np_array = np.frombuffer(my_array.get_obj(), dtype='i')

    # Create a numpy array that uses the shared memory
    # shared_np_arr = np.frombuffer(shared_buffer.get_obj(), dtype=np.int16).reshape(num_ROIs, num_frames)
    # shared_np_arr = np.array(shared_buffer, dtype=np.int16).reshape(num_ROIs, num_frames)
    # shared_np_arr = np.frombuffer(shared_buffer.get_obj(), dtype=np.int16).reshape(num_ROIs, num_frames)
    shared_np_arr.fill(0)

    print("Before:")
    # print(shared_buffer)
    print(shared_np_arr)

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
    # results = [pool.apply_async(worker_function, args=(i, shared_buffer)) for i in range(num_frames)]
    results = [pool.apply_async(worker_function, args=(shared_buffer, i, shared_array_shape, shared_array_dtype)) for i in range(num_frames)]

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
    # print(shared_buffer)
    print(shared_np_arr)

    shared_buffer.close()
    shared_buffer.unlink()
