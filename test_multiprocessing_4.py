import multiprocessing as mp
import numpy as np
import time

from managed_shared_memory import ManagedSharedMemory

# Access and modify the shared array from multiple processes
def worker_function(frame_num, managed_shared_buffer):
    print(f"{frame_num}: Worker starting...")

    # Load the shared memory buffer into a numpy array so we can use numpy functions on it
    # shared_np_arr = create_numpy_array_from_shared_buffer(shared_buffer, shared_memory_dtype, shared_array_shape)
    shared_np_arr = managed_shared_buffer.get_numpy_array_from_shared_buffer()

    # Wait randomly between 2 and 5 seconds
    # time.sleep(np.random.randint(2, 5))

    # Wait 3 seconds
    # time.sleep(3)
    
    shared_np_arr[:, frame_num] = frame_num + 1

    print(f"{frame_num}: Worker exiting...")

if __name__ == '__main__':
    num_ROIs = 7
    num_frames = 40

    # Datatype to be used for shared memory buffer
    shared_memory_dtype = np.float16
    # Shape that the shared memory buffer should be reshaped to after loading and before using
    shared_array_shape = (num_ROIs, num_frames)

    # # Create a shared memory buffer with num_ROIs*num_frames elements of type int16
    # shared_buffer = create_shared_buffer(shared_memory_dtype, "shared_buffer", shared_array_shape)

    # # Create a numpy array that uses the same memory as shared_buffer so we can use numpy functions on it
    # shared_np_arr = create_numpy_array_from_shared_buffer(shared_buffer, shared_memory_dtype, shared_array_shape)

    # # Fill the shared array with zeros
    # shared_np_arr.fill(0)

    # Create a managed shared memory buffer with num_ROIs*num_frames elements of type int16
    managed_shared_buffer = ManagedSharedMemory(shared_memory_dtype, "shared_buffer", shared_array_shape, zero_out_buffer=True)
    shared_np_arr = managed_shared_buffer.get_numpy_array_from_shared_buffer()

    print("Before:")
    # print(shared_buffer)
    print(shared_np_arr)

    start_time = time.time()

    # Get number of threads supported by the CPU
    num_threads = mp.cpu_count()

    # If the CPU does not support multithreading, set num_threads to 1 (single-threaded)
    if num_threads < 1:
        num_threads = 1

    # Create a pool with num_threads processes
    pool = mp.Pool(processes=num_threads)

    results = []

    # Queue each task using apply_async() so that the tasks are executed in parallel
    results = [pool.apply_async(worker_function, args=(i, managed_shared_buffer)) for i in range(num_frames)]

    # Wait for all processes to finish
    pool.close()
    pool.join()

    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time}")

    # Print the contents of the shared array
    print("After:")
    # print(shared_buffer)
    print(shared_np_arr)

    # shared_buffer.close()
    # shared_buffer.unlink()

    managed_shared_buffer.clean_up()
