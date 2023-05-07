import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time

# Access and modify the shared array from multiple processes
def worker_function(frame_num, shared_buffer, shared_memory_dtype, shared_array_shape):
    print(f"{frame_num}: Worker starting...")

    # Load the shared memory buffer into a numpy array so we can use numpy functions on it
    shared_np_arr = create_numpy_array_from_shared_buffer(shared_buffer, shared_memory_dtype, shared_array_shape)

    # Wait randomly between 2 and 5 seconds
    # time.sleep(np.random.randint(2, 5))

    # Wait 3 seconds
    # time.sleep(3)
    
    shared_np_arr[:, frame_num] = frame_num + 1

    print(f"{frame_num}: Worker exiting...")

def create_shared_buffer(shared_memory_dtype, shared_memory_name, shared_array_shape):
    """
    Create a shared memory buffer with the specified datatype and shape.

    NOTE: The memory is NOT initialized to any value. The memory is not guaranteed to be initialized to all zeros.

    Args:
        shared_memory_dtype (numpy.dtype): Datatype to be used for shared memory buffer
        shared_memory_name (str): Name of the shared memory buffer (gives a way to access the buffer from other processes)
        shared_array_shape (tuple): Shape that the shared memory buffer should be reshaped to after loading and before using
    
    Returns:
        shared_buffer (multiprocessing.shared_memory.SharedMemory): Shared memory buffer
    """

    # Create a shared memory buffer with dim_1*dim_2*...*dim_n elements of type shared_memory_dtype

    # Calculate the size of the shared memory buffer in bytes (dim_1*dim_2*...*dim_n*size_in_bytes_of_data_type)
    num_bytes = int(np.prod(shared_array_shape) * np.dtype(shared_memory_dtype).itemsize)
    
    # Create the shared memory buffer with a size of num_bytes bytes
    shared_buffer = shared_memory.SharedMemory(create=True, size=num_bytes, name=shared_memory_name)

    return shared_buffer

def create_numpy_array_from_shared_buffer(shared_buffer, shared_memory_dtype, shared_array_shape):
    """
    Create a numpy array that uses the same memory as the specified shared memory buffer.

    NOTE: This does not create a copy of the shared memory buffer. Instead, it creates a numpy array
    that uses the same memory as the shared memory buffer. This means that any changes made to the
    numpy array will also be made to the shared memory buffer and vice versa.

    Args:
        shared_buffer (multiprocessing.shared_memory.SharedMemory): Shared memory buffer
        shared_memory_dtype (numpy.dtype): Datatype used by shared memory buffer
        shared_array_shape (tuple): Shape that the shared memory buffer should be reshaped to after loading and before using
    Returns:
        shared_np_arr (numpy.ndarray): Numpy array that uses the same memory as shared_buffer
    """
    
    # Load the shared memory buffer into a numpy array that uses the same memory as shared_buffer
    shared_np_arr = np.ndarray(shape=shared_array_shape, dtype=shared_memory_dtype, buffer=shared_buffer.buf)
    
    return shared_np_arr

if __name__ == '__main__':
    num_ROIs = 7
    num_frames = 40

    # Datatype to be used for shared memory buffer
    shared_memory_dtype = np.float16
    # Shape that the shared memory buffer should be reshaped to after loading and before using
    shared_array_shape = (num_ROIs, num_frames)

    # Create a shared memory buffer with num_ROIs*num_frames elements of type int16
    shared_buffer = create_shared_buffer(shared_memory_dtype, "shared_buffer", shared_array_shape)

    # Create a numpy array that uses the same memory as shared_buffer so we can use numpy functions on it
    shared_np_arr = create_numpy_array_from_shared_buffer(shared_buffer, shared_memory_dtype, shared_array_shape)

    # Fill the shared array with zeros
    shared_np_arr.fill(0)

    print("Before:")
    # print(shared_buffer)
    print(shared_np_arr)

    start_time = time.time()

    # Create a pool of 16 worker processes
    num_threads = 16
    pool = mp.Pool(processes=num_threads)

    results = []

    # Queue each task using apply_async() so that the tasks are executed in parallel
    results = [pool.apply_async(worker_function, args=(i, shared_buffer, shared_memory_dtype, shared_array_shape)) for i in range(num_frames)]

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
