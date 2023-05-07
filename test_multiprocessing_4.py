import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time

class ManagedSharedMemory:
    def __init__(self, shared_memory_dtype, shared_memory_name, shared_array_shape, zero_out_buffer=True):
        self.shared_memory_dtype = shared_memory_dtype
        self.shared_memory_name = shared_memory_name
        self.shared_array_shape = shared_array_shape

        # Create a shared memory buffer with the specified datatype and shape
        self.shared_buffer = self._create_shared_buffer()

        if zero_out_buffer:
            # Fill the shared memory buffer with zeros
            self.zero_out_shared_buffer()

        # Set flag to indicate whether or not the shared memory buffer has been cleaned up
        # (either manually or automatically if the destructor was called by the garbage collector)
        self.cleaned_up = False
    
    def get_numpy_array_from_shared_buffer(self):
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
        shared_np_arr = np.ndarray(shape=self.shared_array_shape, dtype=self.shared_memory_dtype, buffer=self.shared_buffer.buf)
        
        return shared_np_arr
    
    def zero_out_shared_buffer(self):
        """
        Fill the shared memory buffer with zeros.
        """

        # Load the shared memory buffer into a numpy array so we can use numpy functions on it
        shared_np_arr = self.get_numpy_array_from_shared_buffer()

        # Initialize the shared numpy array to all zeros
        shared_np_arr.fill(0)

    def clean_up(self):
        """
        Close and unlink the shared memory buffer.

        NOTE: This should be called when the shared memory buffer is no longer needed. If this is not called,
        the garbage collector will attempt to call the destructor automatically, but it should not be relied on.
        """
        self.__del__()

    def __del__(self):
        """
        Destructor for ManagedSharedMemory class.

        NOTE: This should be called automatically by the garbage collector when the object is no longer needed, however it may be
        unreliable. It is recommended to call clean_up() manually when the shared memory buffer is no longer needed.
        """
        if not self.cleaned_up:
            # Close the shared memory buffer
            self.shared_buffer.close()

            # Unlink the shared memory buffer
            self.shared_buffer.unlink()

            self.cleaned_up = True
    
    def _create_shared_buffer(self):
        """
        Create a shared memory buffer with the specified datatype and shape.

        NOTE: The memory is NOT initialized to any value. The memory is not guaranteed to be initialized to all zeros.
        
        Returns:
            shared_buffer (multiprocessing.shared_memory.SharedMemory): Shared memory buffer
        """

        # Create a shared memory buffer with dim_1*dim_2*...*dim_n elements of type shared_memory_dtype

        # Calculate the size of the shared memory buffer in bytes (dim_1*dim_2*...*dim_n*size_in_bytes_of_data_type)
        num_bytes = int(np.prod(self.shared_array_shape) * np.dtype(self.shared_memory_dtype).itemsize)
        
        # Create the shared memory buffer with a size of num_bytes bytes
        shared_buffer = shared_memory.SharedMemory(create=True, size=num_bytes, name=self.shared_memory_name)

        return shared_buffer

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

    # Create a pool of 16 worker processes
    num_threads = 16
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
