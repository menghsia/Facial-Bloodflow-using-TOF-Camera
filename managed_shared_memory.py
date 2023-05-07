from multiprocessing import shared_memory
import numpy as np

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