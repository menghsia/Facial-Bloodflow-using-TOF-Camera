import numpy as np
from multiprocessing import Process, Manager

def my_function(shared_array, start_idx, end_idx):
    # Access shared array as a NumPy array
    np_array = np.frombuffer(shared_array.get_obj())

    # Modify the shared array by setting values within the specified range
    for i in range(start_idx, end_idx):
        np_array[i] = i

if __name__ == '__main__':
    # Create a NumPy array
    # original_array = np.zeros(10)
    original_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Create a shared array using a multiprocessing Manager
    manager = Manager()
    shared_array = manager.Array('i', original_array)

    # Copy the original array into the shared array
    np.frombuffer(shared_array.get_obj())[:] = original_array[:]

    # Start two processes to run my_function with different ranges of indices
    p1 = Process(target=my_function, args=(shared_array, 0, 5))
    p2 = Process(target=my_function, args=(shared_array, 5, 10))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    # Copy the modified shared array back into the original array
    original_array[:] = np.frombuffer(shared_array.get_obj())[:]

    # Print the original array to verify that it was modified
    print(original_array)
