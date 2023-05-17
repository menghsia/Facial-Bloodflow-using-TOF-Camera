import concurrent.futures
import threading
import time
import os
import numpy as np

def worker_function(idx, shared_array):
    print(f"{threading.current_thread().name}: starting...")

    # Do work
    result = idx ** 2
    shared_array[idx] = result

    print(f"{threading.current_thread().name}: finished")

# Get the number of available threads
num_threads = os.cpu_count()

# If the CPU does not support multithreading, set num_threads to 1 (single-threaded)
if num_threads is None:
    num_threads = 1
elif num_threads < 1:
    num_threads = 1

print(f"Using {num_threads} threads")

# Create a ThreadPoolExecutor with num_threads threads
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit tasks to the ThreadPool

    num_tasks = 35

    # Create a shared array of num_tasks elements
    shared_array = np.zeros(num_tasks, dtype=np.int32)

    print(f"shared_array before: {shared_array}")

    # Create array of numbers from 0 to num_tasks
    shared_indices = np.arange(num_tasks)

    futures = []

    for idx in shared_indices:
        new_future = executor.submit(worker_function, idx, shared_array)
        futures.append(new_future)

    # Wait for the tasks to complete
    concurrent.futures.wait(futures)

    print(f"shared_array after: {shared_array}")
