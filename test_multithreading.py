import concurrent.futures
import threading
import time
import os

def process_number(number):
    result = number ** 2
    print(f"Thread {threading.current_thread().name}: {number} squared is {result}")
    time.sleep(2)

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

    # Create array of numbers from 1 to 35
    numbers = range(1, 36)
    
    futures = [executor.submit(process_number, num) for num in numbers]

    # Wait for the tasks to complete
    concurrent.futures.wait(futures)
