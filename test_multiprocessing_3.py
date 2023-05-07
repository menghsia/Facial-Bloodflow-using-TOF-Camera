import numpy as np
import multiprocessing

# Access and modify the shared array from multiple processes
def worker_process(shared_buffer, frame_num, num_ROIs, num_frames):
    print(f"Worker {frame_num} starting...")

    shared_np_arr = np.frombuffer(shared_buffer.get_obj(), dtype=np.int16).reshape(num_ROIs, num_frames)
    
    shared_np_arr[:, frame_num] = frame_num

if __name__ == '__main__':
    # multiprocessing.freeze_support()

    num_ROIs = 7
    # num_frames = 600
    # num_frames = 50
    num_frames = 5

    # Create a shared array of 7*600=4200 int16 values
    shared_buffer = multiprocessing.Array('h', num_ROIs * num_frames)

    # my_obj = shared_buffer.get_obj()
    # type(my_obj)

    # Create a numpy array that uses the shared memory
    # np_arr = np.ndarray(shape=(num_ROIs, num_frames), dtype=np.int16, buffer=arr)
    # np_arr = np.frombuffer(my_obj, dtype='int16').reshape(num_ROIs, num_frames)
    # np_arr = np.frombuffer(my_obj, dtype=np.int16).reshape(num_ROIs, num_frames)
    # np_arr = np.frombuffer(shared_buffer.get_obj(), dtype='int16').reshape(num_ROIs, num_frames)
    shared_np_arr = np.frombuffer(shared_buffer.get_obj(), dtype=np.int16).reshape(num_ROIs, num_frames)
    shared_np_arr.fill(0)

    # print(shared_np_arr.shape)
    # print(shared_np_arr.dtype)
    print(shared_np_arr)

    processes = []

    for i in range(num_frames):
        # Loop through all frames
        # print(f"Creating worker {i}...")

        p = multiprocessing.Process(target=worker_process, args=(shared_buffer, i, num_ROIs, num_frames))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # print(shared_buffer[:])
    print(shared_np_arr)