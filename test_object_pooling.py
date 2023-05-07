# Consider implementing object pooling if we need to create and destroy a large
# number of objects that are expensive to create.

class BufferPool:
    def __init__(self, size, count):
        self.buffers = [bytearray(size) for _ in range(count)]
        self.free = list(range(count))

    def acquire(self):
        if not self.free:
            raise RuntimeError('No free buffers available')
        index = self.free.pop()
        return self.buffers[index]

    def release(self, buffer):
        index = self.buffers.index(buffer)
        self.free.append(index)

def process_data(data, pool):
    # Get buffer from pool
    buffer = pool.acquire()

    # Process data in chunks
    while data:
        # Copy data to buffer
        chunk = data[:1024]
        buffer[:len(chunk)] = chunk

        # Process buffer
        process_buffer(buffer)

        # Remove processed data from input
        data = data[1024:]

    # Release buffer back to pool
    pool.release(buffer)

# Create buffer pool
pool = BufferPool(1024, 10)

# Process data using buffer pool
data = get_data()
process_data(data, pool)
