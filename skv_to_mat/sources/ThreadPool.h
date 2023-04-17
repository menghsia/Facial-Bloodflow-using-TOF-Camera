/* 
* ThreadPool.h
*/

#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>
#include <queue>
#include <condition_variable>
#include <latch>
#include <chrono>

/* FUNCTION DECLARATIONS */

class ThreadPool {
private:
    // Contains num_threads threads that will be infinitely listening for new tasks to execute until destructor is called
    std::vector<std::thread> threads_;

    // Contains all tasks that have been enqueued but not yet executed
    std::mutex mutex_tasks_;
    std::queue<std::function<void()>> tasks_;

    // Used to notify threads that there are new tasks available to execute
    std::condition_variable condition_;

    // Used to notify wait() that all tasks have been completed
    std::condition_variable condition_all_tasks_complete;

    // Used to notify threads that the destructor has been called and they should exit their while loop
    bool shutdown_;

    // Used to keep track of how many tasks are remaining to be executed
    std::mutex mutex_num_tasks_remaining_;
    //std::latch num_tasks_remaining;
    std::size_t num_tasks_remaining;

public:
    ThreadPool(size_t num_threads);

    ~ThreadPool();

    template<typename WorkerFunction, typename... Args>
    void enqueue(WorkerFunction&& function, Args&&... args);

    void wait();
};


/* FUNCTION DEFINITIONS */

inline ThreadPool::ThreadPool(size_t num_threads = 0) : shutdown_(false), num_tasks_remaining(0) {
    if (num_threads == 0) {
        // User did not specify number of threads, so use maximum threads available to the system.
        num_threads = std::thread::hardware_concurrency();

        if (num_threads == 0) {
            num_threads = 1;
            std::cout << "WARNING: Could not determine number of hardware threads. Multithreading will be disabled; defaulting to 1 thread." << std::endl;
        }
        else {
            std::cout << "Using " << num_threads << " threads." << std::endl;
        }
    }

    // Reserve num_threads units of memory for threads_ to prevent wasteful resizing and reallocating
    threads_.reserve(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        // Loop through all num_threads and create listeners for each one
        threads_.emplace_back([this] {
            // Each thread listener 
            while (true) {
                std::function<void()> task;

                {
                    // Create new scope to limit lifetime of unique_lock

                    // Lock access to tasks_ (use unique_lock just in case it's already locked and we accidentally lock it a second time)
                    std::unique_lock<std::mutex> lock(mutex_tasks_);

                    // Check if there are new tasks available
                    // If not, keep checking infinitely.
                    // If so, lock
                    condition_.wait(lock, [this] { return !tasks_.empty() || shutdown_; });

                    // If destructor was called, leave the while loop and do not look for a new task to execute
                    // In reality, you should never have "straggler" threads, as long as you let the
                    // destructor call itself implicitly after all tasks are complete
                    if (shutdown_) {
                        break;
                    }

                    // Get latest task
                    // Move it, don't just use it by reference, to prevent a situation where some
                    // other function has reference to it and attempts to edit it after already
                    // sending it to our tasks_ task buffer
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }

                // Unlock access to tasks_ (unique_lock went out of scope)

                // Execute the task we just grabbed
                task();

                // Decrement the num_tasks_remaining count
                mutex_num_tasks_remaining_.lock();
                --num_tasks_remaining;
                mutex_num_tasks_remaining_.unlock();

                condition_all_tasks_complete.notify_one();
            }
            });
    }
}

inline ThreadPool::~ThreadPool() {
    {
        // Lock access to tasks_
        // If we don't, a thread listener might notice that we set shutdown_ to true, exit the
        // condition_.wait() loop, and attempt to grab and run a new task from tasks_.
        std::scoped_lock<std::mutex> lock(mutex_tasks_);

        // Initiate shutdown sequence for all thread listeners
        shutdown_ = true;
    }

    // Unlock access to tasks_ bc scoped_lock went out of scope

    // Alert all thread listeners to check their condition loop
    condition_.notify_all();

    // Wait for all threads to finish their work before continuing program execution
    for (auto& thread : threads_) {
        thread.join();
    }
}

template<typename WorkerFunction, typename... Args>
inline void ThreadPool::enqueue(WorkerFunction&& function, Args&&... args) {
    {
        // Lock access to tasks_
        std::scoped_lock<std::mutex> lock(mutex_tasks_);

        // "bind" the provided function with its provided args, and add to tasks_ queue
        tasks_.emplace(std::bind(std::forward<WorkerFunction>(function), std::forward<Args>(args)...));

        // Increment the num_tasks_remaining count
        mutex_num_tasks_remaining_.lock();
        ++num_tasks_remaining;
        mutex_num_tasks_remaining_.unlock();
    }

    // Unlock access to tasks_ bc scoped_lock went out of scope

    // Notify an available thread to check its condition loop
    condition_.notify_one();
}

inline void ThreadPool::wait() {
    {
        std::unique_lock<std::mutex> lock(mutex_num_tasks_remaining_);
        condition_all_tasks_complete.wait(lock, [this] { return num_tasks_remaining == 0; });
    }
}

#endif /* THREADPOOL_H */
