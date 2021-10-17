#ifndef THREADS_H
#define THREADS_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <vector>

#define AFFINITY_MASK DWORD_PTR
#include <intrin.h>
#include <Windows.h>

#define NPUEMUL_THREADS npuemulator::Threads::Instance()

namespace npuemulator {

struct ThreadTask
{
    void (*proc)(int8_t *);
    int8_t *args;
};

class Threads
{
public:
    static Threads &Instance()
    {
        static Threads inst;
        return inst;
    }

    void RunTask(void (*proc)(int8_t *), int8_t *args);

    void WaitThreads();

    int Count()
    {
        return static_cast<int>(_additional_threads.size() + 1);
    }

private:
    Threads();

    ~Threads();

    Threads(const Threads &) = delete;

    Threads &operator=(const Threads &) = delete;

    bool _HyperthreadingSupported();

    void _SetCurrentThreadAffinity(const std::vector<unsigned int> &cpus);

    void _SetThreadAffinity(std::thread &th, const std::vector<unsigned int> &cpus);

    void _RunThreads(unsigned int n_cpus_per_core);

    AFFINITY_MASK _MakeAffinityMask(const std::vector<unsigned int> &cpus);

    void _ThreadWork();

private:
    std::vector<std::thread> _additional_threads;
    std::vector<ThreadTask> _thread_tasks;
    std::queue<ThreadTask> _task_queue;

    std::atomic_int _n_working_threads;
    std::condition_variable _condition;
    std::mutex _q_mutex;

    bool _processing;
};

}

#endif
