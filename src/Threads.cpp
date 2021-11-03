#include "Threads.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#include <Windows.h>
#elif defined(__unix__)
#include <cpuid.h>
#include <pthread.h>
#else
#error Platform is not supported!
#endif

namespace {

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

    void RunTask(void (*proc)(int8_t *), int8_t *args)
    {
        ++_n_working_threads;
        {
            std::unique_lock<std::mutex> lock(_q_mutex);
            _task_queue.push({proc, args});
        }
        _condition.notify_one();
    }

    void WaitTasks()
    {
        while ((int)_n_working_threads != 0);
        if (_exception) {
            std::cerr << "exc thrown";
            std::rethrow_exception(_exception);
        }
    }

    int Count()
    {
        return static_cast<int>(_additional_threads.size() + 1);
    }

    void HandleThreadException(std::exception_ptr exc)
    {
        std::lock_guard<std::mutex> lg(_exception_mutex);
        _exception = exc;
    }

private:
    Threads() :
        _processing(true),
        _n_working_threads(0)
    {
        unsigned int n_threads = std::thread::hardware_concurrency();
        if (n_threads > 64) {
            std::cerr << "NPUemulator: CPUs with " << n_threads << " threads are not supported.\n";
            exit(1);
        }
        unsigned int n_cores = n_threads;
        unsigned int n_threads_per_core = 1;
        if (_HyperthreadingSupported()) {
            n_cores = n_threads >> 1;
            n_threads_per_core = 2;
        }
        _additional_threads.resize(n_cores - 1);
        _RunThreads(n_threads_per_core);
    }

    ~Threads()
    {
        _processing = false;
        _condition.notify_all();
        for (auto &th : _additional_threads) {
            if (th.joinable()) {
                th.join();
            }
        }
    }

    Threads(const Threads &) = delete;

    Threads &operator=(const Threads &) = delete;

    bool _HyperthreadingSupported()
    {
        int registers[4];
        constexpr int CPU_FEATURE = 1;
#ifdef _MSC_VER
        __cpuid(registers, CPU_FEATURE);
#elif defined(__unix__)
        __cpuid(CPU_FEATURE, registers[0], registers[1], registers[2], registers[3]);
#else
#error Platform is not supported!
#endif
        return registers[3] >> 28 & 1;
    }

    void _RunThreads(unsigned int n_cpus_per_core)
    {
#ifdef _MSC_VER
        DWORD_PTR mask = 0;
        for (unsigned int i = 0; i < n_cpus_per_core; ++i) {
            mask &= static_cast<DWORD_PTR>(1) << i;
        }
        SetThreadAffinityMask(GetCurrentThread(), mask);
        for (auto &th : _additional_threads) {
            th = std::thread(&Threads::_ThreadWork, this);
            mask <<= n_cpus_per_core;
            SetThreadAffinityMask((HANDLE)th.native_handle(), mask);
        }
#elif defined(__unix__)
        int physical_cpu = 0;
        cpu_set_t mask;
        for (auto &th : _additional_threads) {
            CPU_ZERO(&mask);
            for (int i = 0; i < n_cpus_per_core; ++i) {
                CPU_SET(physical_cpu + i, &mask);
            }
            th = std::thread(&Threads::_ThreadWork, this);
            pthread_setaffinity_np(th.native_handle(), sizeof(mask), &mask);
            physical_cpu += n_cpus_per_core;
        }
#else
#error Platform is not supported!
#endif
    }

    void _ThreadWork()
    {
        ThreadTask task;
        while (_processing) {
            {
                std::unique_lock<std::mutex> lock(_q_mutex);
                _condition.wait(lock, [this]()->bool {
                    return !_task_queue.empty() || !_processing;
                });
                if (!_processing) {
                    return;
                }
                task = _task_queue.front();
                _task_queue.pop();
            }
            task.proc(task.args);
            --_n_working_threads;
        }
    }

private:
    std::vector<std::thread> _additional_threads;
    std::queue<ThreadTask> _task_queue;

    std::mutex _exception_mutex;
    std::exception_ptr _exception;

    std::atomic_int _n_working_threads;
    std::condition_variable _condition;
    std::mutex _q_mutex;

    bool _processing;
};

}

int npuemulator::CountThreads()
{
    return Threads::Instance().Count();
}

void npuemulator::RunTask(void (*proc)(int8_t *), int8_t *args)
{
    Threads::Instance().RunTask(proc, args);
}

void npuemulator::WaitTasks()
{
    Threads::Instance().WaitTasks();
}

void npuemulator::HandleThreadException(std::exception_ptr exc)
{
    Threads::Instance().HandleThreadException(exc);
}
