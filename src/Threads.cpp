#include "Threads.h"

npuemulator::Threads::Threads() :
    _processing(true),
    _initialized(false),
    _n_working_threads(0)
{
    bool expected = false;
    bool changed = _initialized.compare_exchange_strong(expected, true);
    if (!changed) {
        return;
    }
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
    _thread_tasks.resize(n_cores - 1);
    _RunThreads(n_threads_per_core);
}

void npuemulator::Threads::RunTask(void (*proc)(int8_t *), int8_t *args)
{
    ++_n_working_threads;
    {
        std::unique_lock<std::mutex> lock(_q_mutex);
        _task_queue.push({proc, args});
    }
    _condition.notify_one();
}

void npuemulator::Threads::WaitThreads()
{
    while ((int)_n_working_threads != 0);
}

bool npuemulator::Threads::_HyperthreadingSupported()
{
    int registers[4];
    constexpr int cpu_features = 1;
    __cpuid(registers, cpu_features);
    return registers[3] >> 28 & 1;
}

void npuemulator::Threads::_RunThreads(unsigned int n_cpus_per_core)
{
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
}

void npuemulator::Threads::_ThreadWork()
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

npuemulator::Threads::~Threads()
{
    _processing = false;
    _condition.notify_all();
    for (auto &th : _additional_threads) {
        th.join();
    }
}
