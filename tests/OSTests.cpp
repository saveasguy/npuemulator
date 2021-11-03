#include <gtest/gtest.h>

#include <Memory.h>
#include <Threads.h>

void task(int8_t *args)
{

}

TEST(OS_TEST, CreatingThreads)
{
    int n_threads = npuemulator::CountThreads();
    for (int i = 0; i < n_threads - 1; ++i) {
        npuemulator::RunTask(task, nullptr);
    }
    npuemulator::WaitTasks();
}

TEST(OS_TEST, GetCacheSize)
{
    int a = npuemulator::L1CacheSize();
}