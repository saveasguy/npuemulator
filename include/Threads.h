#ifndef THREADS_H
#define THREADS_H

#include <cstdint>
#include <exception>

namespace npuemulator {

int CountThreads();

void RunTask(void (*proc)(int8_t *), int8_t *args);

void WaitTasks();

void HandleThreadException(std::exception_ptr exc);

}

#endif
