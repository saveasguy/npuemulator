#include "Memory.h"

#include <atomic>
#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
#elif defined(__unix__)
#include <cpuid.h>
#else
#error Platform is not supported!
#endif

namespace {

int GetL1CacheSize()
{
    constexpr int CPU_FEATURE = 4;
    int registers[4];
#ifdef _MSC_VER
    __cpuid(registers, CPU_FEATURE);
#elif defined(__unix__)
    asm volatile ("cpuid":
            "=a"(registers[0]),
            "=b"(registers[1]),
            "=c"(registers[2]),
            "=d"(registers[3]):
            "0"(CPU_FEATURE), "2"(0));
#else
#error Platform is not supported!
#endif
    int ways = registers[1] >> 22 & 0x3FF;
    int line_partitions = registers[1] >> 12 & 0x3FF;
    int line_size = registers[1] & 0xFFF;
    int n_sets = registers[2];
    return (ways + 1) * (line_partitions + 1) * (line_size + 1) * (n_sets + 1);
}

}

int npuemulator::L1CacheSize()
{
    static int l1_cache_size = GetL1CacheSize();
    return l1_cache_size;
}