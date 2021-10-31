#include "Memory.h"

#include <atomic>

#include <intrin.h>

namespace {

int GetL1CacheSize()
{
    int info[4];
    __cpuid(info, 4);
    int ways = info[1] >> 22 & 0x3FF;
    int line_partitions = info[1] >> 12 & 0x3FF;
    int line_size = info[1] & 0xFFF;
    int n_sets = info[2];
    return (ways + 1) * (line_partitions + 1) * (line_size + 1) * (n_sets + 1);
}

}

int npuemulator::L1CacheSize()
{
    static int l1_cache_size = GetL1CacheSize();
    return l1_cache_size;
}