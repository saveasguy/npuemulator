#include <benchmark/benchmark.h>

#include <Matmul.h>
#include <Threads.h>
#include <Types.h>

static void BM_Matmul(benchmark::State &state)
{
    constexpr size_t SIZE1 = 1024;
    constexpr size_t SIZE2 = 1024;
    constexpr size_t SIZE3 = 1024;
    int8_t *a = new int8_t[SIZE1 * SIZE2];
    npuemulator::Matrix m1(a, SIZE1, SIZE2);
    int8_t *b = new int8_t[SIZE2 * SIZE3];
    npuemulator::Matrix m2(b, SIZE2, SIZE3);
    int8_t *c = new int8_t[SIZE1 * SIZE3];
    npuemulator::Matrix m3(c, SIZE1, SIZE3);
    constexpr size_t SIZE2_MULTIPLY2 = (SIZE2 + 1) & -2;
    constexpr size_t SIZE3_MULTIPLY32 = (SIZE2 + 31) & -32;
    int8_t *d = new int8_t[2 * NPUEMUL_THREADS.Count() * SIZE2_MULTIPLY2 * SIZE3_MULTIPLY32];
    npuemulator::Matrix m4(d, NPUEMUL_THREADS.Count() * SIZE2_MULTIPLY2, 2 * SIZE3_MULTIPLY32);
    for (auto _ : state) {
        npuemulator::ParallelMatmul(m1, m2, m3, m4);
    }
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
}
BENCHMARK(BM_Matmul)->Repetitions(10)->Unit(benchmark::TimeUnit::kMillisecond);