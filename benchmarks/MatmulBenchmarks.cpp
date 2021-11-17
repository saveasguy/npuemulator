#include <benchmark/benchmark.h>

#include <Matmul.h>
#include <Threads.h>
#include <Types.h>

static void BM_Matmul_256x256x256(benchmark::State &state)
{
    constexpr size_t SIZE1 = 256;
    constexpr size_t SIZE2 = 256;
    constexpr size_t SIZE3 = 256;
    int8_t *a = new int8_t[SIZE1 * SIZE2];
    npuemulator::Matrix m1(a, SIZE1, SIZE2);
    int8_t *b = new int8_t[SIZE2 * SIZE3];
    npuemulator::Matrix m2(b, SIZE2, SIZE3);
    int8_t *c = new int8_t[SIZE1 * SIZE3];
    npuemulator::Matrix m3(c, SIZE1, SIZE3);
    constexpr size_t SIZE2_MULTIPLY2 = (SIZE2 + 1) & -2;
    constexpr size_t SIZE3_MULTIPLY32 = (SIZE3 + 31) & -32;
    int8_t *d = new int8_t[2 * npuemulator::CountThreads() * SIZE2_MULTIPLY2 * SIZE3_MULTIPLY32];
    npuemulator::Matrix m4(d, npuemulator::CountThreads() * SIZE2_MULTIPLY2, 2 * SIZE3_MULTIPLY32);
    for (auto _ : state) {
        npuemulator::ParallelMatmul(m1, m2, m3, m4);
    }
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
}
//BENCHMARK(BM_Matmul_256x256x256)->Iterations(200)->Unit(benchmark::TimeUnit::kMillisecond);

static void BM_Matmul_2048x2048x2048(benchmark::State &state)
{
    constexpr size_t SIZE1 = 2048;
    constexpr size_t SIZE2 = 2048;
    constexpr size_t SIZE3 = 2048;
    int8_t *a = new int8_t[SIZE1 * SIZE2];
    npuemulator::Matrix m1(a, SIZE1, SIZE2);
    int8_t *b = new int8_t[SIZE2 * SIZE3];
    npuemulator::Matrix m2(b, SIZE2, SIZE3);
    int8_t *c = new int8_t[SIZE1 * SIZE3];
    npuemulator::Matrix m3(c, SIZE1, SIZE3);
    constexpr size_t SIZE2_MULTIPLY2 = (SIZE2 + 1) & -2;
    constexpr size_t SIZE3_MULTIPLY32 = (SIZE3 + 31) & -32;
    int8_t *d = new int8_t[2 * npuemulator::CountThreads() * SIZE2_MULTIPLY2 * SIZE3_MULTIPLY32];
    npuemulator::Matrix m4(d, npuemulator::CountThreads() * SIZE2_MULTIPLY2, 2 * SIZE3_MULTIPLY32);
    for (auto _ : state) {
        npuemulator::ParallelMatmul(m1, m2, m3, m4);
    }
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
}
//BENCHMARK(BM_Matmul_2048x2048x2048)->Iterations(10)->Unit(benchmark::TimeUnit::kMillisecond);
