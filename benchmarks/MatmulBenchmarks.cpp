#include <benchmark/benchmark.h>

#include <Matmul.h>
#include <Threads.h>
#include <Types.h>

extern void ReorderMat2(npuemulator::Matrix mat2, npuemulator::Matrix reordered_mat2);

static void BM_ReorderMat2(benchmark::State &state)
{
    constexpr size_t SIZE = 1024;
    uint8_t *a = new uint8_t[SIZE * SIZE];
    npuemulator::Matrix m1(a, SIZE, SIZE);
    uint8_t *b = new uint8_t[SIZE * SIZE];
    npuemulator::Matrix m2(a, SIZE, SIZE);
    for (auto _ : state) {
        ReorderMat2(m1, m2);
    }
    delete[] a;
    delete[] b;
}
//BENCHMARK(BM_ReorderMat2)->Repetitions(10)->Unit(benchmark::TimeUnit::kMillisecond);

extern "C" void mtrx_mul_f(const float *m1, int h1, int w1, const float *m2, int w2, float *r, float *b);

static void BM_Matmulf(benchmark::State &state)
{
    constexpr size_t SIZE1 = 56 * 56;
    constexpr size_t SIZE2 = 256 * 9;
    constexpr size_t SIZE3 = 256;
    float *a = new float[SIZE1 * SIZE2];
    float *b = new float[SIZE2 * SIZE3];
    float *c = new float[SIZE1 * SIZE3];
    float *d = new float[SIZE2 * SIZE3];
    for (auto _ : state) {
        mtrx_mul_f(a, SIZE1, SIZE2, b, SIZE3, c, d);
    }
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
}
//BENCHMARK(BM_Matmulf)->Repetitions(10)->Unit(benchmark::TimeUnit::kMillisecond);

static void BM_Matmul(benchmark::State &state)
{
    constexpr size_t SIZE1 = 512;
    constexpr size_t SIZE2 = 512;
    constexpr size_t SIZE3 = 512;
    uint8_t *a = new uint8_t[SIZE1 * SIZE2];
    npuemulator::Matrix m1(a, SIZE1, SIZE2);
    uint8_t *b = new uint8_t[SIZE2 * SIZE3];
    npuemulator::Matrix m2(b, SIZE2, SIZE3);
    uint8_t *c = new uint8_t[SIZE1 * SIZE3];
    npuemulator::Matrix m3(c, SIZE1, SIZE3);
    constexpr size_t SIZE2_MULTIPLY2 = (SIZE2 + 1) & -2;
    constexpr size_t SIZE3_MULTIPLY32 = (SIZE2 + 31) & -32;
    uint8_t *d = new uint8_t[NPUEMUL_THREADS.Count() * SIZE2_MULTIPLY2 * SIZE3_MULTIPLY32];
    npuemulator::Matrix m4(d, NPUEMUL_THREADS.Count() * SIZE2_MULTIPLY2, SIZE3_MULTIPLY32);
    for (auto _ : state) {
        npuemulator::ParallelMatmul(m1, m2, m3, m4);
    }
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
}
BENCHMARK(BM_Matmul)->Repetitions(10)->Unit(benchmark::TimeUnit::kMillisecond);