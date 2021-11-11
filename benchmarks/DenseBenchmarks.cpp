#include <benchmark/benchmark.h>

#include <Dense.h>

static void BM_Dense_1000x1000(benchmark::State &state)
{
    constexpr int SIZE1 = 1000;
    constexpr int SIZE2 = 1000;
    auto w = new int8_t[SIZE1 * SIZE2];
    npuemulator::Matrix weights(w, SIZE2, SIZE1);
    auto s = new int8_t[SIZE1];
    npuemulator::Vector src(s, SIZE1);
    auto d = new int8_t[SIZE2];
    npuemulator::Vector dst(d, SIZE2);
    auto b = new int8_t[3 * SIZE1];
    npuemulator::Vector buf(b, 3 * SIZE1);
    for (auto _ : state) {
        npuemulator::ParallelDense(weights, src, dst, buf);
    }
    delete[] w;
    delete[] s;
    delete[] d;
}
//BENCHMARK(BM_Dense_1000x1000)->Iterations(100)->Unit(benchmark::TimeUnit::kMillisecond);
