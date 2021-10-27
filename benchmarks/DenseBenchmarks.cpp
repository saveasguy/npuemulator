#include <benchmark/benchmark.h>

#include <Dense.h>

static void BM_Dense(benchmark::State &state)
{
    constexpr int SIZE1 = 4096;
    constexpr int SIZE2 = 1000;
    auto w = new int8_t[SIZE1 * SIZE2];
    npuemulator::Matrix weights(w, SIZE2, SIZE1);
    auto s = new int8_t[SIZE1];
    npuemulator::Vector src(s, SIZE1);
    auto d = new int8_t[SIZE2];
    npuemulator::Vector dst(d, SIZE2);
    for (auto _ : state) {
        npuemulator::Dense(weights, src, dst);
    }
    delete[] w;
    delete[] s;
    delete[] d;
}
//BENCHMARK(BM_Dense)->Repetitions(10)->Unit(benchmark::TimeUnit::kMillisecond)->Iterations(800);
