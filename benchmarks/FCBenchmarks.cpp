#include <benchmark/benchmark.h>

#include <FullyConnected.h>

static void BM_FC(benchmark::State &state)
{
    constexpr int SIZE1 = 7 * 7 * 512;
    constexpr int SIZE2 = 4096;
    auto w = new int8_t[SIZE1 * SIZE2];
    npuemulator::Matrix weights(w, SIZE2, SIZE1);
    auto s = new int8_t[SIZE1];
    npuemulator::Vector src(s, SIZE1);
    auto d = new int8_t[SIZE2];
    npuemulator::Vector dst(d, SIZE2);
    for (auto _ : state) {
        npuemulator::FullyConnected(weights, src, dst);
    }
    delete[] w;
    delete[] s;
    delete[] d;
}
BENCHMARK(BM_FC)->Repetitions(10)->Unit(benchmark::TimeUnit::kMillisecond);
