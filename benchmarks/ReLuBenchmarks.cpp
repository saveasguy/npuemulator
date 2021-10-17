#include <benchmark/benchmark.h>

#include <ReLu.h>
#include <Threads.h>
#include <Types.h>

static void BM_ReLu(benchmark::State &state)
{
    constexpr int SIZE = 128 * 112 * 112;
    auto v1 = new int8_t[SIZE];
    auto v2 = new int8_t[SIZE];
    npuemulator::Vector src(v1, SIZE);
    npuemulator::Vector dst(v2, SIZE);
    for (auto _ : state) {
        npuemulator::ReLu(src, dst);
    }
}
BENCHMARK(BM_ReLu)->Repetitions(10)->Unit(benchmark::TimeUnit::kMillisecond);