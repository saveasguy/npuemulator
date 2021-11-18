#include <benchmark/benchmark.h>

#include <ReLu.h>
#include <Threads.h>
#include <Types.h>

static void BM_ReLu_224x224x64(benchmark::State &state)
{
    constexpr int SIZE = 64 * 224 * 224;
    auto v1 = new int8_t[SIZE];
    auto v2 = new int8_t[SIZE];
    npuemulator::Vector src(v1, SIZE);
    npuemulator::Vector dst(v2, SIZE);
    for (auto _ : state) {
        npuemulator::
        ReLu(src, dst);
    }
    delete[] v1;
    delete[] v2;
}
//BENCHMARK(BM_ReLu_224x224x64)->Iterations(100)->Unit(benchmark::TimeUnit::kMillisecond);
