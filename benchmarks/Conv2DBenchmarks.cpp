#include <benchmark/benchmark.h>

#include <immintrin.h>

#include <cstdint>

#include <Conv2D.h>
#include <Threads.h>
#include <Types.h>

static void BM_Conv2D(benchmark::State &state)
{
    constexpr int SIZE = 14, CHANNELS = 512, FILTER_SIZE = 3, FILTER_CHANNELS = 512;
    auto srct = new int8_t[SIZE * SIZE * CHANNELS];
    npuemulator::Tensor src(srct, SIZE, SIZE, CHANNELS);
    auto filtert = new int8_t[FILTER_SIZE * FILTER_SIZE * CHANNELS * FILTER_CHANNELS];
    npuemulator::Tensor filter(filtert, FILTER_SIZE, FILTER_SIZE, FILTER_CHANNELS, CHANNELS);
    constexpr int RES_HEIGHT = (SIZE + 1 + 1 - (1 * (FILTER_SIZE - 1) + 1)) / 1 + 1;
    constexpr int RES_WIDTH = (SIZE + 1 + 1 - (1 * (FILTER_SIZE - 1) + 1)) / 1 + 1;
    auto rest = new int8_t[RES_HEIGHT * RES_WIDTH * FILTER_CHANNELS];
    npuemulator::Tensor res(rest, RES_HEIGHT, RES_WIDTH, FILTER_CHANNELS);
    auto src_matrix = new int8_t[RES_HEIGHT * RES_WIDTH * FILTER_SIZE * FILTER_SIZE * CHANNELS];
    npuemulator::Matrix src_mat(src_matrix, RES_HEIGHT * RES_WIDTH, FILTER_SIZE * FILTER_SIZE * CHANNELS);
    auto filter_buffer = new int8_t[2 * npuemulator::CountThreads() * FILTER_SIZE * FILTER_SIZE * CHANNELS * FILTER_CHANNELS];
    npuemulator::Matrix filter_buf(filter_buffer, npuemulator::CountThreads() * FILTER_SIZE * FILTER_SIZE * CHANNELS, 2 * FILTER_CHANNELS);
    for (auto _ : state)
    {
        npuemulator::Conv2D(src, filter, {1, 1}, {1, 1, 1, 1}, {1, 1}, res, src_mat, filter_buf);
    }
    delete[] srct;
    delete[] filtert;
    delete[] rest;
    delete[] src_matrix;
    delete[] filter_buffer;
}
//BENCHMARK(BM_Conv2D)->Repetitions(10)->Unit(benchmark::TimeUnit::kMillisecond);
