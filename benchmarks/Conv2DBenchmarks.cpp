#include <benchmark/benchmark.h>

#include <immintrin.h>

#include <Conv2D.h>
/*
extern void ConvertGroupToMatrix(const uint8_t *tensor, int height, int width, int channels, int channels_per_group,
    int filter_height, int filter_width, int dilation_y, int dilation_x, int stride_y, int stride_x,
    int pad_top, int pad_left, uint8_t *matrix, int res_height, int res_width);

static void BM_GroupToMatrix(benchmark::State &state)
{
    constexpr int srcH = 128, srcW = 128, srcC = 128, padY = 0, padH = 0, padX = 0, padW = 0;
    constexpr int dilationY = 1, dilationX = 1, kernelY = 3, kernelX = 3, strideY = 1, strideX = 1;
    auto tensor = new uint8_t[srcH * srcW * srcC];
    int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
    int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
    auto matrix = new uint8_t[dstH * dstW * srcC * kernelY * kernelX];
    auto ptr_matrix = matrix;
    for (auto _ : state) {
        ConvertGroupToMatrix(tensor, srcH, srcW, srcC, srcC, kernelY, kernelX, dilationY, dilationX,
            strideY, strideX, padY, padX, matrix, dstH / 4, dstW);
    }
    delete[] tensor;
    delete[] matrix;
}
BENCHMARK(BM_GroupToMatrix)->Repetitions(10)->Unit(benchmark::TimeUnit::kMillisecond);

static void BM_memcpy(benchmark::State &state)
{
    constexpr size_t SIZE = 1024 * 1024;
    auto src = new char[SIZE];
    auto dst = new char[SIZE];
    for (auto _ : state) {
        memcpy(dst, src, SIZE);
    }
    delete[] src;
    delete[] dst;
}
//BENCHMARK(BM_memcpy)->Repetitions(10);

static void BM_vectorizedmemcpy(benchmark::State &state)
{
    constexpr size_t SIZE = 1024 * 1024 + 1;
    auto src = new char[SIZE];
    auto dst = new char[SIZE];
    for (auto _ : state) {
        auto tmp_src = src;
        auto tmp_dst = dst;
        for (auto i = SIZE; i; i -= 64) {
            __m256i v1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(tmp_src));
            __m256i v2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(tmp_src + 32));
            tmp_src += 64;
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp_dst), v1);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp_dst + 32), v2);
            tmp_dst += 64;
        }
    }
    delete[] src;
    delete[] dst;
}
//BENCHMARK(BM_vectorizedmemcpy)->Repetitions(10);
*/
