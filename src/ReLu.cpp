#include "ReLu.h"

#include <iostream>

#include <immintrin.h>

#include <Threads.h>

void npuemulator::ReLu(Vector src, Vector dst)
{
    if (src.length != dst.length) {
        std::cerr << "npuemulator: ReLu: Wrong lengths!" << std::endl;
        exit(1);
    }
    __m256i zerov = _mm256_setzero_si256();
    int i = src.length;
    for (; i >= 64; i -= 64) {
        __m256i v1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(src.data));
        __m256i v2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(src.data + 32));
        src.data += 64;
        v1 = _mm256_max_epi8(v1, zerov);
        v2 = _mm256_max_epi8(v2, zerov);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst.data), v1);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst.data + 32), v2);
        dst.data += 64;
    }
    if (i) {
        int8_t data[64];
        memset(data, 0, i);
        memcpy(data, src.data, i);
        __m256i v1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(data));
        __m256i v2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(data + 32));
        v1 = _mm256_max_epi8(v1, zerov);
        v2 = _mm256_max_epi8(v2, zerov);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(data), v1);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(data + 32), v2);
        memcpy(dst.data, data, i);
    }
}

void ReLuWrapper(int8_t *args)
{
    auto src = *reinterpret_cast<npuemulator::Vector *>(args);
    auto dst = *reinterpret_cast<npuemulator::Vector *>(args + sizeof(npuemulator::Vector));
    ReLu(src, dst);
}

#define PUSH_RELU_ARGS(ARGS, SRC, DST)\
    *reinterpret_cast<npuemulator::Vector *>(ARGS) = SRC;\
    *reinterpret_cast<npuemulator::Vector *>(ARGS + sizeof(npuemulator::Vector)) = DST;

void npuemulator::ParallelReLu(Vector src, Vector dst)
{
    int n_threads = NPUEMUL_THREADS.Count();
    if (src.length < n_threads) {
        ReLu(src, dst);
        return;
    }
    int length = src.length;
    dst.length = src.length /= n_threads;
    constexpr size_t ARGS_SIZE = 2 * sizeof(npuemulator::Vector);
    auto (*args)[ARGS_SIZE] = new int8_t[n_threads - 1][ARGS_SIZE];
    for (int i = 0; i < n_threads - 1; ++i) {
        PUSH_RELU_ARGS(args[i], src, dst);
        ReLuWrapper(args[i]);
        length -= src.length;
        src.data += src.length;
        dst.data += dst.length;
    }
    ReLu(src, dst);
    NPUEMUL_THREADS.WaitThreads();
    delete[] args;
}
