#include "Dense.h"

#include <iostream>
#include <numeric>

#include <immintrin.h>

#include "Errors.h"
#include "Threads.h"

inline int8_t BuildResult(int16_t *vector, int8_t *other_weights, int8_t *other_src, int len) {
    int8_t r = std::accumulate(vector, vector + 16, (int16_t)0);
    for (int i = 0; i < len; ++i) {
        r += other_weights[i] * other_src[i];
    }
    return r;
}

inline void ComputeValues(int8_t *weights, int width, int8_t *src, int8_t *dst, int rows = 4)
{
    auto a0 = _mm256_setzero_si256();
    auto a1 = _mm256_setzero_si256();
    auto a2 = _mm256_setzero_si256();
    auto a3 = _mm256_setzero_si256();
    int offset1 = rows > 1 ? width : 0;
    int offset2 = rows > 2 ? 2 * width : 0;
    int offset3 = rows > 3 ? 3 * width : 0;
    int j = width;
    for (; j >= 16; j -= 16) {
        auto s1 = _mm256_cvtepi8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(src)));
        src += 16;
        auto v1 = _mm256_cvtepi8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(weights)));
        auto v2 = _mm256_cvtepi8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(weights + offset1)));
        auto r0 = _mm256_mullo_epi16(v1, s1);
        auto r1 = _mm256_mullo_epi16(v2, s1);
        a0 = _mm256_add_epi16(a0, r0);
        a1 = _mm256_add_epi16(a1, r1);
        auto v3 = _mm256_cvtepi8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(weights + offset2)));
        auto v4 = _mm256_cvtepi8_epi16(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(weights + offset3)));
        auto r2 = _mm256_mullo_epi16(v3, s1);
        auto r3 = _mm256_mullo_epi16(v4, s1);
        a2 = _mm256_add_epi16(a2, r2);
        a3 = _mm256_add_epi16(a3, r3);
        weights += 16;
    }
    int16_t res[16];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(res), a0);
    dst[0] = BuildResult(res, weights, src, j);
    if (rows > 1) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(res), a1);
        dst[1] = BuildResult(res, weights + width, src, j);
    }
    if (rows > 2) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(res), a2);
        dst[2] = BuildResult(res, weights + 2 * width, src, j);
    }
    if (rows > 3) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(res), a3);
        dst[3] = BuildResult(res, weights + 3 * width, src, j);
    }
}

void npuemulator::Dense(Matrix weights, Vector src, Vector dst, Vector bias)
{
    EqualOrDie("Dense", "weight height", weights.height, "dst length", dst.length);
    EqualOrDie("Dense", "weight width", weights.width, "src length", src.length);
    if (bias.length != 0) {
        EqualOrDie("Dense", "bias length", bias.length, "dst length", dst.length);
    }
    int i = weights.height;
    for (; i >= 4; i -= 4) {
        ComputeValues(weights.data, weights.width, src.data, dst.data);
        weights.data += 4 * weights.width;
        dst.data += 4;
    }
    if (i) {
        ComputeValues(weights.data, weights.width, src.data, dst.data, i);
    }
    for (int j = 0; j < bias.length; ++j) {
        dst.data[j] += bias.data[j];
    }
}

void DenseWrapper(int8_t *args)
{
    npuemulator::Matrix weights = *reinterpret_cast<npuemulator::Matrix *>(args);
    npuemulator::Vector src = *reinterpret_cast<npuemulator::Vector *>(args + sizeof(npuemulator::Matrix));
    npuemulator::Vector dst = *reinterpret_cast<npuemulator::Vector *>(args + sizeof(npuemulator::Matrix) + sizeof(npuemulator::Vector));
    npuemulator::Vector buffer = *reinterpret_cast<npuemulator::Vector *>(args + sizeof(npuemulator::Matrix) + 2 * sizeof(npuemulator::Vector));
    npuemulator::Vector bias = *reinterpret_cast<npuemulator::Vector *>(args + sizeof(npuemulator::Matrix) + 3 * sizeof(npuemulator::Vector));
    memcpy(buffer.data, src.data, src.length);
    try {
        npuemulator::Dense(weights, buffer, dst, bias);
    }
    catch (...) {
        npuemulator::HandleThreadException(std::current_exception());
    }
}

#define PUSH_DENSE_ARGS(ARGS, WEIGHTS, SRC, DST, BUFFER, BIAS)\
    *reinterpret_cast<npuemulator::Matrix *>(ARGS) = WEIGHTS;\
    *reinterpret_cast<npuemulator::Vector *>(ARGS + sizeof(npuemulator::Matrix)) = SRC;\
    *reinterpret_cast<npuemulator::Vector *>(ARGS + sizeof(npuemulator::Matrix) + sizeof(npuemulator::Vector)) = DST;\
    *reinterpret_cast<npuemulator::Vector *>(ARGS + sizeof(npuemulator::Matrix) + 2 * sizeof(npuemulator::Vector)) = BUFFER;\
    *reinterpret_cast<npuemulator::Vector *>(ARGS + sizeof(npuemulator::Matrix) + 3 * sizeof(npuemulator::Vector)) = BIAS;

void npuemulator::ParallelDense(Matrix weights, Vector src, Vector dst, Vector buffer, Vector bias)
{
    int n_threads = CountThreads();
    GreaterOrEqualOrDie("Dense", "buffer length", buffer.length, "(n_threads - 1) x src length", (n_threads - 1) * src.length);
    constexpr size_t ARGS_SIZE = sizeof(Matrix) + 4 * sizeof(Vector);
    int8_t (*args)[ARGS_SIZE] = new int8_t[n_threads - 1][ARGS_SIZE];
    buffer.length /= n_threads - 1;
    int bias_length = bias.length;
    bias.length /= n_threads;
    int weights_height = weights.height;
    weights.height /= n_threads;
    int dst_length = dst.length;
    dst.length /= n_threads;
    for (int i = 0; i < n_threads - 1; ++i) {
        PUSH_DENSE_ARGS(args[i], weights, src, dst, buffer, bias);
        RunTask(DenseWrapper, args[i]);
        buffer.data += buffer.length;
        weights.data += weights.height * weights.width;
        weights_height -= weights.height;
        dst.data += dst.length;
        dst_length -= dst.length;
        bias.data += bias.length;
        bias_length -= bias.length;
    }
    weights.height = weights_height;
    dst.length = dst_length;
    bias.length = bias_length;
    Dense(weights, src, dst, bias);
    WaitTasks();
    delete[] args;
}

#undef PUSH_DENSE_ARGS
