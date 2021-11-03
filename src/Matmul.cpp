#include "Matmul.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <immintrin.h>

#include "Errors.h"
#include "Memory.h"
#include "Threads.h"

namespace {

inline void ReorderVectors(const int8_t *src1, const int8_t *src2, int8_t *dst1, int8_t *dst2)
{
    __m128i v1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src1));
    __m128i v2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src2));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst1), _mm256_cvtepi8_epi16(v1));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst2), _mm256_cvtepi8_epi16(v2));
}

void ReorderMat2(npuemulator::Matrix mat2, npuemulator::Matrix reordered_mat2)
{
    auto i = mat2.width;
    for (; i >= 32; i -= 32, mat2.data += 32) {
        const int8_t *mat2_data = mat2.data;
        auto j = mat2.height;
        for (; j; --j) {
            ReorderVectors(mat2_data, mat2_data + 16, reordered_mat2.data, reordered_mat2.data + 32);
            reordered_mat2.data += 64;
            mat2_data += mat2.width;
        }
    }
    int8_t tmp_storage[32];
    memset(tmp_storage, 0, 32);
    if (i) {
        for (; mat2.height; --mat2.height) {
            memcpy(tmp_storage, mat2.data, i);
            ReorderVectors(tmp_storage, tmp_storage + 16, reordered_mat2.data, reordered_mat2.data + 32);
            reordered_mat2.data += 64;
            mat2.data += mat2.width;
        }
    }
}

extern "C" void Microkernel(const int8_t *mat1, int mat1_width, const int8_t *reordered_mat2,
    int8_t *res, int res_width, int8_t *bias, int kernel_height, int kernel_width, int internal_iterations);

inline void ComputeColumn(npuemulator::Matrix mat1, npuemulator::Matrix reordered_mat2,
    npuemulator::Matrix res, npuemulator::Vector bias, int kernel_width, int internal_iterations)
{
    int i = mat1.height;
    for (; i >= 4; i -=4) {
        Microkernel(mat1.data, mat1.width, reordered_mat2.data, res.data, res.width, bias.data, 4, kernel_width, internal_iterations);
        mat1.data += 4 * mat1.width;
        res.data += 4 * res.width;
    }
    if (i) {
        Microkernel(mat1.data, mat1.width, reordered_mat2.data, res.data, res.width, bias.data, i, kernel_width, internal_iterations);
    }
}

inline void Macrokernel(npuemulator::Matrix mat1, npuemulator::Matrix reordered_mat2, npuemulator::Matrix res,
    npuemulator::Vector bias, int internal_iterations)
{
    int bias_offset = bias.length ? 32 : 0;
    int i = res.width;
    for (; i >= 32; i -= 32) {
        _mm_prefetch(reinterpret_cast<const char *>(reordered_mat2.data), _MM_HINT_T0);
        ComputeColumn(mat1, reordered_mat2, res, bias, 32, internal_iterations);
        reordered_mat2.data += 64 * mat1.width;
        res.data += 32;
        bias.data += bias_offset;
    }
    if (i) {
        int8_t b[32];
        if (bias.length) {
            memset(b, 0, 32);
            memcpy(b, bias.data, i);
            bias.data = b;
        }
        ComputeColumn(mat1, reordered_mat2, res, bias, i, internal_iterations);
    }
}

}

void npuemulator::Matmul(Matrix mat1, Matrix mat2, Matrix res, Matrix mat2_buffer, Vector bias)
{
    EqualOrDie("Matmul", "mat1 width", mat1.width, "mat2 height", mat2.height);
    EqualOrDie("Matmul", "mat1 height", mat1.height, "res height", res.height);
    EqualOrDie("Matmul", "mat2 width", mat2.width, "res width", res.width);
    if (bias.length != 0) {
        EqualOrDie("Matmul", "bias length", bias.length, "res width", res.width);
    }
    GreaterZeroOrDie("Matmul", "mat1 height", mat1.height);
    GreaterZeroOrDie("Matmul", "mat1 width", mat1.width);
    GreaterZeroOrDie("Matmul", "mat2 width", mat2.width);
    int buffer_size = mat2_buffer.height * mat2_buffer.width;
    int mat2_width_multiply32 = (mat2.width + 31) & -32;
    int expected_buffer_size = 2 * mat2.height * mat2_width_multiply32;
    GreaterOrEqualOrDie("Matmul", "buffer actual size", buffer_size, "buffer expected size", expected_buffer_size);
    ReorderMat2(mat2, mat2_buffer);
    int l1_size = npuemulator::L1CacheSize();
    int step = l1_size / 64 > mat1.width ? l1_size / 64 : mat1.width;
    int i = mat1.width;
    memset(res.data, 0, res.height * res.width);
    for (; i >= step; i -= step)
    {
        Macrokernel(mat1, mat2_buffer, res, bias, step);
        mat1.data += step;
        mat2_buffer.data += 64 * step;
    }
    if (i) {
        Macrokernel(mat1, mat2_buffer, res, bias, i);
    }
}

namespace {

void MatmulWrapper(int8_t *args)
{
    constexpr size_t STRUCT_MAT_SIZE = sizeof(npuemulator::Matrix);
    constexpr size_t STRUCT_VEC_SIZE = sizeof(npuemulator::Vector);
    auto mat1 = *reinterpret_cast<npuemulator::Matrix *>(args);
    auto mat2 = *reinterpret_cast<npuemulator::Matrix *>(args + STRUCT_MAT_SIZE);
    auto res = *reinterpret_cast<npuemulator::Matrix *>(args + 2 * STRUCT_MAT_SIZE);
    auto mat2_buffer = *reinterpret_cast<npuemulator::Matrix *>(args + 3 * STRUCT_MAT_SIZE);
    auto bias = *reinterpret_cast<npuemulator::Vector *>(args + 4 * STRUCT_MAT_SIZE);
    try {
        npuemulator::Matmul(mat1, mat2, res, mat2_buffer, bias);
    }
    catch (...) {
        npuemulator::HandleThreadException(std::current_exception());
    }
}

}

#define PUSH_MATMUL_ARGS(ARGS, MAT1, MAT2, RES, MAT2_BUFFER, BIAS)\
    *reinterpret_cast<npuemulator::Matrix *>(ARGS) = MAT1;\
    *reinterpret_cast<npuemulator::Matrix *>(ARGS + sizeof(npuemulator::Matrix)) = MAT2;\
    *reinterpret_cast<npuemulator::Matrix *>(ARGS + 2 * sizeof(npuemulator::Matrix)) = RES;\
    *reinterpret_cast<npuemulator::Matrix *>(ARGS + 3 * sizeof(npuemulator::Matrix)) = MAT2_BUFFER;\
    *reinterpret_cast<npuemulator::Vector *>(ARGS + 4 * sizeof(npuemulator::Matrix)) = BIAS;

void npuemulator::ParallelMatmul(Matrix mat1, Matrix mat2, Matrix res, Matrix mat2_buffer, Vector bias)
{
    int n_threads = CountThreads();
    if (mat1.height < n_threads) {
        Matmul(mat1, mat2, res, mat2_buffer);
        return;
    }
    int buffer_size = mat2_buffer.height * mat2_buffer.width;
    int mat2_width_multiply32 = (mat2.width + 31) & -32;
    int expected_buffer_size = 2 * n_threads * mat2.height * mat2_width_multiply32;
    GreaterOrEqualOrDie("Parallel Matmul", "buffer actual size", buffer_size, "buffer expected size", expected_buffer_size);
    mat2_buffer.height = n_threads * mat2.height;
    mat2_buffer.width = 2 * mat2_width_multiply32;
    int mat2_buffer_height = mat2_buffer.height / n_threads;
    int mat1_height = mat1.height;
    res.height = mat1.height /= n_threads;
    constexpr size_t ARGS_SIZE = 4 * sizeof(npuemulator::Matrix) + sizeof(npuemulator::Vector);
    int8_t (*args)[ARGS_SIZE] = new int8_t[n_threads - 1][ARGS_SIZE];
    int i = 0;
    for (; i < n_threads - 1; ++i) {
        PUSH_MATMUL_ARGS(args[i], mat1, mat2, res, mat2_buffer, bias);
        mat1.data += mat1.height * mat1.width;
        res.data += mat1.height * mat2.width;
        mat2_buffer.data += mat2_buffer_height * mat2_buffer.width;
        mat2_buffer.height -= mat2_buffer_height;
        mat1_height -= mat1.height;
        RunTask(MatmulWrapper, args[i]);
    }
    mat1.height = mat1_height;
    res.height = mat1_height;
    Matmul(mat1, mat2, res, mat2_buffer, bias);
    WaitTasks();
    delete[] args;
}

#undef PUSH_MATMUL_ARGS
