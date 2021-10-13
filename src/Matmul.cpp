#include "Matmul.h"

#include <immintrin.h>

#include "Threads.h"

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
    int8_t *res, int res_width, int kernel_height, int kernel_width, int8_t *bias);

inline void ComputeColumn(const int8_t *mat1, int mat1_height, int mat1_width, const int8_t *reordered_mat2,
    int8_t *res, int res_width, int kernel_width, int8_t *bias)
{
    for (; mat1_height >= 4; mat1_height -=4) {
        Microkernel(mat1, mat1_width, reordered_mat2, res, res_width, 4, kernel_width, bias);
        mat1 += 4 * mat1_width;
        res += 4 * res_width;
    }
    if (mat1_height) {
        Microkernel(mat1, mat1_width, reordered_mat2, res, res_width, mat1_height, kernel_width, bias);
    }
}

void npuemulator::Matmul(Matrix mat1, Matrix mat2, Matrix res, Matrix mat2_buffer, Vector bias)
{
    if (mat1.width != mat2.height || res.width != mat2.width || mat1.height != mat1.height) {
        std::cerr << "npuemulator: Matmul: wrong sides!" << std::endl;
        exit(1);
    }
    int mat2_width_multiply32 = (mat2.width + 31) & -32;
    if (mat2_buffer.height < mat2.height || mat2_buffer.width < 2 * mat2_width_multiply32) {
        std::cerr << "npuemulator: Matmul: Not enough space for mat2_buffer!" << std::endl;
        exit(1);
    }
    ReorderMat2(mat2, mat2_buffer);
    auto i = mat2.width;
    int bias_offset = 0;
    if (bias.data) {
        if (mat2.width != bias.length) {
            std::cerr << "npuemulator: Matmul: wrong bias length!" << std::endl;
            exit(1);
        }
        bias_offset = 32;
    }
    for (; mat2.width >= 32; mat2.width -= 32) {
        ComputeColumn(mat1.data, mat1.height, mat1.width, mat2_buffer.data, res.data, res.width, 32, bias.data);
        mat2_buffer.data += 64 * mat2.height;
        res.data += 32;
        bias.data += bias_offset;
        bias.length -= bias_offset;
    }
    if (mat2.width) {
        int8_t b[32];
        if (bias.data) {
            memset(b, 0, 32);
            memcpy(b, bias.data, bias.length);
            bias.data = b;
        }
        ComputeColumn(mat1.data, mat1.height, mat1.width, mat2_buffer.data, res.data, res.width, mat2.width, bias.data);
    }
}

void MatmulWrapper(int8_t *args)
{
    constexpr size_t STRUCT_MAT_SIZE = sizeof(npuemulator::Matrix);
    constexpr size_t STRUCT_VEC_SIZE = sizeof(npuemulator::Vector);
    auto mat1 = *reinterpret_cast<npuemulator::Matrix *>(args);
    auto mat2 = *reinterpret_cast<npuemulator::Matrix *>(args + STRUCT_MAT_SIZE);
    auto res = *reinterpret_cast<npuemulator::Matrix *>(args + 2 * STRUCT_MAT_SIZE);
    auto mat2_buffer = *reinterpret_cast<npuemulator::Matrix *>(args + 3 * STRUCT_MAT_SIZE);
    auto bias = *reinterpret_cast<npuemulator::Vector *>(args + 4 * STRUCT_MAT_SIZE);
    npuemulator::Matmul(mat1, mat2, res, mat2_buffer, bias);
}

#define PUSH_MATMUL_ARGS(ARGS, MAT1, MAT2, RES, MAT2_BUFFER, BIAS)\
    *reinterpret_cast<npuemulator::Matrix *>(ARGS) = MAT1;\
    *reinterpret_cast<npuemulator::Matrix *>(ARGS + sizeof(npuemulator::Matrix)) = MAT2;\
    *reinterpret_cast<npuemulator::Matrix *>(ARGS + 2 * sizeof(npuemulator::Matrix)) = RES;\
    *reinterpret_cast<npuemulator::Matrix *>(ARGS + 3 * sizeof(npuemulator::Matrix)) = MAT2_BUFFER;\
    *reinterpret_cast<npuemulator::Vector *>(ARGS + 4 * sizeof(npuemulator::Matrix)) = BIAS;

void npuemulator::ParallelMatmul(Matrix mat1, Matrix mat2, Matrix res, Matrix mat2_buffer, Vector bias)
{
    int n_threads = NPUEMUL_THREADS.Count();
    if (mat1.height < n_threads) {
        Matmul(mat1, mat2, res, mat2_buffer);
        return;
    }
    int mat2_buffer_height = mat2_buffer.height / n_threads;
    int mat2_buffer_offset = mat2_buffer_height * mat2_buffer.width;
    int mat1_height = mat1.height;
    mat1.height /= n_threads;
    int mat1_offset = mat1.height * mat1.width;
    int res_offset = mat1.height * mat2.width;
    constexpr size_t ARGS_SIZE = 4 * sizeof(npuemulator::Matrix) + sizeof(npuemulator::Vector);
    auto (*args)[ARGS_SIZE] = new int8_t[n_threads - 1][ARGS_SIZE];
    int i = 0;
    for (; i < n_threads - 1; ++i) {
        PUSH_MATMUL_ARGS(args[i], mat1, mat2, res, mat2_buffer, bias);
        mat1.data += mat1_offset;
        res.data += res_offset;
        mat2_buffer.data += mat2_buffer_offset;
        mat2_buffer.height -= mat2_buffer_height;
        mat1_height -= mat1.height;
        NPUEMUL_THREADS.RunTask(MatmulWrapper, args[i]);
    }
    mat1.height = mat1_height;
    Matmul(mat1, mat2, res, mat2_buffer, bias);
    NPUEMUL_THREADS.WaitThreads();
    delete[] args;
}