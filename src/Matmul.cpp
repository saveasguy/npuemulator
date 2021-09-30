#include "Matmul.h"

#include <immintrin.h>

#include "Threads.h"

inline void ReorderMat2VectorsOf32(const uint8_t *src1, const uint8_t *src2, uint8_t *dst1, uint8_t *dst2)
{
    __m128i vector1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src1));
    __m128i vector2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src2));
    __m128i vector3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src1 + 16));
    __m128i vector4 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src2 + 16));
    __m128i res1_low = _mm_unpacklo_epi8(vector1, vector2);
    __m128i res1_high = _mm_unpackhi_epi8(vector1, vector2);
    __m128i res2_low = _mm_unpacklo_epi8(vector3, vector4);
    __m128i res2_high = _mm_unpackhi_epi8(vector3, vector4);
    __m256i res1 = _mm256_permute2x128_si256(_mm256_castsi128_si256(res1_low), _mm256_castsi128_si256(res1_high), 0x20);
    __m256i res2 = _mm256_permute2x128_si256(_mm256_castsi128_si256(res2_low), _mm256_castsi128_si256(res2_high), 0x20);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst1), res1);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst2), res2);
}

inline void ReorderMat2ColumnOf32(const uint8_t *mat2, int height, int width, uint8_t *&reordered_mat2)
{
    uint8_t zero_arr[32];
    memset(zero_arr, 0, 32);
    for (; height >= 2; height -= 2, mat2 += 2 * width) {
        ReorderMat2VectorsOf32(mat2, mat2 + width, reordered_mat2, reordered_mat2 + 32);
        reordered_mat2 += 64;
    }
    if (height) {
        ReorderMat2VectorsOf32(mat2, zero_arr, reordered_mat2, reordered_mat2 + 32);
        reordered_mat2 += 64;
    }
}

inline void ReorderMat2ColumnLess32(const uint8_t *mat2, int height, int width, uint8_t *reordered_mat2, int length)
{
    uint8_t vectors_arr[96];
    memset(vectors_arr, 0, 96);
    for (; height >= 2; height -= 2, mat2 += 2 * width) {
        memcpy(vectors_arr, mat2, length);
        memcpy(vectors_arr + 32, mat2 + width, length);
        ReorderMat2VectorsOf32(vectors_arr, vectors_arr + 32, reordered_mat2, reordered_mat2 + 32);
        reordered_mat2 += 64;
    }
    if (height) {
        memcpy(vectors_arr, mat2, length);
        ReorderMat2VectorsOf32(vectors_arr, vectors_arr + 64, reordered_mat2, reordered_mat2 + 32);
        reordered_mat2 += 64;
    }
}

void ReorderMat2(npuemulator::Matrix mat2, npuemulator::Matrix reordered_mat2)
{
    auto i = mat2.width;
    for (; i >= 32; i -= 32, mat2.data += 32) {
        ReorderMat2ColumnOf32(mat2.data, mat2.height, mat2.width, reordered_mat2.data);
    }
    if (i) {
        ReorderMat2ColumnLess32(mat2.data, mat2.height, mat2.width, reordered_mat2.data, i);
    }
}

/*
void dMicrokernel(const uint8_t *mat1, int mat1_width, const uint8_t *mat2, int mat2_width, uint8_t *res)
{
    __m256i accumulator0 = _mm256_setzero_si256();
    __m256i accumulator1 = _mm256_setzero_si256();
    __m256i accumulator2 = _mm256_setzero_si256();
    __m256i accumulator3 = _mm256_setzero_si256();
    __m256i accumulator4 = _mm256_setzero_si256();
    __m256i accumulator5 = _mm256_setzero_si256();
    __m256i accumulator6 = _mm256_setzero_si256();
    __m256i accumulator7 = _mm256_setzero_si256();
    for (auto i = mat1_width; i >= 2; i -= 2, mat2 += 64, mat1 += 2) {
        __m256i vector1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(mat2));
        __m256i vector2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(mat2 + 32));
        __m256i val1 = _mm256_set1_epi16(*reinterpret_cast<const short *>(mat1));
        __m256i val2 = _mm256_set1_epi16(*reinterpret_cast<const short *>(mat1 + mat1_width));
        __m256i val3 = _mm256_set1_epi16(*reinterpret_cast<const short *>(mat1 + 2 * mat1_width));
        __m256i val4 = _mm256_set1_epi16(*reinterpret_cast<const short *>(mat1 + 3 * mat1_width));
        __m256i res1 = _mm256_maddubs_epi16(vector1, val1);
        val1 = _mm256_maddubs_epi16(vector2, val1);
        accumulator0 = _mm256_add_epi16(accumulator0, res1);
        accumulator1 = _mm256_add_epi16(accumulator1, val1);
        __m256i res2 = _mm256_maddubs_epi16(vector1, val2);
        val2 = _mm256_maddubs_epi16(vector2, val2);
        accumulator2 = _mm256_add_epi16(accumulator2, res2);
        accumulator3 = _mm256_add_epi16(accumulator3, val2);
        res1 = _mm256_maddubs_epi16(vector1, val3);
        val3 = _mm256_maddubs_epi16(vector2, val3);
        accumulator4 = _mm256_add_epi16(accumulator4, res1);
        accumulator5 = _mm256_add_epi16(accumulator5, val3);
        vector1 = _mm256_maddubs_epi16(vector1, val4);
        vector2 = _mm256_maddubs_epi16(vector2, val4);
        accumulator6 = _mm256_add_epi16(accumulator6, vector1);
        accumulator7 = _mm256_add_epi16(accumulator7, vector1);
    }
    _mm_storeu_si128(reinterpret_cast<__m128i *>(res), _mm256_castsi256_si128(accumulator0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(res + 16), _mm256_castsi256_si128(accumulator1));
    res += mat2_width;
    _mm_storeu_si128(reinterpret_cast<__m128i *>(res), _mm256_castsi256_si128(accumulator2));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(res + 16), _mm256_castsi256_si128(accumulator3));
    res += mat2_width;
    _mm_storeu_si128(reinterpret_cast<__m128i *>(res), _mm256_castsi256_si128(accumulator4));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(res + 16), _mm256_castsi256_si128(accumulator5));
    res += mat2_width;
    _mm_storeu_si128(reinterpret_cast<__m128i *>(res), _mm256_castsi256_si128(accumulator6));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(res + 16), _mm256_castsi256_si128(accumulator7));
}
*/

extern "C" void Microkernel(const uint8_t *mat1, int mat1_width, const uint8_t *reordered_mat2,
    uint8_t *res, int res_width, int kernel_height, int kernel_width);

inline void ComputeColumn(const uint8_t *mat1, int mat1_height, int mat1_width, const uint8_t *reordered_mat2,
    uint8_t *res, int res_width, int kernel_width)
{
    for (; mat1_height >= 4; mat1_height -=4) {
        Microkernel(mat1, mat1_width, reordered_mat2, res, res_width, 4, kernel_width);
        mat1 += 4 * mat1_width;
        res += 4 * res_width;
    }
    if (mat1_height) {
        Microkernel(mat1, mat1_width, reordered_mat2, res, res_width, mat1_height, kernel_width);
    }
}

void npuemulator::Matmul(Matrix mat1, Matrix mat2, Matrix res, Matrix mat2_buffer)
{
    if (mat1.width != mat2.height || res.width != mat2.width || mat1.height != mat1.height) {
        std::cerr << "npuemulator: Matmul: wrong sides!" << std::endl;
        exit(1);
    }
    int mat2_height_multiply2 = (mat2.height + 1) & -2;
    int mat2_width_multiply32 = (mat2.width + 31) & -32;
    if (mat2_buffer.height < mat2_height_multiply2 || mat2_buffer.width < mat2_width_multiply32) {
        std::cerr << "npuemulator: Matmul: Not enough space for mat2_buffer!" << std::endl;
        exit(1);
    }
    ReorderMat2(mat2, mat2_buffer);
    auto i = mat2.width;
    for (; mat2.width >= 32; mat2.width -= 32) {
        ComputeColumn(mat1.data, mat1.height, mat1.width, mat2_buffer.data, res.data, res.width, 32);
        mat2_buffer.data += 32 * mat2_height_multiply2;
        res.data += 32;
    }
    if (mat2.width) {
        ComputeColumn(mat1.data, mat1.height, mat1.width, mat2_buffer.data, res.data, res.width, mat2.width);
    }
}

void MatmulAsync(uint8_t *args)
{
    constexpr size_t STRUCT_SIZE = sizeof(npuemulator::Matrix);
    auto mat1 = *reinterpret_cast<npuemulator::Matrix *>(args);
    auto mat2 = *reinterpret_cast<npuemulator::Matrix *>(args + STRUCT_SIZE);
    auto res = *reinterpret_cast<npuemulator::Matrix *>(args + 2 * STRUCT_SIZE);
    auto mat2_buffer = *reinterpret_cast<npuemulator::Matrix *>(args + 3 * STRUCT_SIZE);
    npuemulator::Matmul(mat1, mat2, res, mat2_buffer);
}

#define PUSH_MATMUL_ARGS(ARGS, MAT1, MAT2, RES, MAT2_BUFFER) \
    *reinterpret_cast<npuemulator::Matrix *>(ARGS) = MAT1; \
    *reinterpret_cast<npuemulator::Matrix *>(ARGS + sizeof(npuemulator::Matrix)) = MAT2; \
    *reinterpret_cast<npuemulator::Matrix *>(ARGS + 2 * sizeof(npuemulator::Matrix)) = RES; \
    *reinterpret_cast<npuemulator::Matrix *>(ARGS + 3 * sizeof(npuemulator::Matrix)) = MAT2_BUFFER;

void npuemulator::ParallelMatmul(Matrix mat1, Matrix mat2, Matrix res, Matrix mat2_buffer)
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
    constexpr size_t ARGS_SIZE = 4 * sizeof(npuemulator::Matrix);
    uint8_t (*args)[ARGS_SIZE] = new uint8_t[n_threads - 1][ARGS_SIZE];
    int i = 0;
    for (; i < n_threads - 1; ++i) {
        PUSH_MATMUL_ARGS(args[i], mat1, mat2, res, mat2_buffer);
        mat1.data += mat1_offset;
        res.data += res_offset;
        mat2_buffer.data += mat2_buffer_offset;
        mat2_buffer.height -= mat2_buffer_height;
        mat1_height -= mat1.height;
        NPUEMUL_THREADS.RunTask(MatmulAsync, args[i]);
    }
    mat1.height = mat1_height;
    Matmul(mat1, mat2, res, mat2_buffer);
    NPUEMUL_THREADS.WaitThreads();
    delete[] args;
}