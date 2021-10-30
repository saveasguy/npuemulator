#include <iostream>

#include <gtest/gtest.h>

#include <Matmul.h>
#include <Threads.h>

template <typename T>
void PutValues(T *arr, int size)
{
    for (int i = 0; i < size; ++i) {
        arr[i] = i % 256 - 128;
    }
}

#include <immintrin.h>

void TestMatmul(int mat1_height, int mat1_width, int mat2_width, bool use_bias = false)
{
    int8_t *ptr_mat1 = new int8_t[mat1_height * mat1_width];
    PutValues(ptr_mat1, mat1_height * mat1_width);
    npuemulator::Matrix mat1(ptr_mat1, mat1_height, mat1_width);
    int8_t *ptr_mat2 = new int8_t[mat1_width * mat2_width];
    PutValues(ptr_mat2, mat1_width * mat2_width);
    npuemulator::Matrix mat2(ptr_mat2, mat1_width, mat2_width);
    int width_multiply32 = (mat2_width + 31) & -32;
    int8_t *reordered = new int8_t[2 * NPUEMUL_THREADS.Count() * mat1_width * width_multiply32];
    npuemulator::Matrix mat2_buf(reordered, NPUEMUL_THREADS.Count() * mat1_width, 2 *  width_multiply32);
    int8_t *ptr_res = new int8_t[mat1_height * mat2_width];
    npuemulator::Matrix res(ptr_res, mat1_height, mat2_width);
    npuemulator::Vector bias(nullptr, 0);
    if (use_bias) {
        bias.data = new int8_t[mat2_width];
        bias.length = mat2_width;
        PutValues(bias.data, bias.length);
    }
    npuemulator::ParallelMatmul(mat1, mat2, res, mat2_buf, bias);
    for (int i = 0; i < mat1_height; ++i) {
        for (int j = 0; j < mat2_width; ++j) {
            int8_t val = 0;
            if (use_bias) {
                val = bias.data[j];
            }
            for (int k = 0; k < mat1_width; ++k) {
                val += mat1.data[i * mat1_width + k] * mat2.data[j + k * mat2_width];
            }
            if (ptr_res[i * mat2_width + j] != val) {
                std::cout << i << ' ' << j << '\n';
            }
            ASSERT_EQ(res.data[i * mat2_width + j], val);
        }
    }
    delete[] ptr_mat1;
    delete[] ptr_mat2;
    delete[] reordered;
    delete[] ptr_res;
}

TEST(MATMULB, Matmul4x4x32)
{
    TestMatmul(4, 4, 32);
}

TEST(MATMULB, Matmul256x256x256)
{
    constexpr int SIZE = 256;
    TestMatmul(SIZE, SIZE, SIZE);
}

TEST(MATMULB, Matmul259x256x256)
{
    constexpr int SIZE = 256;
    constexpr int SIZE2 = 259;
    TestMatmul(SIZE2, SIZE, SIZE);
}

TEST(MATMULB, Matmul256x257x256)
{
    constexpr int SIZE = 256;
    constexpr int SIZE2 = 257;
    TestMatmul(SIZE, SIZE2, SIZE);
}

TEST(MATMULB, Matmul257x257x256)
{
    constexpr int SIZE = 256;
    constexpr int SIZE2 = 257;
    TestMatmul(SIZE2, SIZE2, SIZE);
}

TEST(MATMULB, Matmul258x257x256)
{
    constexpr int SIZE = 256;
    constexpr int SIZE1 = 258;
    constexpr int SIZE2 = 257;
    TestMatmul(SIZE1, SIZE2, SIZE);
}

TEST(MATMULB, Matmul256x256x259)
{
    constexpr int SIZE = 256;
    constexpr int SIZE2 = 259;
    TestMatmul(SIZE, SIZE, SIZE2);
}

TEST(MATMULB, Matmul258x257x259)
{
    constexpr int SIZE = 259;
    constexpr int SIZE1 = 258;
    constexpr int SIZE2 = 257;
    TestMatmul(SIZE1, SIZE2, SIZE);
}

TEST(MATMULB, Matmul256x257x287)
{
    constexpr int SIZE = 287;
    constexpr int SIZE1 = 256;
    constexpr int SIZE2 = 257;
    TestMatmul(SIZE1, SIZE2, SIZE);
}

TEST(MATMULB, Matmul259x257x287)
{
    constexpr int SIZE = 287;
    constexpr int SIZE1 = 259;
    constexpr int SIZE2 = 257;
    TestMatmul(SIZE1, SIZE2, SIZE);
}

TEST(MATMULB, Matmul3x4x5)
{
    constexpr int SIZE = 5;
    constexpr int SIZE1 = 3;
    constexpr int SIZE2 = 4;
    TestMatmul(SIZE1, SIZE2, SIZE);
}

TEST(MATMULB, Matmul1024x1025x1025)
{
    constexpr int SIZE = 1024;
    constexpr int SIZE1 = 1025;
    constexpr int SIZE2 = 1025;
    TestMatmul(SIZE1, SIZE2, SIZE);
}

TEST(MATMULB, Matmul256x256x256_with_bias)
{
    constexpr int SIZE = 256;
    constexpr int SIZE1 = 256;
    constexpr int SIZE2 = 256;
    TestMatmul(SIZE1, SIZE2, SIZE, true);
}