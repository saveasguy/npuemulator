#include <iostream>

#include <gtest/gtest.h>

#include <Matmul.h>
#include <Threads.h>

extern void ReorderMat2(npuemulator::Matrix mat2, npuemulator::Matrix reordered_mat2);

template <typename T>
void PutValues(T *arr, int size)
{
    for (int i = 0; i < size; ++i) {
        arr[i] = i % 33 + 1;
    }
}

void TestReorderMat2(int height, int width)
{
    uint8_t *mat = new uint8_t[height * width];
    int width_multiply32 = (width + 31) & -32;
    int height_multiply2 = (height + 1) & -2;
    uint8_t *reordered = new uint8_t[height_multiply2 * width_multiply32];
    PutValues(mat, height * width);
    npuemulator::Matrix matr(mat, height, width);
    npuemulator::Matrix reordered_matr(reordered, height_multiply2, width_multiply32);
    ReorderMat2(matr, reordered_matr);
    for (int i = 0; i < height; ++i) {
        int h_offset = i / 2 * 64;
        int w_offset = i % 2;
        for (int j = 0; j < width; ++j) {
            int n_column = j / 32;
            int n_elem = j % 32;
            if (mat[i * width + j] != reordered[32 * n_column * height_multiply2 + 2 * n_elem + h_offset + w_offset]) {
                std::cout << i << ' ' << j << std::endl;
            }
            ASSERT_EQ(mat[i * width + j], reordered[32 * n_column * height_multiply2 + 2 * n_elem + h_offset + w_offset]);
        }
    }
    delete[] mat;
    delete[] reordered;
}

void TestMatmul(int mat1_height, int mat1_width, int mat2_width)
{
    uint8_t *ptr_mat1 = new uint8_t[mat1_height * mat1_width];
    PutValues(ptr_mat1, mat1_height * mat1_width);
    npuemulator::Matrix mat1(ptr_mat1, mat1_height, mat1_width);
    uint8_t *ptr_mat2 = new uint8_t[mat1_width * mat2_width];
    PutValues(ptr_mat2, mat1_width * mat2_width);
    npuemulator::Matrix mat2(ptr_mat2, mat1_width, mat2_width);
    int width_multiply32 = (mat2_width + 31) & -32;
    int height_multiply2 = (mat1_width + 1) & -2;
    uint8_t *reordered = new uint8_t[NPUEMUL_THREADS.Count() * height_multiply2 * width_multiply32];
    npuemulator::Matrix mat2_buf(reordered, NPUEMUL_THREADS.Count() * height_multiply2, width_multiply32);
    uint8_t *ptr_res = new uint8_t[mat1_height * mat2_width];
    npuemulator::Matrix res(ptr_res, mat1_height, mat2_width);
    npuemulator::ParallelMatmul(mat1, mat2, res, mat2_buf);
    for (int i = 0; i < mat1_height; ++i) {
        for (int j = 0; j < mat2_width; ++j) {
            uint8_t val = 0;
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

TEST(MATMULB_UTILS, ReorderMat2_256x256)
{
    constexpr int SIZE = 256;
    TestReorderMat2(SIZE, SIZE);
}

TEST(MATMULB_UTILS, ReorderMat2_13x13)
{
    constexpr int SIZE = 13;
    TestReorderMat2(SIZE, SIZE);
}

TEST(MATMULB_UTILS, ReorderMat2_257x257)
{
    constexpr int SIZE = 257;
    TestReorderMat2(SIZE, SIZE);
}

TEST(MATMULB_UTILS, ReorderMat2_1025x1025)
{
    constexpr int SIZE = 1025;
    TestReorderMat2(SIZE, SIZE);
}

TEST(MATMULB_UTILS, ReorderMat2_257x256)
{
    TestReorderMat2(257, 256);
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