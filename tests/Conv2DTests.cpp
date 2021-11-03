#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>

#include <Conv2D.h>
#include <Threads.h>

extern void TensorToMatrix(npuemulator::Tensor src, npuemulator::Dilation dilation, npuemulator::Padding pad, npuemulator::Stride stride,
    int8_t *matrix, int filter_height, int filter_width, int res_height, int res_width);

template <typename T>
void PutValues(T *arr, int size)
{
    for (int i = 0; i < size; ++i) {
        arr[i] = i % 256 - 128;
    }
}

void Print(uint8_t *arr, int height, int width, int channels = 1, int batches = 1)
{
    for (int b = 0; b < batches; ++b) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                for (int k = 0; k < channels; ++k) {
                    std::cout << (int)*arr++ << ' ';
                }
                std::cout << std::endl;
            }
        }
    }
}

void TestTensorToMatrix(int srcC, int srcH, int srcW,
    int kernelY, int kernelX, int dilationY, int dilationX, int strideY, int strideX,
    int padY, int padX, int padH, int padW)
{
    auto tensor = new int8_t[srcH * srcW * srcC];
    PutValues(tensor, srcH * srcW * srcC);
    npuemulator::Tensor src(tensor, srcH, srcW, srcC);
    int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
    int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
    auto matrix = new int8_t[dstH * dstW * srcC * kernelY * kernelX];
    auto ptr_mat = matrix;
    TensorToMatrix(src, {dilationY, dilationY}, {padY, padH, padX, padW}, {strideY, strideX}, matrix, kernelY, kernelX, dstH, dstW);
    for (int dy = 0; dy < dstH; ++dy)
    {
        for (int dx = 0; dx < dstW; ++dx)
        {
            for (int ky = 0; ky < kernelY; ky++)
            {
                for (int kx = 0; kx < kernelX; kx++)
                {
                    int sy = dy * strideY + ky * dilationY - padY;
                    int sx = dx * strideX + kx * dilationX - padX;
                    for (int sc = 0; sc < srcC; ++sc)
                    {
                        if (sy >= 0 && sy < srcH && sx >= 0 && sx < srcW)
                            ASSERT_EQ(*ptr_mat++, tensor[(sy * srcW + sx) * srcC + sc]);
                        else
                            ASSERT_EQ(*ptr_mat++, 0);
                    }
                }
            }
        }
    }
    delete[] matrix;
    delete[] tensor;
}

void GetParamsAndCreateTestFile(int8_t *tensor, int8_t *filter, int height, int width, int channels, int filter_height, int filter_width, int filter_channels,
    int dilation_y, int dilation_x, int stride_y, int stride_x, int pad_top, int pad_left, int pad_bot, int pad_right)
{
    auto cls = [](std::ofstream *s) {
        s->close();
    };
    std::unique_ptr<std::ofstream> test_file(new std::ofstream("tests/test_file.txt"));
    test_file->clear();
    ASSERT_TRUE((bool)*test_file);
    *test_file << height << ' ' << width << ' ' << channels << '\n';
    *test_file << filter_height << ' ' << filter_width << ' ' << filter_channels << '\n';
    *test_file << dilation_y << ' ' << dilation_x << ' ' << stride_y << ' ' << stride_x << '\n';
    *test_file << pad_top << ' ' << pad_left << ' ' << pad_bot << ' ' << pad_right << '\n';
    for (int i = 0; i < height * width * channels - 1; ++i) {
        *test_file << (int)*tensor++ << ' ';
    }
    *test_file << (int)*tensor << '\n';
    for (int i = 0; i < filter_height * filter_width * channels * filter_channels - 1; ++i) {
        *test_file << (int)*filter++ << ' ';
    }
    *test_file << (int)*filter << '\n';
}

void CheckResults(int8_t *res, int size)
{
    auto cls = [](std::ifstream *s) {
        s->close();
    };
    std::unique_ptr<std::ifstream> test_file(new std::ifstream("tests/test_file.txt"));
    ASSERT_TRUE((bool)*test_file);
    for (int i = 0; i < size; ++i) {
        int r = 0;
        *test_file >> r;
        ASSERT_EQ(*res++, (int8_t)r);
    }
}

void TestConv2D(int height, int width, int channels, int filter_height, int filter_width, int filter_channels,
    int dilation_y, int dilation_x, int stride_y, int stride_x, int pad_top, int pad_left, int pad_bot, int pad_right)
{
    auto tensor = new int8_t[height * width * channels];
    PutValues(tensor, height * width * channels);
    npuemulator::Tensor src(tensor, height, width, channels);
    auto filter = new int8_t[filter_height * filter_width * channels * filter_channels];
    npuemulator::Tensor fil(filter, filter_height, filter_width, channels, filter_channels);
    PutValues(filter, filter_height * filter_width * channels * filter_channels);
    GetParamsAndCreateTestFile(tensor, filter, height, width, channels, filter_height, filter_width, filter_channels,
        dilation_y, dilation_x, stride_y, stride_x, pad_top, pad_left, pad_bot, pad_right);
    int res_height = (height + pad_top + pad_bot - (dilation_y * (filter_height - 1) + 1)) / stride_y + 1;
    int res_width = (width + pad_left + pad_right - (dilation_x * (filter_width - 1) + 1)) / stride_x + 1;
    auto res = new int8_t[res_width * res_height * filter_channels];
    npuemulator::Tensor r(res, res_height, res_width, filter_channels);
    auto tensor_matrix = new int8_t[res_height * res_width * filter_height * filter_width * channels];
    npuemulator::Matrix src_mat(tensor_matrix, res_height * res_width, filter_height * filter_width * channels);
    int reord_height = (filter_height * filter_width * channels + 1) & -2;
    int reord_width = (filter_channels + 31) & -32;
    auto filter_reordered_mat = new int8_t[2 * npuemulator::CountThreads() * reord_height * reord_width];
    npuemulator::Matrix reord(filter_reordered_mat, npuemulator::CountThreads() * reord_height, 2 * reord_width);
    npuemulator::Conv2D(src, fil, {dilation_y, dilation_y}, {pad_top, pad_bot, pad_left, pad_right}, {stride_y, stride_x}, r, src_mat, reord);
    int python_err = std::system("python -q tests/Conv2DTest.py");
    if (python_err != 0) {
        python_err = std::system("python3 -q tests/Conv2DTest.py");
    }
    ASSERT_EQ(python_err, 0);
    CheckResults(res, res_height * res_width * filter_channels);
    std::remove("tests/test_file.txt");
    delete[] tensor;
    delete[] filter;
    delete[] res;
    delete[] tensor_matrix;
    delete[] filter_reordered_mat;
}

TEST(CONV2D_UTILS, TensorToMatrixPointwise_128x128x128)
{
    constexpr int SIZE = 128;
    TestTensorToMatrix(SIZE, SIZE, SIZE, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
}

TEST(CONV2D_UTILS, TensorToMatrix_Kernel3x3_Tensor128x128x128)
{
    constexpr int SIZE = 128;
    TestTensorToMatrix(SIZE, SIZE, SIZE, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
}

TEST(CONV2D, Conv2D_Kernel1x1_Tensor128x128x128)
{
    constexpr int SIZE = 128;
    TestConv2D(SIZE, SIZE, SIZE, 1, 1, SIZE, 1, 1, 1, 1, 0, 0, 0, 0);
}

TEST(CONV2D, Conv2D_Kernel3x3_Tensor128x128x128)
{
    constexpr int SIZE = 128;
    TestConv2D(SIZE, SIZE, SIZE, 3, 3, SIZE, 1, 1, 1, 1, 1, 1, 1, 1);
}

TEST(CONV2D, Conv2D_Kernel3x3_Tensor112x112x128)
{
    constexpr int SIZE = 128;
    TestConv2D(112, 112, SIZE, 3, 3, SIZE, 1, 1, 1, 1, 1, 1, 1, 1);
}

TEST(CONV2D, Conv2D_Kernel1x1_Tensor3x3x3)
{
    constexpr int SIZE = 3;
    TestConv2D(SIZE, SIZE, SIZE, 1, 1, SIZE, 1, 1, 1, 1, 0, 0, 0, 0);
}

TEST(CONV2D, Conv2D_Kernel3x3_Tensor3x3x3)
{
    constexpr int SIZE = 3;
    TestConv2D(SIZE, SIZE, SIZE, 3, 3, SIZE, 1, 1, 1, 1, 1, 1, 1, 1);
}
