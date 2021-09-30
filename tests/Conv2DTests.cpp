#include <gtest/gtest.h>
#include <cstdint>

#include <Conv2D.h>
#include <Threads.h>
/*
extern void ConvertGroupToMatrix(const uint8_t *tensor, int height, int width, int channels, int channels_per_group,
    int filter_height, int filter_width, int dilation_y, int dilation_x, int stride_y, int stride_x,
    int pad_top, int pad_left, uint8_t *matrix, int res_height, int res_width);

template <typename T>
void PutValues(T *arr, int size)
{
    for (int i = 0; i < size; ++i) {
        arr[i] = i % 33 + 1;
    }
}

void TestGroupToMatrix(int srcC, int srcH, int srcW,
    int kernelY, int kernelX, int dilationY, int dilationX, int strideY, int strideX,
    int padY, int padX, int padH, int padW)
{
    auto tensor = new uint8_t[srcH * srcW * srcC];
    PutValues(tensor, srcH * srcW * srcC);
    int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
    int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
    auto matrix = new uint8_t[dstH * dstW * srcC * kernelY * kernelX];
    auto ptr_matrix = matrix;
    ConvertGroupToMatrix(tensor, srcH, srcW, srcC, srcC, kernelY, kernelX, dilationY, dilationX,
        strideY, strideX, padY, padX, matrix, dstH, dstW);
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
                            *matrix++ = tensor[(sy * srcW + sx) * srcC + sc];
                        else
                            *matrix++ = 0;
                    }
                }
            }
        }
    }
    delete[] ptr_matrix;
    delete[] tensor;
}

void TestConv2D(int height, int width, int channels, int groups, int filter_height, int filter_width, int filter_channels,
    int dilation_y, int dilation_x, int stride_y, int stride_x, int pad_top, int pad_left, int pad_bot, int pad_right)
{
    auto tensor = new uint8_t[height * width * channels * groups];
    PutValues(tensor, height * width * channels * groups);
    auto filter = new uint8_t[filter_height * filter_width * channels * filter_channels / groups];
    PutValues(filter, filter_height * filter_width * channels * filter_channels);
    int res_height = (height + pad_top + pad_bot - (dilation_y * (filter_height - 1) + 1)) / stride_y + 1;
    int res_width = (width + pad_left + pad_right - (dilation_x * (filter_width - 1) + 1)) / stride_x + 1;
    auto res = new uint8_t[res_width * res_height * filter_channels / groups];
    auto tensor_matrix = new uint8_t[res_height * res_width * filter_height * filter_width * channels / groups];
    auto filter_reordered_mat = new uint8_t[NPUEMUL_THREADS.Count() * filter_height * filter_width * channels / groups * filter_channels / groups];
    npuemulator::Conv2D(tensor, height, width, channels, groups, filter, filter_height, filter_width, filter_channels,
        dilation_y, dilation_x, stride_y, stride_x, pad_top, pad_left, pad_bot, pad_right, res, tensor_matrix, filter_reordered_mat);
    channels /= groups;
    filter_channels /= groups;
    for (int g = 0; g < groups; ++g) {
        for (int dc = 0; dc < filter_channels; ++dc) {
            for (int dy = 0; dy < res_height; ++dy) {
                for (int dx = 0; dx < res_width; ++dx) {
                    uint8_t sum = 0;
                    for (int sc = 0; sc < channels; ++sc) {
                        for (int ky = 0; ky < filter_height; ky++) {
                            for (int kx = 0; kx < filter_width; kx++) {
                            int sy = dy * stride_y + ky * dilation_y - pad_top;
                            int sx = dx * stride_y + kx * dilation_x - pad_left;
                            if (sy >= 0 && sy < height && sx >= 0 && sx < width)
                                sum += tensor[((g * channels + sc) * height + sy) * width + sx] *
                                    filter[((dc * channels + sc) * filter_height + ky) * filter_width + kx];
                            }
                        }
                    }
                    if (sum != res[((g * filter_channels + dc) * res_height + dy) * res_width + dx]) {
                        std::cout << g << ' ' << dc << ' ' << dy << ' ' << dx << std::endl;
                    }
                    ASSERT_EQ(sum, res[((g * filter_channels + dc) * res_height + dy) * res_width + dx]);
                }
            }
        }
    }
    delete[] tensor;
    delete[] filter;
    delete[] res;
    delete[] tensor_matrix;
    delete[] filter_reordered_mat;
}

TEST(CONV2D_UTILS, GroupToMatrixPointwise_128x128x128)
{
    constexpr int SIZE = 128;
    TestGroupToMatrix(SIZE, SIZE, SIZE, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
}

TEST(CONV2D_UTILS, GroupToMatrix_Kernel3x3_Tensor128x128x128)
{
    constexpr int SIZE = 128;
    TestGroupToMatrix(SIZE, SIZE, SIZE, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
}

TEST(CONV2D, Conv2D_Kernel1x1_Tensor128x128x128)
{
    constexpr int SIZE = 128;
    TestConv2D(SIZE, SIZE, SIZE, 1, 1, 1, 128, 1, 1, 1, 1, 0, 0, 0, 0);
}

TEST(CONV2D, Conv2D_Kernel3x3_Tensor128x128x128)
{
    constexpr int SIZE = 128;
    TestConv2D(SIZE, SIZE, SIZE, 1, 3, 3, 128, 1, 1, 1, 1, 1, 1, 1, 1);
}
*/
