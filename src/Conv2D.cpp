#include "Conv2D.h"

#include "Matmul.h"
#include "Threads.h"

void TensorToMatrix(npuemulator::Tensor src, npuemulator::Dilation dilation, npuemulator::Padding pad, npuemulator::Stride stride,
    int8_t *matrix, int filter_height, int filter_width, int res_height, int res_width)
{
    int dilation_y_add_offset = dilation.y * src.width * src.channels;
    int dilation_x_add_offset = dilation.x * src.channels;
    int stride_y_add_offset = stride.y * src.width * src.channels;
    int stride_x_add_offset = stride.x * src.channels;
    pad.top *= src.width * src.channels;
    pad.left *= src.channels;
    src.height *= src.width * src.channels;
    src.width *= src.channels;
    int stride_y_offset = -pad.top;
    for (int res_y = 0; res_y < res_height; ++res_y) {
        int stride_x_offset = -pad.left;
        for (int res_x = 0; res_x < res_width; ++res_x) {
            int dilation_y_offset = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                int dilation_x_offset = 0;
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                    int y = stride_y_offset + dilation_y_offset;
                    int x = stride_x_offset + dilation_x_offset;
                    if (y >= 0 && y < src.height && x >= 0 && x < src.width) {
                        memcpy(matrix, src.data + y + x, src.channels);
                        matrix += src.channels;
                    }
                    else {
                        memset(matrix, 0, src.channels);
                        matrix += src.channels;
                    }
                    dilation_x_offset += dilation_x_add_offset;
                }
                dilation_y_offset += dilation_y_add_offset;
            }
            stride_x_offset += stride_x_add_offset;
        }
        stride_y_offset += stride_y_add_offset;
    }
}

void npuemulator::Conv2D(Tensor src, Tensor filter, Dilation dilation, Padding pad, Stride stride, Tensor res, Matrix src_buffer, Matrix filter_buffer,
    Vector bias)
{
    if (src_buffer.height < res.height * res.width || src_buffer.width < filter.height * filter.width * src.channels) {

        std::cerr << "npuemulator: Conv2D: Not enough space for src_buffer!" << std::endl;
        exit(1);
    }
    TensorToMatrix(src, dilation, pad, stride, src_buffer.data, filter.height, filter.width, res.height, res.width);
    Matrix res_mat(res.data, res.height * res.width, res.channels);
    Matrix filter_mat(filter.data, filter.height * filter.width * filter.batches, filter.channels);
    ParallelMatmul(src_buffer, filter_mat, res_mat, filter_buffer, bias);
}

/*
#define TENSOR_SIZE sizeof(npuemulator::Tensor)
#define DILATION_SIZE sizeof(npuemulator::Dilation)
#define PAD_SIZE sizeof(npuemulator::Padding)
#define STRIDE_SIZE sizeof(npuemulator::Stride)
#define MATRIX_SIZE sizeof(npuemulator::Matrix)
#define VECTOR_SIZE sizeof(npuemulator::Vector)

void Conv2DWrapper(int8_t *args)
{
    auto src = *reinterpret_cast<npuemulator::Tensor *>(args);
    auto filter = *reinterpret_cast<npuemulator::Tensor *>(args + TENSOR_SIZE);
    auto dilation = *reinterpret_cast<npuemulator::Dilation *>(args + 2 * TENSOR_SIZE);
    auto pad = *reinterpret_cast<npuemulator::Padding *>(args + 2 * TENSOR_SIZE + DILATION_SIZE);
    auto stride = *reinterpret_cast<npuemulator::Stride *>(args + 2 * TENSOR_SIZE + DILATION_SIZE + PAD_SIZE);
    auto res = *reinterpret_cast<npuemulator::Tensor *>(args + 2 * TENSOR_SIZE + DILATION_SIZE + PAD_SIZE + STRIDE_SIZE);
    auto src_buffer = *reinterpret_cast<npuemulator::Matrix *>(args + 3 * TENSOR_SIZE + DILATION_SIZE + PAD_SIZE + STRIDE_SIZE);
    auto filter_buffer = *reinterpret_cast<npuemulator::Matrix *>(args + 3 * TENSOR_SIZE + DILATION_SIZE + PAD_SIZE + STRIDE_SIZE + MATRIX_SIZE);
    auto bias = *reinterpret_cast<npuemulator::Vector *>(args + 3 * TENSOR_SIZE + DILATION_SIZE + PAD_SIZE + STRIDE_SIZE + 2 * MATRIX_SIZE);
    npuemulator::Conv2D(src, filter, dilation, pad, stride, res, src_buffer, filter_buffer, bias);
}

#define PUSH_CONV2D_ARGS(ARGS, SRC, FILTER, DILATION, PAD, STRIDE, RES, SRC_BUFFER, FILTER_BUFFER, BIAS)\
    *reinterpret_cast<npuemulator::Tensor *>(ARGS) = SRC;\
    *reinterpret_cast<npuemulator::Tensor *>(ARGS + TENSOR_SIZE) = FILTER;\
    *reinterpret_cast<npuemulator::Dilation *>(ARGS + 2 * TENSOR_SIZE) = DILATION;\
    *reinterpret_cast<npuemulator::Padding *>(ARGS + 2 * TENSOR_SIZE + DILATION_SIZE) = PAD;\
    *reinterpret_cast<npuemulator::Stride *>(ARGS + 2 * TENSOR_SIZE + DILATION_SIZE + PAD_SIZE) = STRIDE;\
    *reinterpret_cast<npuemulator::Tensor *>(ARGS + 2 * TENSOR_SIZE + DILATION_SIZE + PAD_SIZE + STRIDE_SIZE) = RES;\
    *reinterpret_cast<npuemulator::Matrix *>(ARGS + 3 * TENSOR_SIZE + DILATION_SIZE + PAD_SIZE + STRIDE_SIZE) = SRC_BUFFER;\
    *reinterpret_cast<npuemulator::Matrix *>(ARGS + 3 * TENSOR_SIZE + DILATION_SIZE + PAD_SIZE + STRIDE_SIZE + MATRIX_SIZE) = FILTER_BUFFER;\
    *reinterpret_cast<npuemulator::Vector *>(ARGS + 3 * TENSOR_SIZE + DILATION_SIZE + PAD_SIZE + STRIDE_SIZE + 2 * MATRIX_SIZE) = BIAS;

void npuemulator::ParallelConv2D(Tensor src, Tensor filter, Dilation dilation, Padding pad, Stride stride, Tensor res, Matrix src_buffer, Matrix filter_buffer,
    Vector bias)
{
    int n_threads = NPUEMUL_THREADS.Count();
    if (res.height < n_threads) {
        Conv2D(src, filter, dilation, pad, stride, res, src_buffer, filter_buffer, bias);
        return;
    }
    constexpr size_t ARGS_SIZE = 3 * TENSOR_SIZE + DILATION_SIZE + PAD_SIZE + STRIDE_SIZE + 2 * MATRIX_SIZE + VECTOR_SIZE;
    auto (*args)[ARGS_SIZE] = new int8_t[n_threads - 1][ARGS_SIZE];
    int res_height = res.height;
    res.height /= n_threads;
    int src_buffer_height = src_buffer.height;
    src_buffer.height /= n_threads;
    int res_offset = res.height * res.width *  res.channels;
    int src_buffer_offset = res.height * res.width * src.channels * filter.width * filter.height;
    int filter_buffer_offset = src.channels * filter.width * filter.height * res.channels;
    int filter_buffer_height = filter_buffer.height;
    filter_buffer.height /= n_threads;
    int i = 0;
    if (i < n_threads - 1) {
        PUSH_CONV2D_ARGS(args[i], src, filter, dilation, pad, stride, res, src_buffer, filter_buffer, bias);
        NPUEMUL_THREADS.RunTask(Conv2DWrapper, args[i]);
        res_height -= res.height;
        src_buffer_height -= src_buffer.height;
        filter_buffer_height -= filter_buffer.height;
        res.data += res_offset;
        src_buffer.data += src_buffer_offset;
        filter_buffer.data += filter_buffer_offset;
        pad.top = 0;
        ++i;
    }
    for (; i < n_threads - 1; ++i) {
        PUSH_CONV2D_ARGS(args[i], src, filter, dilation, pad, stride, res, src_buffer, filter_buffer, bias);
        NPUEMUL_THREADS.RunTask(Conv2DWrapper, args[i]);
        res_height -= res.height;
        src_buffer_height -= src_buffer.height;
        filter_buffer_height -= filter_buffer.height;
        res.data += res_offset;
        src_buffer.data += src_buffer_offset;
        filter_buffer.data += filter_buffer_offset;
    }
    res.height = res_height;
    src_buffer.height = src_buffer_height;
    filter_buffer.height = filter_buffer_height;
    Conv2D(src, filter, dilation, pad, stride, res, src_buffer, filter_buffer, bias);
    NPUEMUL_THREADS.WaitThreads();
    delete[] args;
}
*/