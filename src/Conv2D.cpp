#include "Conv2D.h"

#include <iostream>

#include "Errors.h"
#include "Matmul.h"

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
    int src_buffer_size = src_buffer.height * src_buffer.width;
    int expected_src_buffer_size = res.height * res.width * filter.height * filter.width * src.channels;
    GreaterOrEqualOrDie("Conv2D", "src buffer size", src_buffer_size, "expected src buffer size", expected_src_buffer_size);
    TensorToMatrix(src, dilation, pad, stride, src_buffer.data, filter.height, filter.width, res.height, res.width);
    src_buffer.height = res.height * res.width;
    src_buffer.width = filter.height * filter.width * src.channels;
    Matrix res_mat(res.data, res.height * res.width, res.channels);
    Matrix filter_mat(filter.data, filter.height * filter.width * filter.batches, filter.channels);
    ParallelMatmul(src_buffer, filter_mat, res_mat, filter_buffer, bias);
}
