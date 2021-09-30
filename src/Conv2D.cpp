#include "Conv2D.h"

#include "Matmul.h"
#include "Threads.h"

void ConvertGroupToMatrix(const uint8_t *tensor, int height, int width, int channels, int channels_per_group,
    int filter_height, int filter_width, int dilation_y, int dilation_x, int stride_y, int stride_x,
    int pad_top, int pad_left, uint8_t *matrix, int res_height, int res_width)
{
    int dilation_y_add_offset = dilation_y * width * channels;
    int dilation_x_add_offset = dilation_x * channels;
    int stride_y_add_offset = stride_y * width * channels;
    int stride_x_add_offset = stride_x * channels;
    pad_top *= width * channels;
    pad_left *= channels;
    height *= width * channels;
    width *= channels;
    int stride_y_offset = -pad_top;
    for (int res_y = 0; res_y < res_height; ++res_y) {
        int stride_x_offset = -pad_left;
        for (int res_x = 0; res_x < res_width; ++res_x) {
            int dilation_y_offset = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                int dilation_x_offset = 0;
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                    int y = stride_y_offset + dilation_y_offset;
                    int x = stride_x_offset + dilation_x_offset;
                    if (y >= 0 && y < height && x >= 0 && x < width) {
                        memcpy(matrix, tensor + y + x, channels_per_group);
                        matrix += channels_per_group;
                    }
                    else {
                        memset(matrix, 0, channels_per_group);
                        matrix += channels_per_group;
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
