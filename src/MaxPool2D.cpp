#include "MaxPool2D.h"

#include <cstring>

#include <immintrin.h>

inline void Max(const int8_t *src, int8_t *dst, int length)
{
    for (; length >= 64; length -= 64) {
        __m256i src1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(src));
        __m256i dst1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(dst));
        __m256i src2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(src + 32));
        __m256i dst2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(dst + 32));
        src += 64;
        dst1 = _mm256_max_epi8(src1, dst1);
        dst2 = _mm256_max_epi8(src2, dst2);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), dst1);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + 32), dst2);
        dst += 64;
    }
    if (length) {
        int8_t data[128];
        memcpy(data, src, length);
        memcpy(data + 64, dst, length);
        __m256i src1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(data));
        __m256i dst1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(data + 64));
        __m256i src2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(data + 32));
        __m256i dst2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(data + 96));
        dst1 = _mm256_max_epi8(src1, dst1);
        dst2 = _mm256_max_epi8(src2, dst2);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(data), dst1);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(data + 32), dst2);
        memcpy(dst, data, length);
    }
}

void npuemulator::MaxPool2D(Tensor src, int filter_height, int filter_width, Stride stride, Padding pad, Tensor res)
{
    // TODO: ERROR IF WRONG SIDES
    int dilation_y_add_offset = src.width * src.channels;
    int dilation_x_add_offset = src.channels;
    int stride_y_add_offset = stride.y * src.width * src.channels;
    int stride_x_add_offset = stride.x * src.channels;
    pad.top *= src.width * src.channels;
    pad.left *= src.channels;
    src.height *= src.width * src.channels;
    src.width *= src.channels;
    int stride_y_offset = -pad.top;
    for (int res_y = 0; res_y < res.height; ++res_y) {
        int stride_x_offset = -pad.left;
        for (int res_x = 0; res_x < res.width; ++res_x) {
            int dilation_y_offset = 0;
            int cpy_filter_width = filter_width;
            --filter_width;
            int fy = stride_y_offset + dilation_y_offset;
            int fx = stride_x_offset;
            if (fy >= 0 && fy < src.height && fx >= 0 && fx < src.width) {
                memcpy(res.data, src.data + fy + fx, src.channels);
            }
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                int dilation_x_offset = 0;
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                    int y = stride_y_offset + dilation_y_offset;
                    int x = stride_x_offset + dilation_x_offset;
                    if (y >= 0 && y < src.height && x >= 0 && x < src.width) {
                        Max(src.data, res.data, src.channels);
                    }
                    dilation_x_offset += dilation_x_add_offset;
                }
                filter_width = cpy_filter_width;
                dilation_y_offset += dilation_y_add_offset;
            }
            res.data += res.channels;
            stride_x_offset += stride_x_add_offset;
        }
        stride_y_offset += stride_y_add_offset;
    }
}
