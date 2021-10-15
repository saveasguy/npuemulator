#ifndef CONV2D_H
#define CONV2D_H

#include "Types.h"

namespace npuemulator {

void Conv2D(Tensor src, Tensor filter, Dilation dilation, Padding pad, Stride stride, Tensor res, Matrix src_buffer, Matrix filter_buffer,
    Vector bias = Vector(nullptr, 0));

}

#endif