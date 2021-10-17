#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

#include "Types.h"

namespace npuemulator {

void MaxPool2D(Tensor src, int filter_height, int filter_width, Stride stride, Padding pad, Tensor res);

}

#endif
