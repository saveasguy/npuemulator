#ifndef MATMULB_H
#define MATMULB_H

#include "Types.h"

namespace npuemulator {

void Matmul(Matrix mat1, Matrix mat2, Matrix res, Matrix mat2_buffer);

void ParallelMatmul(Matrix mat1, Matrix mat2, Matrix res, Matrix mat2_buffer);

}

#endif
