#ifndef RELU_H
#define RELU_H

#include "Types.h"

namespace npuemulator {

void ReLu(Vector src, Vector dst);

void ParallelReLu(Vector src, Vector dst);

}

#endif
