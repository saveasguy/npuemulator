#ifndef DENSE_H
#define DENSE_H

#include "Types.h"

namespace npuemulator {

void Dense(Matrix weights, Vector src, Vector dst, Vector bias = Vector(nullptr, 0));

}

#endif
