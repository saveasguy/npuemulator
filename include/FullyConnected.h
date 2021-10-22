#ifndef FULLYCONNECTED_H
#define FULLYCONNECTED_H

#include "Types.h"

namespace npuemulator {

void FullyConnected(Matrix weights, Vector src, Vector dst, Vector bias = Vector(nullptr, 0));

}

#endif
