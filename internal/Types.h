#ifndef TYPES_H
#define TYPES_H

#include <cstdint>

namespace npuemulator {

struct Matrix
{
    uint8_t *data;
    int height;
    int width;

public:
    Matrix(uint8_t *data_, int height_, int width_);
};

struct Tensor
{
    uint8_t *data;
    int batches;
    int height;
    int width;
    int channels;

public:
    Tensor(uint8_t *data_, int height_, int width_, int n_channels, int n_batches = 1);
};

struct Dilation
{
    int x, y;
};

struct Stride
{
    int x, y;
};

struct Padding
{
    int bot, top;
    int left, right;
};

}

#endif
