#ifndef TYPES_H
#define TYPES_H

#include <cstdint>

namespace npuemulator {

struct Matrix
{
    uint8_t *data;
    int height;
    int width;
    int offset;

public:
    Matrix(uint8_t *data_, int height_, int width_, int offset_ = 0);
};

struct Tensor
{
    uint8_t *data;
    int groups;
    int height;
    int width;
    int channels;

public:
    Tensor(uint8_t *data_, int height_, int width_, int n_channels, int n_groups = 1);
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
