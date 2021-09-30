#include <Types.h>

npuemulator::Matrix::Matrix(uint8_t *data_, int height_, int width_, int offset_) :
    data(data_),
    height(height_),
    width(width_),
    offset(offset_)
{
    if (offset == 0) {
        offset = width;
    }
}

npuemulator::Tensor::Tensor(uint8_t *data_, int height_, int width_, int n_channels, int n_groups = 1) :
    data(data_),
    groups(n_groups),
    height(height_),
    width(width_),
    channels(n_channels)
{}