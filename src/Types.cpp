#include <Types.h>

npuemulator::Matrix::Matrix(uint8_t *data_, int height_, int width_) :
    data(data_),
    height(height_),
    width(width_)
{}

npuemulator::Tensor::Tensor(uint8_t *data_, int height_, int width_, int n_channels) :
    data(data_),
    height(height_),
    width(width_),
    channels(n_channels)
{}