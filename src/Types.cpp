#include <Types.h>

npuemulator::Matrix::Matrix(uint8_t *data_, int height_, int width_) :
    data(data_),
    height(height_),
    width(width_)
{}

npuemulator::Tensor::Tensor(uint8_t *data_, int height_, int width_, int n_channels, int n_batches) :
    data(data_),
    batches(n_batches),
    height(height_),
    width(width_),
    channels(n_channels)
{}