#include <benchmark/benchmark.h>

#include <Conv2D.h>
#include <Dense.h>
#include <MaxPool2D.h>
#include <Types.h>
#include <ReLu.h>
#include <Threads.h>

constexpr size_t MAX_TENSOR_SIZE = 224 * 224 * 64;
constexpr size_t MAX_FILTER_SIZE = 3 * 3 * 512 * 512;
static int8_t *src = new int8_t[MAX_TENSOR_SIZE];
static int8_t *dst = new int8_t[MAX_TENSOR_SIZE];
static int8_t *src_buffer = new int8_t[MAX_TENSOR_SIZE * 3 * 3];
static int8_t *filter_buffer = new int8_t[npuemulator::CountThreads() * 2 * MAX_FILTER_SIZE];
static int8_t *filter1 = new int8_t[3 * 3 * 3 * 64];
static int8_t *filter2 = new int8_t[3 * 3 * 64 * 64];
static int8_t *filter3 = new int8_t[3 * 3 * 64 * 128];
static int8_t *filter4 = new int8_t[3 * 3 * 128 * 128];
static int8_t *filter5 = new int8_t[3 * 3 * 128 * 256];
static int8_t *filter6 = new int8_t[3 * 3 * 256 * 256];
static int8_t *filter7 = new int8_t[3 * 3 * 256 * 256];
static int8_t *filter8 = new int8_t[3 * 3 * 256 * 512];
static int8_t *filter9 = new int8_t[3 * 3 * 512 * 512];
static int8_t *filter10 = new int8_t[3 * 3 * 512 * 512];
static int8_t *filter11 = new int8_t[3 * 3 * 512 * 512];
static int8_t *filter12 = new int8_t[3 * 3 * 512 * 512];
static int8_t *filter13 = new int8_t[3 * 3 * 512 * 512];
static int8_t *weights1 = new int8_t[7 * 7 * 512 * 4096];
static int8_t *weights2 = new int8_t[4096 * 4096];
static int8_t *weights3 = new int8_t[4096 * 1000];

using namespace npuemulator;
void vgg16()
{
    constexpr int SRC_BUFFFER_HEIGHT = 224 * 224, SRC_BUFFER_WIDTH = 3 * 3 * 64;
    int SRC_FILTER_HEIGHT = npuemulator::CountThreads() * 3 * 3 * 512, SRC_FILTER_WIDTH = 2 * 512;
    Conv2D({src, 224, 224, 3}, {filter1, 3, 3, 64, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {dst, 224, 224, 64},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({dst, 224 * 224 * 64}, {src, 224 * 224 * 64});
    Conv2D({src, 224, 224, 64}, {filter2, 3, 3, 64, 64}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {dst, 224, 224, 64},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({dst, 224 * 224 * 64}, {src, 224 * 224 * 64});
    MaxPool2D({src, 224, 224, 64}, 2, 2, {2, 2}, {0, 0, 0, 0}, {dst, 112, 112, 64});
    Conv2D({dst, 112, 112, 64}, {filter3, 3, 3, 128, 64}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {src, 112, 112, 128},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({src, 112 * 112 * 128}, {dst, 112 * 112 * 128});
    Conv2D({dst, 112, 112, 128}, {filter4, 3, 3, 128, 128}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {src, 112, 112, 128},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({src, 112 * 112 * 128}, {dst, 112 * 112 * 128});
    MaxPool2D({dst, 112, 112, 128}, 2, 2, {2, 2}, {0, 0, 0, 0}, {src, 56, 56, 128});
    Conv2D({src, 56, 56, 128}, {filter5, 3, 3, 256, 128}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {dst, 56, 56, 256},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({dst, 56 * 56 * 256}, {src, 56 * 56 * 256});
    Conv2D({src, 56, 56, 256}, {filter6, 3, 3, 256, 256}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {dst, 56, 56, 256},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({dst, 56 * 56 * 256}, {src, 56 * 56 * 256});
    Conv2D({src, 56, 56, 256}, {filter7, 3, 3, 256, 256}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {dst, 56, 56, 256},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({dst, 56 * 56 * 256}, {src, 56 * 56 * 256});
    MaxPool2D({src, 56, 56, 256}, 2, 2, {2, 2}, {0, 0, 0, 0}, {dst, 28, 28, 256});
    Conv2D({dst, 28, 28, 256}, {filter8, 3, 3, 512, 256}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {src, 28, 28, 512},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({src, 28 * 28 * 512}, {dst, 28 * 28 * 512});
    Conv2D({dst, 28, 28, 512}, {filter9, 3, 3, 512, 512}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {src, 28, 28, 512},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({src, 28 * 28 * 512}, {dst, 28 * 28 * 512});
    Conv2D({dst, 28, 28, 512}, {filter10, 3, 3, 512, 512}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {src, 28, 28, 512},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({src, 28 * 28 * 512}, {dst, 28 * 28 * 512});
    MaxPool2D({dst, 28, 28, 512}, 2, 2, {2, 2}, {0, 0, 0, 0}, {src, 14, 14, 512});
    Conv2D({src, 14, 14, 512}, {filter11, 3, 3, 512, 512}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {dst, 14, 14, 512},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({dst, 14 * 14 * 512}, {src, 14 * 14 * 512});
    Conv2D({src, 14, 14, 512}, {filter12, 3, 3, 512, 512}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {dst, 14, 14, 512},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({dst, 14 * 14 * 512}, {src, 14 * 14 * 512});
    Conv2D({src, 14, 14, 512}, {filter13, 3, 3, 512, 512}, {1, 1}, {1, 1, 1, 1}, {1, 1}, {dst, 14, 14, 512},
        {src_buffer, SRC_BUFFFER_HEIGHT, SRC_BUFFER_WIDTH}, {filter_buffer, SRC_FILTER_HEIGHT, SRC_FILTER_WIDTH});
    ReLu({dst, 14 * 14 * 512}, {src, 14 * 14 * 512});
    MaxPool2D({src, 14, 14, 512}, 2, 2, {2, 2}, {0, 0, 0, 0}, {dst, 7, 7, 512});
    Dense({weights1, 4096, 7 * 7 * 512}, {dst, 7 * 7 * 512}, {src, 4096});
    ReLu({src, 4096}, {dst, 4096});
    Dense({weights2, 4096, 4096}, {dst, 4096}, {src, 4096});
    ReLu({src, 4096}, {dst, 4096});
    Dense({weights3, 1000, 4096}, {dst, 4096}, {src, 1000});
    ReLu({src, 1000}, {dst, 1000});
}

static void BM_VGG16(benchmark::State &state)
{
    vgg16();
    vgg16();
    for (auto _ : state) {
        vgg16();
    }
}
BENCHMARK(BM_VGG16)->Iterations(2)->Unit(benchmark::TimeUnit::kMillisecond)->Repetitions(5);
