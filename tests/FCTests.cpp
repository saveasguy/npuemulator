#include <gtest/gtest.h>

#include <FullyConnected.h>
#include <Types.h>

void TestFC(int height, int width)
{
    auto w = new int8_t[height * width];
    npuemulator::Matrix weights(w, height, width);
    auto s = new int8_t[width];
    npuemulator::Vector src(s, width);
    auto d = new int8_t[height];
    npuemulator::Vector dst(d, height);
    npuemulator::FullyConnected(weights, src, dst);
    for (int i = 0; i < height; ++i) {
        int8_t res = 0;
        for (int j = 0; j < width; ++j) {
            res += w[i * width + j] * s[j];
        }
        ASSERT_EQ(res, d[i]);
    }
    delete[] w;
    delete[] s;
    delete[] d;
}

TEST(FC, FC_1024x1024)
{
    TestFC(1024, 1024);
}

TEST(FC, FC_1001x1024)
{
    TestFC(1001, 1024);
}

TEST(FC, FC_1001x1001)
{
    TestFC(1001, 1001);
}