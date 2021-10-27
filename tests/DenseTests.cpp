#include <gtest/gtest.h>

#include <Dense.h>
#include <Types.h>

void TestDense(int height, int width)
{
    auto w = new int8_t[height * width];
    npuemulator::Matrix weights(w, height, width);
    auto s = new int8_t[width];
    npuemulator::Vector src(s, width);
    auto d = new int8_t[height];
    npuemulator::Vector dst(d, height);
    npuemulator::Dense(weights, src, dst);
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

TEST(Dense, Dense_1024x1024)
{
    TestDense(1024, 1024);
}

TEST(FC, FC_1001x1024)
{
    TestDense(1001, 1024);
}

TEST(FC, FC_1001x1001)
{
    TestDense(1001, 1001);
}