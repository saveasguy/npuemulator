#include <gtest/gtest.h>

#include <ReLu.h>
#include <Threads.h>

template <typename T>
void PutValues(T *arr, int size)
{
    for (int i = 0; i < size; ++i) {
        arr[i] = i % 256 - 128;
    }
}

void TestReLu(int length)
{
    auto v1 = new int8_t[length];
    PutValues(v1, length);
    auto v2 = new int8_t[length];
    npuemulator::Vector src(v1, length);
    npuemulator::Vector dst(v2, length);
    npuemulator::ParallelReLu(src, dst);
    for (int i = 0; i < length; ++i) {
        if (v1[i] > 0) {
            ASSERT_EQ(v1[i], v2[i]);
        }
        else {
            ASSERT_EQ(0, v2[i]);
        }
    }
    delete[] v1;
    delete[] v2;
}

TEST(ReLu, ReLu_1)
{
    TestReLu(1);
}

TEST(ReLu, ReLu_1000)
{
    TestReLu(1);
}

TEST(ReLu, ReLu_1014)
{
    TestReLu(1);
}

TEST(ReLu, ReLu_15000)
{
    TestReLu(1);
}
