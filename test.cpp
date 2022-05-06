#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <lttb.hpp>

// The fixture for testing class Foo.
class FastLTTB : public ::testing::Test {
 protected:
  FastLTTB() {}

  ~FastLTTB() override {}

  static void SetUpTestCase() {
    len = 100000000;  // 100 mil
    test_x = new double[len];
    test_y = new double[len];

    std::printf("Generating dummy data...\n");
    for (int i = 0; i < len; ++i) {
      test_x[i] = i;
      test_y[i] = (float)(std::sin(i * 0.0006) + std::sin(i * 0.005) +
                          std::sin(i * 0.04) + std::sin(i * 0.3));
    }
    std::printf("Generation done\n");
  }

  static void TearDownTestCase() {
    delete[] test_x;
    delete[] test_y;
  }

  void SetUp() override {
    out_cap = len / 8 + 2;
    out_x = new double[out_cap];
    out_y = new double[out_cap];
  }

  void TearDown() override {
    delete[] out_x;
    delete[] out_y;
  }

 public:
  static uint64_t len;
  static double *test_x, *test_y;
  uint64_t out_cap;
  double *out_x, *out_y;
};

uint64_t FastLTTB::len;
double *FastLTTB::test_x, *FastLTTB::test_y;

TEST_F(FastLTTB, Works) {
  int out_len = lttb::downsample(test_x, test_y, len, out_x, out_y, len, 1024);
}

TEST_F(FastLTTB, StillWorks) {
  int out_len = lttb::downsample(test_x, test_y, len, out_x, out_y, len, 1024);
}
