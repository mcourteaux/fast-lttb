#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <lttb.hpp>

// The fixture for testing class Foo.
class FastLTTB : public ::testing::Test {
 protected:
  FastLTTB() {}

  ~FastLTTB() override {}

  static void SetUpTestCase() {
    len = 100000000UL;  // 100 mil
    test_x = new double[len];
    test_y = new double[len];

    std::printf("Generating dummy data...\n");
    constexpr int lut_size = 2048;
    float sin_lut[lut_size];
    for (int i = 0; i < lut_size; ++i) {
      sin_lut[i] = std::sin(float(i) / lut_size * 2 * M_PI);
    }
    for (int i = 0; i < len; ++i) {
      test_x[i] = i;
      float val = 0.0f;
      val += 1.000f * sin_lut[(i >> 8) % lut_size];
      val += 0.500f * sin_lut[(i >> 4) % lut_size];
      val += 0.250f * sin_lut[(i >> 2) % lut_size];
      val += 0.125f * sin_lut[(i >> 0) % lut_size];
      test_y[i] = val;
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

  void write_output(std::string fname, uint64_t out_len) {
    std::printf("Writing %s with %lu entries.\n", fname.c_str(), out_len);
    std::ofstream file(fname);
    for (uint64_t i = 0; i < out_len; ++i) {
      file << out_x[i] << "," << out_y[i] << "\n";
    }
    file.close();
  }

 public:
  static uint64_t len;
  static double *test_x, *test_y;
  uint64_t out_cap;
  double *out_x, *out_y;
};

uint64_t FastLTTB::len;
double *FastLTTB::test_x, *FastLTTB::test_y;

TEST_F(FastLTTB, Correct_I100_B1024) {
  int out_len = lttb::downsample(test_x, test_y, 100, out_x, out_y, len, 1024);
  ASSERT_EQ(out_len, 2);
  ASSERT_EQ(out_x[0], test_x[0]);
  ASSERT_EQ(out_x[1], test_x[99]);
  ASSERT_EQ(out_y[0], test_y[0]);
  ASSERT_EQ(out_y[1], test_y[99]);
}

TEST_F(FastLTTB, Works1K) {
  int out_len = lttb::downsample(test_x, test_y, 1000, out_x, out_y, len, 1024);
  ASSERT_EQ(out_len, 2);
}

TEST_F(FastLTTB, EndPoints) {
  int out_len =
      lttb::downsample(test_x, test_y, 10000, out_x, out_y, len, 1024);
  ASSERT_NE(out_x[0], out_x[out_len - 1]);
}

TEST_F(FastLTTB, Works1M) {
  int out_len;
  out_len = lttb::downsample(test_x, test_y, 1000000, out_x, out_y, len, 1024);
  write_output("ds_1m_1024.csv", out_len);
  out_len = lttb::downsample(test_x, test_y, 1000000, out_x, out_y, len, 10240);
  write_output("ds_1m_10240.csv", out_len);
}

TEST_F(FastLTTB, Works10M) {
  int out_len =
      lttb::downsample(test_x, test_y, 10000000, out_x, out_y, len, 1024);
  write_output("ds_10m.csv", out_len);
}

TEST_F(FastLTTB, Works100M) {
  int out_len =
      lttb::downsample(test_x, test_y, 100000000, out_x, out_y, len, 1024);
}
