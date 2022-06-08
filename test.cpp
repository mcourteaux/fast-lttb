#include <gtest/gtest.h>

#include <Tracy.hpp>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <lttb.hpp>

template <typename T>
class FastLTTB : public ::testing::Test {
 protected:
  FastLTTB() {}

  ~FastLTTB() override {}

  static void SetUpTestCase() {
    ZoneScoped;
    len = 100000000UL;  // 100 mil
    test_x = new T[len];
    test_y = new T[len];

    std::printf("Generating dummy f%d data...\n", int(sizeof(T) * 8));
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
    out_x = new T[out_cap];
    out_y = new T[out_cap];
  }

  void TearDown() override {
    delete[] out_x;
    delete[] out_y;
  }

  void write_output(std::string fname, uint64_t out_len, uint64_t offset = 0) {
    ZoneScoped;
    std::printf("Writing %s with %lu entries.\n", fname.c_str(), out_len);
    std::ofstream file(fname);
    for (uint64_t i = 0; i < out_len; ++i) {
      file << out_x[i + offset] << "," << out_y[i + offset] << "\n";
    }
    file.close();
  }

 public:
  static uint64_t len;
  static T *test_x, *test_y;

  uint64_t out_cap;
  T *out_x, *out_y;
};

template <typename T>
uint64_t FastLTTB<T>::len;
template <typename T>
T *FastLTTB<T>::test_x;
template <typename T>
T *FastLTTB<T>::test_y;

typedef ::testing::Types<float, double> FloatTypes;
TYPED_TEST_SUITE(FastLTTB, FloatTypes);

TYPED_TEST(FastLTTB, Correct_I100_B1024_Scalar) {
  auto test_x = this->test_x;
  auto test_y = this->test_y;
  auto out_x = this->out_x;
  auto out_y = this->out_y;
  auto len = this->len;

  int out_len = lttb::downsample(test_x, test_y, 100, out_x, out_y, len, 1024);
  ASSERT_EQ(out_len, 2);
  ASSERT_EQ(out_x[0], test_x[0]);
  ASSERT_EQ(out_x[1], test_x[99]);
  ASSERT_EQ(out_y[0], test_y[0]);
  ASSERT_EQ(out_y[1], test_y[99]);
}

TYPED_TEST(FastLTTB, Correct_I100_B1024_SIMD) {
  auto test_x = this->test_x;
  auto test_y = this->test_y;
  auto out_x = this->out_x;
  auto out_y = this->out_y;
  auto len = this->len;

  int out_len =
      lttb::downsample_simd(test_x, test_y, 100, out_x, out_y, len, 1024);
  ASSERT_EQ(out_len, 2);
  ASSERT_EQ(out_x[0], test_x[0]);
  ASSERT_EQ(out_x[1], test_x[99]);
  ASSERT_EQ(out_y[0], test_y[0]);
  ASSERT_EQ(out_y[1], test_y[99]);
}

TYPED_TEST(FastLTTB, EndPoints_Scalar) {
  auto test_x = this->test_x;
  auto test_y = this->test_y;
  auto out_x = this->out_x;
  auto out_y = this->out_y;
  auto len = this->len;

  int out_len =
      lttb::downsample(test_x, test_y, 10000, out_x, out_y, len, 1024);
  ASSERT_NE(out_x[0], out_x[out_len - 1]);
}

TYPED_TEST(FastLTTB, EndPoints_SIMD) {
  auto test_x = this->test_x;
  auto test_y = this->test_y;
  auto out_x = this->out_x;
  auto out_y = this->out_y;
  auto len = this->len;

  int out_len =
      lttb::downsample_simd(test_x, test_y, 10000, out_x, out_y, len, 1024);
  ASSERT_NE(out_x[0], out_x[out_len - 1]);
}

TYPED_TEST(FastLTTB, Timing100M_Scalar) {
  ZoneScoped;
  auto test_x = this->test_x;
  auto test_y = this->test_y;
  auto out_x = this->out_x;
  auto out_y = this->out_y;
  auto len = this->len;

  int out_len =
      lttb::downsample(test_x, test_y, 100000000, out_x, out_y, len, 1024);
}

TYPED_TEST(FastLTTB, Timing100M_NoX_Scalar) {
  ZoneScoped;
  auto test_x = this->test_x;
  auto test_y = this->test_y;
  auto out_x = this->out_x;
  auto out_y = this->out_y;
  auto len = this->len;

  int out_len =
      lttb::downsample(nullptr, test_y, 100000000, out_x, out_y, len, 1024);
}

TYPED_TEST(FastLTTB, Timing100M_SIMD) {
  ZoneScoped;
  auto test_x = this->test_x;
  auto test_y = this->test_y;
  auto out_x = this->out_x;
  auto out_y = this->out_y;
  auto len = this->len;

  int out_len =
      lttb::downsample_simd(test_x, test_y, 100000000, out_x, out_y, len, 1024);
}

TYPED_TEST(FastLTTB, Speedup10M_10Reps) {
  ZoneScoped;
  auto test_x = this->test_x;
  auto test_y = this->test_y;
  auto out_x = this->out_x;
  auto out_y = this->out_y;
  auto len = this->len;

  const int reps = 10;
  constexpr uint64_t num_samples = 10000000;

  for (int use_x = 0; use_x < 2; ++use_x) {
    ZoneScopedN("UseX");
    ZoneValue(use_x);
    for (int bucket_size = 16; bucket_size < 1 << 16; bucket_size <<= 1) {
      double timing_simd = 0.0;
      double timing_scalar = 0.0;

      double best_timing_simd = 10000.0;
      double best_timing_scalar = 10000.0;

      ZoneScopedN("BucketSize");
      ZoneValue(bucket_size);
      {
        auto start = std::chrono::high_resolution_clock::now();
        ZoneScopedN("SIMD");
        for (int i = 0; i < reps; ++i) {
          ZoneScoped;
          auto single_start = std::chrono::high_resolution_clock::now();
          int out_len;
          if (use_x != 0) {
            out_len = lttb::downsample_simd(test_x, test_y, num_samples, out_x,
                                            out_y, len, bucket_size);
          } else {
            out_len = lttb::downsample_simd(nullptr, test_y, num_samples, out_x,
                                            out_y, len, bucket_size);
          }
          auto single_stop = std::chrono::high_resolution_clock::now();
          double timing = std::chrono::duration_cast<
                              std::chrono::duration<double, std::milli>>(
                              single_stop - single_start)
                              .count();
          best_timing_simd = std::min(best_timing_simd, timing);

          ASSERT_NE(out_len, 0);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        timing_simd =
            std::chrono::duration_cast<
                std::chrono::duration<double, std::milli>>(stop - start)
                .count() /
            reps;
      }
      {
        auto start = std::chrono::high_resolution_clock::now();
        ZoneScopedN("Scalar");
        for (int i = 0; i < reps; ++i) {
          ZoneScoped;
          auto single_start = std::chrono::high_resolution_clock::now();
          int out_len;
          if (use_x != 0) {
            out_len = lttb::downsample(test_x, test_y, num_samples, out_x,
                                       out_y, len, bucket_size);
          } else {
            out_len = lttb::downsample(nullptr, test_y, num_samples, out_x,
                                       out_y, len, bucket_size);
          }
          ASSERT_NE(out_len, 0);
          auto single_stop = std::chrono::high_resolution_clock::now();
          double timing = std::chrono::duration_cast<
                              std::chrono::duration<double, std::milli>>(
                              single_stop - single_start)
                              .count();
          best_timing_scalar = std::min(best_timing_scalar, timing);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        timing_scalar =
            std::chrono::duration_cast<
                std::chrono::duration<double, std::milli>>(stop - start)
                .count() /
            reps;
      }

      // clang-format off
      std::printf(
          "Speedup [bucket size %5d | %10s]: "
            "<mean> x%.2f  (simd=%8.2f ms; scalar=%8.2f ms) "
            "<best> x%.2f  (simd=%8.2f ms; scalar=%8.2f ms) "
            "\n",
          bucket_size, use_x ? "with x" : "without x",
          timing_scalar / timing_simd, timing_simd, timing_scalar,
          best_timing_scalar / best_timing_simd, best_timing_simd, best_timing_scalar
          );
      // clang-format on
    }
  }
}

TYPED_TEST(FastLTTB, Timing100M_NoX_SIMD) {
  ZoneScoped;
  auto test_x = this->test_x;
  auto test_y = this->test_y;
  auto out_x = this->out_x;
  auto out_y = this->out_y;
  auto len = this->len;

  int out_len = lttb::downsample_simd(nullptr, test_y, 100000000, out_x, out_y,
                                      len, 1024);
}

TYPED_TEST(FastLTTB, SIMDCorrect) {
  ZoneScoped;
  auto test_x = this->test_x;
  auto test_y = this->test_y;
  auto out_x = this->out_x;
  auto out_y = this->out_y;
  auto len = this->len;

  uint64_t size = 100000;
  int bs = 1024;
  auto *out_x_1 = out_x;
  auto *out_y_1 = out_y;
  auto *out_x_2 = out_x + size;
  auto *out_y_2 = out_y + size;

  // Generate reference
  int out_len_ref =
      lttb::downsample(test_x, test_y, size, out_x_1, out_y_1, len, 8);
  this->write_output("input.csv", out_len_ref);

  int out_len_1 =
      lttb::downsample(test_x, test_y, size, out_x_1, out_y_1, len, bs);
  int out_len_2 =
      lttb::downsample_simd(test_x, test_y, size, out_x_2, out_y_2, len, bs);

  this->write_output("output_scalar.csv", out_len_1, 0);
  this->write_output("output_simd.csv", out_len_2, size);

  ASSERT_EQ(out_len_1, out_len_2);

  for (uint64_t i = 0; i < out_len_1; ++i) {
    EXPECT_EQ(out_x_1[i], out_x_2[i]) << "i=" << i << ", out_len=" << out_len_1;
    ASSERT_EQ(out_y_1[i], out_y_2[i]) << "i=" << i << ", out_len=" << out_len_1;
  }
}
