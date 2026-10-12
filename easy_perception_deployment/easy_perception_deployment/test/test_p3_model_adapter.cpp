#include "ort_cpp_lib/p3_model_adapter.hpp"
#include <gtest/gtest.h>
using namespace Ort;
TEST(P3Adapter, RgbBytesAndExactRoi)
{
  cv::Mat bgr(2, 3, CV_8UC3, cv::Scalar(10, 20, 30));
  auto tensor = rgb01Tensor(bgr, 3, 2);
  ASSERT_EQ(tensor.size(), 18u);
  EXPECT_FLOAT_EQ(tensor[0], 30.0f / 255.0f);
  EXPECT_FLOAT_EQ(tensor[6], 20.0f / 255.0f);
  EXPECT_FLOAT_EQ(tensor[12], 10.0f / 255.0f);
  float boxes[]{ 1.2f, 0, 3, 2 }, scores[]{ 0.8f };
  int64_t labels[]{ 1 };
  float masks[]{ 0, .5f, .50001f, .1f, .9f, 1 };
  std::vector<OrtBase::DataOutputType> out{ { boxes, { 1, 4 } },
                                            { reinterpret_cast<float*>(labels), { 1 } },
                                            { scores, { 1 } },
                                            { masks, { 1, 1, 2, 3 } } };
  auto decoded = decodeP3(out, ModelProfile::rgb01(), 3, 2, 3, 2, 2, .8f);
  ASSERT_EQ(decoded.masks.size(), 1u);
  EXPECT_EQ(decoded.boxes[0], (std::array<float, 4>{ 1, 0, 3, 2 }));
  EXPECT_FLOAT_EQ(decoded.masks[0].at<float>(0, 0), .5f);
  EXPECT_FLOAT_EQ(decoded.masks[0].at<float>(0, 1), .50001f);
  EXPECT_EQ(cv::countNonZero(decoded.masks[0] > .5f), 3);
}
struct Fixture
{
  float boxes[4]{ -.2f, 0, 3, 2 }, scores[1]{ .8f }, masks[6]{ 0, .5f, .5001f, .1f, .9f, 1 };
  int64_t labels[1]{ 1 };
  std::vector<OrtBase::DataOutputType> out()
  {
    return { { boxes, { 1, 4 } },
             { reinterpret_cast<float*>(labels), { 1 } },
             { scores, { 1 } },
             { masks, { 1, 1, 2, 3 } } };
  }
  P3Decoded run(ModelProfile p = ModelProfile::rgb01())
  {
    return decodeP3(out(), p, 3, 2, 3, 2, 2, .8f);
  }
};
TEST(P3Adapter, DefaultsAndInvalidProfiles)
{
  Json::Value root;
  EXPECT_FALSE(ModelProfile::parse(root).fullImage());
  root["model_profile"]["preprocessing"] = "rgb01";
  EXPECT_THROW(ModelProfile::parse(root), std::invalid_argument);
  root["model_profile"]["mask_layout"] = "full_image";
  root["model_profile"]["input_layout"] = "nchw";
  EXPECT_TRUE(ModelProfile::parse(root).fullImage());
  root["model_profile"]["extra"] = true;
  EXPECT_THROW(ModelProfile::parse(root), std::invalid_argument);
}
TEST(P3Adapter, LegacyThresholdAndRoiSemantics)
{
  Fixture f;
  EXPECT_TRUE(f.run({}).boxes.empty()); // legacy >, v2 >=
  EXPECT_EQ(f.run().boxes.size(), 1u);
  f.scores[0] = .81f;
  auto old = f.run({});
  ASSERT_EQ(old.masks.size(), 1u);
  EXPECT_EQ(old.masks[0].size(), cv::Size(3, 2));
  EXPECT_EQ(memcmp(old.masks[0].ptr<float>(), f.masks, sizeof(f.masks)), 0);
}
TEST(P3Adapter, EmptyAndInvalidRuntimeShapes)
{
  std::vector<OrtBase::DataOutputType> empty{
    { nullptr, { 0, 4 } }, { nullptr, { 0 } }, { nullptr, { 0 } }, { nullptr, { 0, 1, 2, 3 } }
  };
  EXPECT_TRUE(decodeP3(empty, ModelProfile::rgb01(), 3, 2, 3, 2, 2, .8f).boxes.empty());
  Fixture f;
  for (size_t i = 0; i < 4; ++i) {
    auto out = f.out();
    out[i].second = { 1 };
    if (i == 1 || i == 2)
      out[i].second = { 2 };
    EXPECT_THROW(decodeP3(out, ModelProfile::rgb01(), 3, 2, 3, 2, 2, .8f), std::invalid_argument);
  }
  EXPECT_THROW(checkedTensorSize({ INT64_MAX, INT64_MAX }), std::invalid_argument);
  EXPECT_THROW(rgb01Tensor(cv::Mat(2, 3, CV_8UC3), 4, 2), std::invalid_argument);
  EXPECT_THROW(rgb01Tensor(cv::Mat(2, 3, CV_32FC3), 3, 2), std::invalid_argument);
}
TEST(P3Adapter, MalformedDataFailsClosed)
{
  Fixture f;
  f.boxes[0] = NAN;
  EXPECT_THROW(f.run(), std::invalid_argument);
  f.boxes[0] = 0;
  f.boxes[2] = -1;
  EXPECT_THROW(f.run(), std::invalid_argument);
  f.boxes[2] = 3;
  f.boxes[2] = INFINITY;
  EXPECT_THROW(f.run(), std::invalid_argument);
  f.boxes[2] = 3;
  f.scores[0] = NAN;
  EXPECT_THROW(f.run(), std::invalid_argument);
  f.scores[0] = .8f;
  f.labels[0] = INT64_MAX;
  EXPECT_THROW(f.run(), std::invalid_argument);
  f.labels[0] = 0;
  EXPECT_THROW(f.run(), std::invalid_argument);
  f.labels[0] = 1;
  f.masks[0] = NAN;
  EXPECT_THROW(f.run(), std::invalid_argument);
  f.masks[0] = 1.01f;
  EXPECT_THROW(f.run(), std::invalid_argument);
}
TEST(P3Adapter, HugeCoordinatesClipBeforeConversion)
{
  Fixture f;
  f.boxes[0] = -std::numeric_limits<float>::max();
  f.boxes[2] = std::numeric_limits<float>::max();
  EXPECT_EQ(f.run().boxes[0], (std::array<float, 4>{ 0, 0, 3, 2 }));
}
TEST(P3Adapter, MetadataIsNotJustNames)
{
  std::vector<TensorContract> in{ { "images", 1, { 1, 3, 2, 3 } } };
  std::vector<TensorContract> out{ { "boxes", 1, { -1, 4 } },
                                   { "labels", 7, { -1 } },
                                   { "scores", 1, { -1 } },
                                   { "masks", 1, { -1, 1, 2, 3 } } };
  EXPECT_NO_THROW(validateRgb01Contract(in, out, 3, 2));
  auto bad = out;
  std::swap(bad[1], bad[2]);
  EXPECT_THROW(validateRgb01Contract(in, bad, 3, 2), std::invalid_argument);
  bad = out;
  bad[1].type = 1;
  EXPECT_THROW(validateRgb01Contract(in, bad, 3, 2), std::invalid_argument);
  bad = out;
  bad[3].shape[1] = 2;
  EXPECT_THROW(validateRgb01Contract(in, bad, 3, 2), std::invalid_argument);
  in[0].shape[0] = -1;
  EXPECT_THROW(validateRgb01Contract(in, out, 3, 2), std::invalid_argument);
}
TEST(P3Adapter, LegacyBgrMeanTensorBytesAndPadding)
{
  cv::Mat image(1, 2, CV_8UC3);
  image.at<cv::Vec3b>(0, 0) = { 1, 2, 3 };
  image.at<cv::Vec3b>(0, 1) = { 4, 5, 6 };
  const cv::Scalar mean(102.9801, 115.9465, 122.7717);
  float tensor[12]{};
  legacyP3Tensor(tensor, image, 2, 1, 2, 2, InputTensorLayout::CHW, false, mean);
  cv::Mat old;
  image.convertTo(old, CV_32FC3);
  old -= mean;
  float expected[12]{};
  for (int c = 0; c < 3; ++c)
    for (int x = 0; x < 2; ++x)
      expected[c * 4 + x] = old.at<cv::Vec3f>(0, x)[c];
  EXPECT_EQ(memcmp(tensor, expected, sizeof(tensor)), 0);
  legacyP3Tensor(tensor, image, 2, 1, 2, 2, InputTensorLayout::NHWC, true, mean);
  const float uint8_expected[]{ 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0 };
  EXPECT_EQ(memcmp(tensor, uint8_expected, sizeof(tensor)), 0);
}
TEST(P3Adapter, RgbExactBytesFromNoncontiguousBgr)
{
  cv::Mat parent(4, 5, CV_8UC3);
  cv::Mat image = parent(cv::Rect(1, 1, 3, 2));
  ASSERT_FALSE(image.isContinuous());
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 3; ++x)
      image.at<cv::Vec3b>(y, x) = { uint8_t(x + y), uint8_t(20 + x + y), uint8_t(200 + x + y) };
  auto actual = rgb01Tensor(image, 3, 2);
  float expected[18];
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 3; ++x)
        expected[c * 6 + y * 3 + x] = float(image.at<cv::Vec3b>(y, x)[2 - c]) / 255.0f;
  EXPECT_EQ(memcmp(actual.data(), expected, sizeof(expected)), 0);
}
TEST(P3Adapter, LegacyFloatScalingBoundaryUnchanged)
{
  Fixture f;
  f.scores[0] = .81f;
  f.boxes[0] = 0.4166666567325592f;
  f.boxes[2] = 2.f;
  auto decoded = decodeP3(f.out(), {}, 1920, 2, 800, 2, 2, .8f);
  ASSERT_EQ(decoded.boxes.size(), 1u);
  EXPECT_EQ(decoded.boxes[0][0], 1.f);
}
