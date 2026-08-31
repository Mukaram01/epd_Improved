#include <limits>

#include "gtest/gtest.h"
#include "epd_utils_lib/geometry_quality.hpp"

namespace
{
EPD::LocalizedObject validObject()
{
  EPD::LocalizedObject object;
  object.roi.x_offset = 1;
  object.roi.y_offset = 1;
  object.roi.width = 4;
  object.roi.height = 4;
  object.mask = cv::Mat::ones(4, 4, CV_32FC1);
  object.mask_pixel_count = 16;
  object.valid_depth_pixel_count = 16;
  object.valid_depth_ratio = 1.0;
  object.centroid.x = 0.1;
  object.centroid.y = 0.2;
  object.centroid.z = 0.8;
  object.length = 0.2F;
  object.breadth = 0.1F;
  object.height = 0.05F;
  object.axis.x = 1.0;
  for (size_t i = 0; i < 16; ++i) {
    object.segmented_pcl.emplace_back(0.01F * i, 0.0F, 0.8F);
  }
  return object;
}

TEST(GeometryQuality, ValidSyntheticMaskDepthProducesFiniteCentroid)
{
  EPD::LocalizedObject object;
  cv::Mat mask = cv::Mat::ones(8, 8, CV_8UC1);
  cv::Mat depth(8, 8, CV_32FC1, cv::Scalar(0.8F));
  ASSERT_TRUE(EPD::populateMaskedDepthCentroid(object, mask, depth, 600, 600, 4, 4));
  EXPECT_TRUE(EPD::finitePoint(object.centroid));
  EXPECT_EQ(object.segmented_pcl.size(), 64U);
}

TEST(GeometryQuality, ValidGeometryPasses)
{
  auto object = validObject();
  EXPECT_EQ(
    EPD::validateLocalizedObject(object, 10, 10, 600, 600, 5, 5),
    EPD::GeometryQuality::VALID);
  EXPECT_GT(object.length, 0.0F);
  EXPECT_GT(object.breadth, 0.0F);
  EXPECT_GT(object.height, 0.0F);
}

TEST(GeometryQuality, NonfiniteInputRejected)
{
  auto object = validObject();
  object.centroid.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
    EPD::validateLocalizedObject(object, 10, 10, 600, 600, 5, 5),
    EPD::GeometryQuality::INVALID);
}

TEST(GeometryQuality, InvalidFocalLengthsRejected)
{
  auto object = validObject();
  EXPECT_EQ(
    EPD::validateLocalizedObject(object, 10, 10, 0, 600, 5, 5),
    EPD::GeometryQuality::INVALID);
  EXPECT_FALSE(EPD::validIntrinsics(600, std::numeric_limits<double>::infinity(), 5, 5));
}

TEST(GeometryQuality, EmptyMaskRejected)
{
  auto object = validObject();
  object.mask.release();
  EXPECT_EQ(
    EPD::validateLocalizedObject(object, 10, 10, 600, 600, 5, 5),
    EPD::GeometryQuality::INVALID);
}

TEST(GeometryQuality, EmptyCloudRejected)
{
  auto object = validObject();
  object.segmented_pcl.clear();
  EXPECT_EQ(
    EPD::validateLocalizedObject(object, 10, 10, 600, 600, 5, 5),
    EPD::GeometryQuality::INVALID);
}

TEST(GeometryQuality, InsufficientDepthRejected)
{
  auto object = validObject();
  object.valid_depth_pixel_count = 1;
  object.valid_depth_ratio = 0.01;
  EXPECT_EQ(
    EPD::validateLocalizedObject(object, 10, 10, 600, 600, 5, 5),
    EPD::GeometryQuality::DEGRADED);
}

TEST(GeometryQuality, OutsideRoiRejectedWithoutClamping)
{
  auto object = validObject();
  object.roi.x_offset = 9;
  EXPECT_FALSE(EPD::roiInsideImage(object.roi, 10, 10));
}

TEST(GeometryQuality, InvalidDepthExcluded)
{
  EPD::LocalizedObject object;
  cv::Mat mask = cv::Mat::ones(2, 2, CV_8UC1);
  cv::Mat depth(2, 2, CV_32FC1);
  depth.at<float>(0, 0) = 0.0F;
  depth.at<float>(0, 1) = std::numeric_limits<float>::quiet_NaN();
  depth.at<float>(1, 0) = std::numeric_limits<float>::infinity();
  depth.at<float>(1, 1) = 1.0F;
  ASSERT_TRUE(EPD::populateMaskedDepthCentroid(object, mask, depth, 600, 600, 1, 1));
  EXPECT_EQ(object.valid_depth_pixel_count, 1U);
}

TEST(GeometryQuality, NormalizedAxisRequired)
{
  auto object = validObject();
  object.axis.x = 2.0;
  EXPECT_EQ(
    EPD::validateLocalizedObject(object, 10, 10, 600, 600, 5, 5),
    EPD::GeometryQuality::INVALID);
}

TEST(GeometryQuality, DegenerateOrientationIsExplicit)
{
  auto object = validObject();
  object.axis.x = 0.0;
  EXPECT_EQ(
    EPD::validateLocalizedObject(object, 10, 10, 600, 600, 5, 5),
    EPD::GeometryQuality::INVALID);
  EXPECT_NE(object.failure_reasons & EPD::reason(EPD::GeometryFailure::INVALID_ORIENTATION), 0U);
}

TEST(GeometryQuality, SourceIdentitySurvivesValidation)
{
  auto object = validObject();
  object.source_observation_id = 42;
  object.source_sensor_stamp.sec = 123;
  object.source_frame = "camera_color_optical_frame";
  EPD::validateLocalizedObject(object, 10, 10, 600, 600, 5, 5);
  EXPECT_EQ(object.source_observation_id, 42U);
  EXPECT_EQ(object.source_sensor_stamp.sec, 123);
  EXPECT_EQ(object.source_frame, "camera_color_optical_frame");
}

TEST(GeometryQuality, DifferentObservationIdentityIsDetectable)
{
  auto mask = validObject();
  auto depth = validObject();
  mask.source_observation_id = 7;
  depth.source_observation_id = 8;
  EXPECT_NE(mask.source_observation_id, depth.source_observation_id);
}

TEST(GeometryQuality, OneInvalidObjectDoesNotAffectAnother)
{
  auto invalid = validObject();
  auto valid = validObject();
  invalid.height = 0.0F;
  EXPECT_EQ(
    EPD::validateLocalizedObject(invalid, 10, 10, 600, 600, 5, 5),
    EPD::GeometryQuality::INVALID);
  EXPECT_EQ(
    EPD::validateLocalizedObject(valid, 10, 10, 600, 600, 5, 5),
    EPD::GeometryQuality::VALID);
}

TEST(GeometryQuality, ZeroDetectionsIsHealthy)
{
  EPD::EPDObjectLocalization result(0);
  EXPECT_TRUE(result.objects.empty());
}
}  // namespace
