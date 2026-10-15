// Copyright 2026 Advanced Remanufacturing and Technology Centre
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include "epd_utils_lib/hybrid_rgbd.hpp"

namespace
{
struct Fixture
{
  cv::Mat depth = cv::Mat(6, 12, CV_32FC1);
  sensor_msgs::msg::Image::SharedPtr rgb = std::make_shared<sensor_msgs::msg::Image>();
  sensor_msgs::msg::Image::SharedPtr dep = std::make_shared<sensor_msgs::msg::Image>();
  sensor_msgs::msg::CameraInfo::SharedPtr info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  EPD::EPDObjectDetection detections{0};
  EPD::HybridRgbdOptions options;
  Fixture()
  {
    rgb->width = dep->width = info->width = 12;
    rgb->height = dep->height = info->height = 6;
    rgb->header.frame_id = dep->header.frame_id = info->header.frame_id = "optical";
    rgb->header.stamp.sec = dep->header.stamp.sec = info->header.stamp.sec = 7;
    info->k = {100, 0, 5.5, 0, 100, 2.5, 0, 0, 1};
    for (int y = 0; y < 6; ++y) {
      for (int x = 0; x < 12; ++x) {depth.at<float>(y, x) = 0.5F + x * 0.001F;}
    }
  }
  void add(int x0, int x1)
  {
    detections.bboxes.push_back({x0, 0, x1, 6});
    detections.masks.emplace_back(6, x1 - x0, CV_32FC1, cv::Scalar(1));
    detections.classIndices.push_back(0);
    detections.scores.push_back(0.9F);
  }
  EPD::EPDObjectLocalization run()
  {
    EPD::SynchronizationMetadata sync;
    sync.synchronized = sync.exact_sensor_stamp = true;
    EPD::Observation observation(42, "camera", rgb, dep, info, sync);
    return EPD::refineRgbd(detections, depth, observation, {"part"}, options);
  }
};
}  // namespace

TEST(HybridRgbd, ConvertsMetricDepthAndPreservesSource)
{
  Fixture f;
  f.add(0, 12);
  auto a = f.run().objects.at(0);
  ASSERT_EQ(a.quality, EPD::GeometryQuality::VALID);
  EXPECT_NEAR(a.centroid.z, 0.5055, 1e-7);
  EXPECT_NEAR(a.centroid.x, 0.0001191667, 1e-7);
  EXPECT_NEAR(a.centroid.y, 0, 1e-7);
  EXPECT_EQ(a.source_frame, "optical");
  EXPECT_EQ(a.source_sensor_stamp.sec, 7);
  EXPECT_EQ(a.source_observation_id, 42U);
  EXPECT_EQ(a.segmented_pcl.header.stamp, 7000000U);
  cv::Mat millimetres;
  f.depth.convertTo(millimetres, CV_16UC1, 1000);
  f.depth = millimetres;
  auto b = f.run().objects.at(0);
  EXPECT_NEAR(a.centroid.z, b.centroid.z, 1e-7);
  EXPECT_DOUBLE_EQ(b.axis.x, f.run().objects.at(0).axis.x);
  EXPECT_NEAR(a.axis.x, b.axis.x, 1e-6);
  EXPECT_EQ(cv::countNonZero(a.mask), 72);
}

TEST(HybridRgbd, TouchingInstancesRemainSeparate)
{
  Fixture f;
  f.add(0, 6);
  f.add(6, 12);
  auto r = f.run();
  ASSERT_EQ(r.size(), 2U);
  EXPECT_EQ(r.objects[0].segmented_pcl.size(), 36U);
  EXPECT_EQ(r.objects[1].segmented_pcl.size(), 36U);
  EXPECT_LT(r.objects[0].centroid.x, r.objects[1].centroid.x);
}

TEST(HybridRgbd, RemovesDisputedPixelsWithoutAssigningThemToEitherOwner)
{
  Fixture f;
  f.add(0, 7);
  f.add(6, 12);
  auto r = f.run();
  EXPECT_EQ(cv::countNonZero(r.objects[0].mask), 36);
  EXPECT_EQ(cv::countNonZero(r.objects[1].mask), 30);
  EXPECT_EQ(cv::countNonZero(r.objects[0].mask.col(6)), 0);
  EXPECT_EQ(cv::countNonZero(r.objects[1].mask.col(0)), 0);
}

TEST(HybridRgbd, RejectsSevereOwnershipConflictEvenWithValidGeometry)
{
  Fixture f;
  f.add(0, 12);
  f.add(0, 12);
  auto r = f.run();
  for (const auto & o : r.objects) {
    EXPECT_NE(o.quality, EPD::GeometryQuality::VALID);
    EXPECT_NE(o.failure_reasons & EPD::reason(EPD::GeometryFailure::OVERLAPPING_OWNERSHIP), 0U);
  }
}

TEST(HybridRgbd, RejectsMissingNegativeAndNonfiniteDepth)
{
  for (float z : {0.F, -0.5F, std::numeric_limits<float>::quiet_NaN()}) {
    Fixture f;
    f.add(0, 12);
    f.depth.setTo(z);
    const auto o = f.run().objects.at(0);
    EXPECT_NE(o.quality, EPD::GeometryQuality::VALID);
    EXPECT_TRUE(o.segmented_pcl.empty());
  }
}

TEST(HybridRgbd, KeepsDominantConnectedSurfaceAndRejectsEqualSurfaces)
{
  Fixture f;
  f.add(0, 12);
  f.depth.colRange(10, 12).setTo(0.9F);
  auto o = f.run().objects.at(0);
  EXPECT_EQ(cv::countNonZero(o.mask), 60);
  EXPECT_LT(o.centroid.z, 0.51);
  f.depth.colRange(6, 12).setTo(0.9F);
  o = f.run().objects.at(0);
  EXPECT_NE(o.failure_reasons & EPD::reason(EPD::GeometryFailure::AMBIGUOUS_SURFACE), 0U);
  EXPECT_NE(o.quality, EPD::GeometryQuality::VALID);
}

TEST(HybridRgbd, OptionalSizeCheckRejectsWrongVisibleExtent)
{
  Fixture f;
  f.add(0, 12);
  EXPECT_EQ(f.run().objects.at(0).quality, EPD::GeometryQuality::VALID);
  f.options.maximum_visible_extent_m = 0.02;
  const auto o = f.run().objects.at(0);
  EXPECT_NE(o.failure_reasons & EPD::reason(EPD::GeometryFailure::GEOMETRY_MISMATCH), 0U);
  EXPECT_NE(o.quality, EPD::GeometryQuality::VALID);
}

TEST(HybridRgbd, RejectsMismatchedFramesStampsCalibrationAndMask)
{
  Fixture f;
  f.add(0, 12);
  f.dep->header.frame_id = "other";
  EXPECT_NE(f.run().objects.at(0).quality, EPD::GeometryQuality::VALID);
  f.dep->header.frame_id = "optical";
  f.dep->header.stamp.sec = 8;
  EXPECT_NE(f.run().objects.at(0).quality, EPD::GeometryQuality::VALID);
  f.dep->header.stamp.sec = 7;
  f.info->k[0] = 0;
  EXPECT_NE(f.run().objects.at(0).quality, EPD::GeometryQuality::VALID);
  f.info->k[0] = 100;
  f.detections.masks[0] = cv::Mat();
  EXPECT_NE(f.run().objects.at(0).quality, EPD::GeometryQuality::VALID);
}

TEST(HybridRgbd, RecoversSupportedOwnerFromWeakOverlappingProposal)
{
  Fixture f;
  f.add(0, 12);
  f.add(0, 4);
  auto r = f.run();
  EXPECT_EQ(cv::countNonZero(r.objects[0].mask), 72);
  EXPECT_EQ(r.objects[0].quality, EPD::GeometryQuality::VALID);
  EXPECT_EQ(cv::countNonZero(r.objects[1].mask), 0);
  EXPECT_NE(r.objects[1].quality, EPD::GeometryQuality::VALID);
  EXPECT_NE(
    r.objects[1].failure_reasons & EPD::reason(
      EPD::GeometryFailure::OVERLAPPING_OWNERSHIP), 0U);
  // Proposal order must not decide ownership.
  std::swap(f.detections.bboxes[0], f.detections.bboxes[1]);
  std::swap(f.detections.masks[0], f.detections.masks[1]);
  const auto reversed = f.run();
  EXPECT_EQ(cv::countNonZero(reversed.objects[1].mask), 72);
  EXPECT_EQ(cv::countNonZero(reversed.objects[0].mask), 0);
}

TEST(HybridRgbd, NeverReclaimsAcrossDepthDiscontinuity)
{
  Fixture f;
  f.add(0, 12);
  f.add(0, 4);
  f.depth.colRange(0, 4).setTo(0.9F);
  const auto r = f.run();
  EXPECT_EQ(cv::countNonZero(r.objects[0].mask.colRange(0, 4)), 0);
  EXPECT_NE(r.objects[0].quality, EPD::GeometryQuality::VALID);
}

TEST(HybridRgbd, PreservesDepthSupportedDisconnectedVisibleFragments)
{
  Fixture f;
  f.add(0, 12);
  f.detections.masks[0].col(8).setTo(0);
  const auto r = f.run().objects.at(0);
  EXPECT_EQ(cv::countNonZero(r.mask), 66);
  EXPECT_EQ(r.quality, EPD::GeometryQuality::VALID);
  EXPECT_EQ(cv::countNonZero(r.mask.col(8)), 0);  // no invented occluded pixels
  EXPECT_EQ(cv::countNonZero(r.mask.colRange(9, 12)), 18);
}

TEST(HybridRgbd, DoesNotRecoverDisconnectedDifferentDepthSurface)
{
  Fixture f;
  f.add(0, 12);
  f.detections.masks[0].col(8).setTo(0);
  f.depth.colRange(9, 12).setTo(0.9F);
  const auto r = f.run().objects.at(0);
  EXPECT_EQ(cv::countNonZero(r.mask), 48);
  EXPECT_EQ(cv::countNonZero(r.mask.colRange(9, 12)), 0);
}

TEST(HybridRgbd, AmbiguousAxisRemainsUnpublishedWithoutRelaxingOrientationGate)
{
  Fixture f;
  f.add(0, 6);
  const auto object = f.run().objects.at(0);
  EXPECT_EQ(object.quality, EPD::GeometryQuality::DEGRADED);
  EXPECT_NE(object.failure_reasons & EPD::reason(EPD::GeometryFailure::INVALID_ORIENTATION), 0U);
  EXPECT_NE(object.failure_reasons & EPD::reason(EPD::GeometryFailure::AMBIGUOUS_ORIENTATION), 0U);
  EXPECT_EQ(object.axis.x, 0);
  EXPECT_FALSE(object.segmented_pcl.empty());  // measured geometry exists, pose does not
}

TEST(HybridRgbd, DisconnectedRecoveryNeverFillsAnotherOwnersPixels)
{
  Fixture f;
  f.add(0, 12);
  f.detections.masks[0].col(8).setTo(0);
  f.add(8, 9);
  const auto result = f.run();
  EXPECT_EQ(cv::countNonZero(result.objects[0].mask.col(8)), 0);
  EXPECT_EQ(cv::countNonZero(result.objects[1].mask), 6);
}
