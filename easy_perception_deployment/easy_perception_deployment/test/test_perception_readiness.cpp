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

// Detection-to-grasp readiness: identity survives pose rejection, while only
// fully validated, published poses become grasp candidates.
#include <gtest/gtest.h>
#include <map>
#include "epd_utils_lib/hybrid_rgbd.hpp"
#include "epd_utils_lib/perception_readiness.hpp"
#include "epd_utils_lib/temporal_tracker.hpp"

namespace
{
struct Fixture
{
  cv::Mat depth = cv::Mat(6, 12, CV_32FC1);
  sensor_msgs::msg::Image::SharedPtr rgb = std::make_shared<sensor_msgs::msg::Image>();
  sensor_msgs::msg::Image::SharedPtr dep = std::make_shared<sensor_msgs::msg::Image>();
  sensor_msgs::msg::CameraInfo::SharedPtr info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  EPD::EPDObjectDetection detections{0};
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
  void add(int x0, int x1, uint64_t label = 0, float score = 0.9F)
  {
    detections.bboxes.push_back({x0, 0, x1, 6});
    detections.masks.emplace_back(6, x1 - x0, CV_32FC1, cv::Scalar(1));
    detections.classIndices.push_back(label);
    detections.scores.push_back(score);
  }
  EPD::EPDObjectLocalization run()
  {
    EPD::SynchronizationMetadata sync;
    sync.synchronized = sync.exact_sensor_stamp = true;
    EPD::Observation observation(42, "camera", rgb, dep, info, sync);
    return EPD::refineRgbd(detections, depth, observation, {"banana", "apple"});
  }
};

std::map<std::string, std::string> values(const diagnostic_msgs::msg::DiagnosticStatus & status)
{
  std::map<std::string, std::string> out;
  for (const auto & item : status.values) {
    out[item.key] = item.value;
  }
  return out;
}
}  // namespace

// Scenario A: recognised object with valid geometry.
TEST(PerceptionReadiness, ValidGeometryIsGraspCandidateOnlyWhenPublished)
{
  Fixture f;
  f.add(0, 12, 0, 0.97F);
  const auto o = f.run().objects.at(0);
  ASSERT_EQ(o.quality, EPD::GeometryQuality::VALID);
  EXPECT_EQ(EPD::classifyPerceptionState(o, true), EPD::PerceptionState::GRASP_CANDIDATE);
  EXPECT_EQ(EPD::classifyPerceptionState(o, false), EPD::PerceptionState::POSE_VALID);
  const auto record = EPD::makeReadinessRecord(0, o, true);
  EXPECT_EQ(record.name, "banana");
  EXPECT_FLOAT_EQ(record.confidence, 0.97F);
  EXPECT_NEAR(record.centroid.z, o.centroid.z, 1e-12);
  EXPECT_NEAR(record.axis.x, o.axis.x, 1e-12);
}

// Scenario B: round apple, orientation ambiguous. Identity and measured position
// survive; the pose gate is not relaxed and no axis is reported.
TEST(PerceptionReadiness, AmbiguousAxisAppleIsLocalisedNotGraspCandidate)
{
  Fixture f;
  f.add(0, 6, 1, 0.95F);
  const auto o = f.run().objects.at(0);
  ASSERT_NE(o.quality, EPD::GeometryQuality::VALID);
  EXPECT_NE(o.failure_reasons & EPD::reason(EPD::GeometryFailure::AMBIGUOUS_ORIENTATION), 0U);
  // Even if a caller wrongly claims publication, an invalid pose is never a candidate.
  EXPECT_EQ(EPD::classifyPerceptionState(o, true), EPD::PerceptionState::LOCALISED);
  const auto record = EPD::makeReadinessRecord(3, o, false);
  EXPECT_EQ(record.name, "apple");
  EXPECT_FLOAT_EQ(record.confidence, 0.95F);
  EXPECT_NEAR(record.centroid.z, o.centroid.z, 1e-12);
  EXPECT_GT(record.centroid.z, 0.49);
  EXPECT_EQ(record.axis.x, 0);
  EXPECT_EQ(record.axis.y, 0);
  EXPECT_EQ(record.axis.z, 0);

  std_msgs::msg::Header header;
  header.frame_id = "optical";
  header.stamp.sec = 7;
  const auto report = EPD::makeReadinessReport({record}, header, 42, "cam");
  ASSERT_EQ(report.status.size(), 1U);
  EXPECT_EQ(report.header, header);
  EXPECT_EQ(report.status[0].name, "easy_perception_deployment/detection/3");
  EXPECT_EQ(report.status[0].level, diagnostic_msgs::msg::DiagnosticStatus::WARN);
  auto v = values(report.status[0]);
  EXPECT_EQ(v["state"], "LOCALISED");
  EXPECT_EQ(v["class"], "apple");
  EXPECT_EQ(v["position_valid"], "true");
  EXPECT_EQ(v["orientation_valid"], "false");
  EXPECT_EQ(v["axis"], "");
  EXPECT_NE(v["reasons"].find("ambiguous_axis_no_pose"), std::string::npos);
  EXPECT_EQ(v["observation_id"], "42");
}

// Scenario C: touching objects. Disputed pixels have one owner or none; both
// objects keep identity and remain localisable.
TEST(PerceptionReadiness, TouchingObjectsKeepIdentityAndExclusiveMasks)
{
  Fixture f;
  f.add(0, 7, 0);
  f.add(6, 12, 1);
  const auto r = f.run();
  ASSERT_EQ(r.size(), 2U);
  EXPECT_EQ(r.objects[0].name, "banana");
  EXPECT_EQ(r.objects[1].name, "apple");
  cv::Mat owners = cv::Mat::zeros(6, 12, CV_32SC1);
  for (const auto & o : r.objects) {
    cv::Mat votes;
    o.mask.convertTo(votes, CV_32SC1, 1.0 / 255.0);
    cv::Rect roi(o.roi.x_offset, o.roi.y_offset, o.roi.width, o.roi.height);
    owners(roi) += votes;
    EXPECT_NE(EPD::classifyPerceptionState(o, false), EPD::PerceptionState::REJECTED);
  }
  double max_owners = 0;
  cv::minMaxLoc(owners, nullptr, &max_owners);
  EXPECT_LE(max_owners, 1.0);
}

// Scenario D: duplicate proposals with contested ownership. Neither receives a
// pose, both remain reported with class, confidence and a specific reason.
TEST(PerceptionReadiness, DuplicateProposalsAreRejectedWithIdentityRetained)
{
  Fixture f;
  f.add(0, 12, 1, 0.9F);
  f.add(0, 12, 1, 0.6F);
  const auto r = f.run();
  for (size_t i = 0; i < r.size(); ++i) {
    const auto record = EPD::makeReadinessRecord(i, r.objects[i], false);
    EXPECT_EQ(record.state, EPD::PerceptionState::REJECTED);
    EXPECT_EQ(record.name, "apple");
    EXPECT_GT(record.confidence, 0.5F);
    EXPECT_NE(
      EPD::failureNames(record.failure_reasons).find("overlapping_ownership"),
      std::string::npos);
    EXPECT_EQ(record.centroid.z, 0);
  }
  std_msgs::msg::Header header;
  auto v = values(
    EPD::makeReadinessReport(
      {EPD::makeReadinessRecord(0, r.objects[0], false)}, header, 1, "cam").status.at(0));
  EXPECT_EQ(v["position_valid"], "false");
  EXPECT_EQ(v["centroid_m"], "");
}

TEST(PerceptionReadiness, MissingDepthKeepsDetectionWithoutPosition)
{
  Fixture f;
  f.add(0, 12, 1);
  f.depth.setTo(0);
  const auto o = f.run().objects.at(0);
  EXPECT_EQ(EPD::classifyPerceptionState(o, true), EPD::PerceptionState::DETECTED);
  const auto record = EPD::makeReadinessRecord(0, o, true);
  EXPECT_EQ(record.name, "apple");
  EXPECT_EQ(record.centroid.z, 0);
}

TEST(PerceptionReadiness, UnsynchronisedInputRejectsGeometryButKeepsIdentity)
{
  Fixture f;
  f.add(0, 12, 1, 0.8F);
  f.dep->header.stamp.sec = 8;
  const auto o = f.run().objects.at(0);
  EXPECT_NE(o.quality, EPD::GeometryQuality::VALID);
  EXPECT_EQ(o.name, "apple");
  EXPECT_FLOAT_EQ(o.confidence, 0.8F);
  EXPECT_EQ(EPD::classifyPerceptionState(o, true), EPD::PerceptionState::REJECTED);
  EXPECT_NE(EPD::failureNames(o.failure_reasons).find("unsynchronized_input"), std::string::npos);
}

// Scenario E: a 2D-only (pose-invalid) object keeps its track across a missed
// observation, so identity continuity does not depend on orientation validity.
TEST(PerceptionReadiness, PoseInvalidObjectKeepsTrackAcrossTemporaryLoss)
{
  EPD::TemporalTracker tracker;
  EPD::TrackingDetection apple;
  apple.name = "apple";
  apple.roi.x_offset = 10;
  apple.roi.y_offset = 10;
  apple.roi.width = apple.roi.height = 40;
  apple.centroid.z = 0.5;
  apple.geometry_valid = false;
  apple.detector_confidence = 0.95F;
  builtin_interfaces::msg::Time t;
  t.sec = 1;
  const auto first = tracker.update(1, t, {apple});
  t.sec = 2;
  const auto second = tracker.update(2, t, {apple});
  t.sec = 3;
  tracker.update(3, t, {});
  t.sec = 4;
  apple.roi.x_offset = 14;
  const auto reacquired = tracker.update(4, t, {apple});
  ASSERT_EQ(first.size(), 1U);
  ASSERT_EQ(reacquired.size(), 1U);
  EXPECT_NE(first[0].track_id, 0U);
  EXPECT_EQ(second[0].track_id, first[0].track_id);
  EXPECT_EQ(reacquired[0].track_id, first[0].track_id);
  EXPECT_EQ(tracker.metrics().id_switches, 0U);
}
