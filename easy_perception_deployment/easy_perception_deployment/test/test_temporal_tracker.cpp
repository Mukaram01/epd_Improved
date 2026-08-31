#include <cmath>
#include <limits>

#include "gtest/gtest.h"
#include "epd_utils_lib/temporal_tracker.hpp"

namespace
{
builtin_interfaces::msg::Time stamp(double seconds)
{
  builtin_interfaces::msg::Time value;
  value.sec = static_cast<int32_t>(seconds);
  value.nanosec = static_cast<uint32_t>((seconds - value.sec) * 1e9);
  return value;
}

EPD::TrackingDetection detection(
  std::string name = "mouse", uint32_t x = 10, double px = 0.0, bool valid_3d = true)
{
  EPD::TrackingDetection value;
  value.name = std::move(name);
  value.roi.x_offset = x;
  value.roi.y_offset = 10;
  value.roi.width = 40;
  value.roi.height = 40;
  value.centroid.x = px;
  value.centroid.z = 0.5;
  value.geometry_valid = valid_3d;
  value.detector_confidence = 0.9F;
  return value;
}

TEST(TemporalTracker, SameObjectKeepsId)
{
  EPD::TemporalTracker tracker;
  const auto first = tracker.update(1, stamp(1), {detection()});
  const auto second = tracker.update(2, stamp(2), {detection("mouse", 12, 0.01)});
  ASSERT_EQ(first.size(), 1U); ASSERT_EQ(second.size(), 1U);
  EXPECT_EQ(first[0].track_id, second[0].track_id);
  EXPECT_EQ(second[0].lifecycle, EPD::TrackLifecycle::CONFIRMED);
}

TEST(TemporalTracker, SameClassObjectsStayDistinct)
{
  EPD::TemporalTracker tracker;
  const auto values =
    tracker.update(1, stamp(1), {detection("mouse", 10), detection("mouse", 200)});
  ASSERT_EQ(values.size(), 2U);
  EXPECT_NE(values[0].track_id, values[1].track_id);
}

TEST(TemporalTracker, ShortMissRetainsId)
{
  EPD::TemporalTracker tracker;
  const auto id = tracker.update(1, stamp(1), {detection()})[0].track_id;
  tracker.update(2, stamp(2), {});
  EXPECT_EQ(tracker.update(3, stamp(3), {detection()})[0].track_id, id);
}

TEST(TemporalTracker, LongMissExpiresAndReappearanceGetsNewId)
{
  EPD::TemporalTracker tracker;
  const auto id = tracker.update(1, stamp(1), {detection()})[0].track_id;
  for (uint64_t observation = 2; observation <= 5; ++observation) {
    tracker.update(observation, stamp(observation), {});
  }
  EXPECT_EQ(tracker.activeTrackCount(), 0U);
  const auto new_id = tracker.update(6, stamp(6), {detection()})[0].track_id;
  EXPECT_NE(new_id, id);
  EXPECT_EQ(tracker.metrics().tracks_expired, 1U);
}

TEST(TemporalTracker, RegressedObservationRejected)
{
  EPD::TemporalTracker tracker;
  tracker.update(2, stamp(2), {detection()});
  EXPECT_TRUE(tracker.update(1, stamp(1), {detection()}).empty());
  EXPECT_EQ(tracker.metrics().out_of_order_observations, 1U);
}

TEST(TemporalTracker, DuplicateObservationRejected)
{
  EPD::TemporalTracker tracker;
  tracker.update(1, stamp(1), {detection()});
  EXPECT_TRUE(tracker.update(1, stamp(1), {detection()}).empty());
  EXPECT_EQ(tracker.metrics().duplicate_update_attempts, 1U);
}

TEST(TemporalTracker, ClassMismatchCreatesNewTrack)
{
  EPD::TemporalTracker tracker;
  const auto first = tracker.update(1, stamp(1), {detection("mouse")})[0].track_id;
  const auto second = tracker.update(2, stamp(2), {detection("bottle")})[0].track_id;
  EXPECT_NE(first, second);
}

TEST(TemporalTracker, LargeCentroidJumpRejected)
{
  EPD::TemporalTracker tracker;
  const auto first = tracker.update(1, stamp(1), {detection("mouse", 10, 0.0)})[0].track_id;
  const auto second = tracker.update(2, stamp(2), {detection("mouse", 300, 1.0)})[0].track_id;
  EXPECT_NE(first, second);
}

TEST(TemporalTracker, AssociationIsDeterministic)
{
  EPD::TemporalTracker tracker;
  const auto first = tracker.update(1, stamp(1), {detection("mouse", 10), detection("mouse", 100)});
  const auto second =
    tracker.update(2, stamp(2), {detection("mouse", 102), detection("mouse", 12)});
  EXPECT_EQ(second[0].track_id, first[1].track_id);
  EXPECT_EQ(second[1].track_id, first[0].track_id);
}

TEST(TemporalTracker, InvalidGeometryHasNoVelocity)
{
  EPD::TemporalTracker tracker;
  tracker.update(1, stamp(1), {detection("mouse", 10, 0.0, false)});
  EXPECT_FALSE(tracker.update(2, stamp(2), {detection("mouse", 12, 0.1, false)})[0].velocity);
}

TEST(TemporalTracker, ValidMotionHasFiniteVelocity)
{
  EPD::TemporalTracker tracker;
  tracker.update(1, stamp(1), {detection("mouse", 10, 0.0)});
  const auto velocity = tracker.update(2, stamp(2), {detection("mouse", 12, 0.1)})[0].velocity;
  ASSERT_TRUE(velocity);
  EXPECT_NEAR(velocity->x, 0.1, 1e-6);
  EXPECT_TRUE(std::isfinite(velocity->x));
}

TEST(TemporalTracker, InvalidTimeDeltaHasNoVelocity)
{
  EPD::TemporalTracker tracker;
  tracker.update(1, stamp(2), {detection()});
  EXPECT_FALSE(tracker.update(2, stamp(2), {detection("mouse", 12, 0.1)})[0].velocity);
}

TEST(TemporalTracker, ZeroDetectionsHealthy)
{
  EPD::TemporalTracker tracker;
  EXPECT_TRUE(tracker.update(1, stamp(1), {}).empty());
  EXPECT_EQ(tracker.metrics().latest_track_observation_id, 1U);
}

TEST(TemporalTracker, TrackingConfidenceIsSeparateFromDetectorConfidence)
{
  EPD::TemporalTracker tracker;
  auto object = detection();
  object.detector_confidence = 0.91F;
  const auto first = tracker.update(1, stamp(1), {object})[0];
  const auto second = tracker.update(2, stamp(2), {object})[0];
  EXPECT_FLOAT_EQ(first.detector_confidence, 0.91F);
  EXPECT_FLOAT_EQ(second.detector_confidence, 0.91F);
  EXPECT_GT(second.tracking_confidence, first.tracking_confidence);
}

TEST(TemporalTracker, StorageRemainsBounded)
{
  EPD::TrackerThresholds thresholds;
  thresholds.maximum_active_tracks = 4;
  EPD::TemporalTracker tracker(thresholds);
  for (uint64_t i = 1; i <= 1000; ++i) {
    tracker.update(i, stamp(i), {detection("mouse", static_cast<uint32_t>(i * 200), i)});
    EXPECT_LE(tracker.activeTrackCount(), 4U);
  }
}

TEST(TemporalTracker, MalformedObjectDoesNotKillValidTracking)
{
  EPD::TemporalTracker tracker;
  auto malformed = detection();
  malformed.name.clear();
  malformed.roi.width = 0;
  const auto values = tracker.update(1, stamp(1), {malformed, detection("mouse", 100)});
  ASSERT_EQ(values.size(), 2U);
  EXPECT_EQ(values[0].track_id, 0U);
  EXPECT_NE(values[1].track_id, 0U);
}
}  // namespace
