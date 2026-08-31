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

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "epd_msgs/msg/epd_object_localization.hpp"
#include "epd_utils_lib/observation.hpp"

using namespace std::chrono_literals;

namespace
{
sensor_msgs::msg::Image::SharedPtr image(int32_t seconds, const std::string & frame = "color")
{
  auto message = std::make_shared<sensor_msgs::msg::Image>();
  message->header.stamp.sec = seconds;
  message->header.stamp.nanosec = 123456789;
  message->header.frame_id = frame;
  return message;
}

EPD::Observation::ConstSharedPtr publish(EPD::LatestObservationStore & store, int32_t seconds)
{
  auto rgb = image(seconds);
  auto depth = image(seconds, "depth");
  auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  info->header = rgb->header;
  EPD::SynchronizationMetadata synchronization;
  synchronization.synchronized = true;
  synchronization.exact_sensor_stamp = true;
  synchronization.source_healthy = true;
  return store.publish("d435i", rgb, depth, info, synchronization);
}
}  // namespace

TEST(Observation, OwnsImmutableSynchronizedInputAndTruthfulIdentity)
{
  EPD::LatestObservationStore store;
  const auto observation = publish(store, 42);
  ASSERT_NE(observation, nullptr);
  EXPECT_EQ(observation->observation_id(), 1u);
  EXPECT_EQ(observation->sensor_stamp(), observation->rgb()->header.stamp);
  EXPECT_EQ(observation->frame_id(), "color");
  EXPECT_EQ(observation->camera_id(), "d435i");
  EXPECT_TRUE(observation->synchronization().synchronized);
  EXPECT_TRUE(observation->synchronization().exact_sensor_stamp);
  EXPECT_EQ(observation->aligned_depth()->header.stamp, observation->rgb()->header.stamp);
  EXPECT_EQ(observation->camera_info()->header.stamp, observation->rgb()->header.stamp);
}

TEST(Observation, OutputHeaderPreservesSourceStampAndFrame)
{
  EPD::LatestObservationStore store;
  const auto observation = publish(store, 1788040128);
  epd_msgs::msg::EPDObjectLocalization output;
  EPD::preserve_observation_header(output, *observation);
  EXPECT_EQ(output.header, observation->rgb()->header);
}

TEST(LatestObservationStore, IdsAreStrictlyMonotonicAndProcessLocal)
{
  EPD::LatestObservationStore first;
  EXPECT_EQ(publish(first, 1)->observation_id(), 1u);
  EXPECT_EQ(publish(first, 2)->observation_id(), 2u);
  EXPECT_EQ(publish(first, 3)->observation_id(), 3u);
  EPD::LatestObservationStore restarted_process;
  EXPECT_EQ(publish(restarted_process, 4)->observation_id(), 1u);
}

TEST(LatestObservationStore, SlowConsumerRetainsOnlyNewestObservation)
{
  EPD::LatestObservationStore store;
  for (int i = 1; i <= 1000; ++i) {
    publish(store, i);
  }
  ASSERT_EQ(store.retained_observation_count(), 1u);
  ASSERT_NE(store.latest(), nullptr);
  EXPECT_EQ(store.latest()->observation_id(), 1000u);
  EXPECT_EQ(store.latest()->sensor_stamp().sec, 1000);
  EXPECT_EQ(store.latest_after(999)->observation_id(), 1000u);
  EXPECT_EQ(store.latest_after(1000), nullptr);
}

TEST(LatestObservationStore, WaitCannotAcceptCachedObservationAtBaseline)
{
  EPD::LatestObservationStore store;
  const auto baseline = publish(store, 1)->observation_id();
  EXPECT_EQ(store.wait_for_newer(baseline, 20ms), nullptr);
}

TEST(LatestObservationStore, WaitWakesOnlyForStrictlyNewerObservation)
{
  EPD::LatestObservationStore store;
  const auto baseline = publish(store, 1)->observation_id();
  auto waiter = std::async(
    std::launch::async, [&store, baseline]() {
      return store.wait_for_newer(baseline, 1s);
    });
  EXPECT_EQ(waiter.wait_for(20ms), std::future_status::timeout);
  const auto next = publish(store, 2);
  ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(waiter.get()->observation_id(), next->observation_id());
  EXPECT_GT(next->observation_id(), baseline);
}

TEST(LatestObservationStore, ConsecutiveRequestsRequireDifferentNewObservations)
{
  EPD::LatestObservationStore store;
  const auto baseline_one = publish(store, 1)->observation_id();
  const auto result_one = publish(store, 2);
  ASSERT_EQ(store.latest_after(baseline_one), result_one);
  const auto baseline_two = store.latest_id();
  EXPECT_EQ(store.latest_after(baseline_two), nullptr);
  const auto result_two = publish(store, 3);
  ASSERT_EQ(store.latest_after(baseline_two), result_two);
  EXPECT_GT(result_two->observation_id(), result_one->observation_id());
}

TEST(LatestObservationStore, ShutdownUnblocksWaitWithoutStaleReplay)
{
  EPD::LatestObservationStore store;
  const auto baseline = publish(store, 1)->observation_id();
  auto waiter = std::async(
    std::launch::async, [&store, baseline]() {
      return store.wait_for_newer(baseline, 5s);
    });
  store.shutdown();
  ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(waiter.get(), nullptr);
  EXPECT_EQ(publish(store, 2), nullptr);
}

TEST(LatestObservationStore, ConcurrentProducerAndConsumerAreRaceSafe)
{
  EPD::LatestObservationStore store;
  std::atomic<bool> producer_done{false};
  std::atomic<bool> observed_regression{false};
  std::thread consumer([&]() {
      uint64_t previous = 0;
      while (!producer_done.load() || store.latest_id() < 2000u) {
        const auto current = store.latest();
        if (current) {
          if (current->observation_id() < previous) {
            observed_regression.store(true);
          }
          previous = current->observation_id();
        }
      }
    });
  std::thread producer([&]() {
      for (int i = 1; i <= 2000; ++i) {
        publish(store, i);
      }
      producer_done.store(true);
    });
  producer.join();
  consumer.join();
  EXPECT_FALSE(observed_regression.load());
  EXPECT_EQ(store.latest_id(), 2000u);
  EXPECT_EQ(store.retained_observation_count(), 1u);
}
