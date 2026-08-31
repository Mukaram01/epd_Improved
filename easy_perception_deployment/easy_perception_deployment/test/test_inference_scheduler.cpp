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

#include <chrono>
#include <future>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "epd_utils_lib/inference_scheduler.hpp"

using namespace std::chrono_literals;

namespace
{
EPD::Observation::ConstSharedPtr observation(EPD::LatestObservationStore & store, int id)
{
  auto rgb = std::make_shared<sensor_msgs::msg::Image>();
  auto depth = std::make_shared<sensor_msgs::msg::Image>();
  auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  rgb->header.stamp.sec = id;
  rgb->header.stamp.nanosec = 123;
  rgb->header.frame_id = "camera_color_optical_frame";
  depth->header = rgb->header;
  info->header = rgb->header;
  EPD::SynchronizationMetadata sync;
  sync.synchronized = true;
  sync.exact_sensor_stamp = true;
  return store.publish("d435i", rgb, depth, info, sync);
}

EPD::PerceptionResult result(const EPD::Observation & input)
{
  EPD::PerceptionResult output;
  output.source_observation_id = input.observation_id();
  output.sensor_stamp = input.sensor_stamp();
  output.frame_id = input.frame_id();
  output.success = true;
  return output;
}
}  // namespace

TEST(InferenceScheduler, FastProducerSlowConsumerStaysBounded)
{
  EPD::LatestObservationStore observations;
  EPD::LatestInferenceScheduler scheduler;
  for (int id = 1; id <= 1000; ++id) {
    scheduler.submit(observation(observations, id));
  }
  EXPECT_EQ(scheduler.retained_observation_count(), 1u);
  EXPECT_EQ(scheduler.metrics().backlog_high_water_mark, 1u);
}

TEST(InferenceScheduler, BusyWorkerTakesNewestInsteadOfBacklog)
{
  EPD::LatestObservationStore observations;
  EPD::LatestInferenceScheduler scheduler;
  scheduler.submit(observation(observations, 1));
  const auto first = scheduler.wait_for_next(10ms);
  ASSERT_EQ(first->observation_id(), 1u);
  for (int id = 2; id <= 100; ++id) {
    scheduler.submit(observation(observations, id));
  }
  scheduler.complete(*first, 50);
  const auto next = scheduler.wait_for_next(10ms);
  ASSERT_EQ(next->observation_id(), 100u);
  EXPECT_EQ(scheduler.metrics().observations_skipped_before_inference, 98u);
}

TEST(InferenceScheduler, ConsumedIdNeverRegresses)
{
  EPD::LatestObservationStore observations;
  EPD::LatestInferenceScheduler scheduler;
  const auto one = observation(observations, 1);
  const auto two = observation(observations, 2);
  scheduler.submit(two);
  ASSERT_EQ(scheduler.wait_for_next(10ms)->observation_id(), 2u);
  scheduler.submit(one);
  EXPECT_EQ(scheduler.wait_for_next(10ms), nullptr);
  EXPECT_EQ(scheduler.metrics().last_consumed_observation_id, 2u);
  EXPECT_EQ(scheduler.metrics().duplicate_or_regressed_submissions, 1u);
}

TEST(InferenceScheduler, SameObservationIsNotProcessedTwice)
{
  EPD::LatestObservationStore observations;
  EPD::LatestInferenceScheduler scheduler;
  const auto one = observation(observations, 1);
  scheduler.submit(one);
  ASSERT_NE(scheduler.wait_for_next(10ms), nullptr);
  scheduler.complete(*one, 1);
  scheduler.submit(one);
  EXPECT_EQ(scheduler.wait_for_next(10ms), nullptr);
  EXPECT_EQ(scheduler.metrics().inference_started, 1u);
}

TEST(InferenceScheduler, SkippedAccountingIncludesInitialOverwrite)
{
  EPD::LatestObservationStore observations;
  EPD::LatestInferenceScheduler scheduler;
  for (int id = 1; id <= 10; ++id) {
    scheduler.submit(observation(observations, id));
  }
  ASSERT_EQ(scheduler.wait_for_next(10ms)->observation_id(), 10u);
  EXPECT_EQ(scheduler.metrics().observations_skipped_before_inference, 9u);
}

TEST(InferenceScheduler, WorkerSleepsAndWakesForSubmission)
{
  EPD::LatestObservationStore observations;
  EPD::LatestInferenceScheduler scheduler;
  auto waiter = std::async(
    std::launch::async, [&scheduler]() {
      return scheduler.wait_for_next(1s);
    });
  EXPECT_EQ(waiter.wait_for(20ms), std::future_status::timeout);
  scheduler.submit(observation(observations, 1));
  ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(waiter.get()->observation_id(), 1u);
}

TEST(InferenceScheduler, ShutdownUnblocksWorker)
{
  EPD::LatestInferenceScheduler scheduler;
  auto waiter = std::async(
    std::launch::async, [&scheduler]() {
      return scheduler.wait_for_next(5s);
    });
  scheduler.shutdown();
  ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(waiter.get(), nullptr);
}

TEST(PerceptionResultStore, BaselineCannotAcceptOldResult)
{
  EPD::LatestObservationStore observations;
  EPD::LatestPerceptionResultStore results;
  const auto input = observation(observations, 1);
  results.publish(result(*input));
  EXPECT_EQ(results.latest_after(input->observation_id()), nullptr);
}

TEST(PerceptionResultStore, NewlyCompletedContinuousResultSatisfiesBaseline)
{
  EPD::LatestObservationStore observations;
  EPD::LatestPerceptionResultStore results;
  const auto baseline = observation(observations, 1)->observation_id();
  const auto next = observation(observations, 2);
  results.publish(result(*next));
  ASSERT_NE(results.latest_after(baseline), nullptr);
  EXPECT_EQ(results.latest_after(baseline)->source_observation_id, next->observation_id());
}

TEST(PerceptionResultStore, ConsecutiveRequestsRequireNewerResults)
{
  EPD::LatestObservationStore observations;
  EPD::LatestPerceptionResultStore results;
  const auto first_baseline = observation(observations, 1)->observation_id();
  const auto first_result = observation(observations, 2);
  results.publish(result(*first_result));
  ASSERT_EQ(results.latest_after(first_baseline)->source_observation_id, 2u);
  const auto second_baseline = first_result->observation_id();
  EXPECT_EQ(results.latest_after(second_baseline), nullptr);
  const auto second_result = observation(observations, 3);
  results.publish(result(*second_result));
  EXPECT_EQ(results.latest_after(second_baseline)->source_observation_id, 3u);
}

TEST(InferenceScheduler, FailureDoesNotKillSubsequentWork)
{
  EPD::LatestObservationStore observations;
  EPD::LatestInferenceScheduler scheduler;
  scheduler.submit(observation(observations, 1));
  const auto failed = scheduler.wait_for_next(10ms);
  scheduler.fail(*failed, 5);
  scheduler.submit(observation(observations, 2));
  const auto recovered = scheduler.wait_for_next(10ms);
  scheduler.complete(*recovered, 6);
  const auto metrics = scheduler.metrics();
  EXPECT_EQ(metrics.inference_failed, 1u);
  EXPECT_EQ(metrics.inference_completed, 1u);
  EXPECT_EQ(metrics.last_completed_observation_id, 2u);
}

TEST(InferenceScheduler, LongProducerRunRetainsOneObservationAndResult)
{
  EPD::LatestObservationStore observations;
  EPD::LatestInferenceScheduler scheduler;
  EPD::LatestPerceptionResultStore results;
  for (int id = 1; id <= 10000; ++id) {
    const auto input = observation(observations, id);
    scheduler.submit(input);
    results.publish(result(*input));
  }
  EXPECT_EQ(observations.retained_observation_count(), 1u);
  EXPECT_EQ(scheduler.retained_observation_count(), 1u);
  EXPECT_EQ(results.retained_result_count(), 1u);
}

TEST(PerceptionResultStore, OriginalSensorStampAndFrameSurviveResult)
{
  EPD::LatestObservationStore observations;
  EPD::LatestPerceptionResultStore results;
  const auto input = observation(observations, 42);
  results.publish(result(*input));
  const auto output = results.latest();
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->sensor_stamp, input->sensor_stamp());
  EXPECT_EQ(output->frame_id, input->frame_id());
  EXPECT_EQ(output->source_observation_id, input->observation_id());
}

TEST(PerceptionResultStore, LatestOnlyRejectsRegressionAndDuplicate)
{
  EPD::LatestObservationStore observations;
  EPD::LatestPerceptionResultStore results;
  const auto one = observation(observations, 1);
  const auto two = observation(observations, 2);
  EXPECT_TRUE(results.publish(result(*two)));
  EXPECT_FALSE(results.publish(result(*one)));
  EXPECT_FALSE(results.publish(result(*two)));
  EXPECT_EQ(results.latest()->source_observation_id, 2U);
  EXPECT_EQ(results.retained_result_count(), 1U);
  EXPECT_EQ(results.metrics().result_store_regressions, 1U);
  EXPECT_EQ(results.metrics().duplicate_result_publish, 1U);
}

TEST(PerceptionResultStore, TimeoutDoesNotPreventLaterSuccess)
{
  EPD::LatestObservationStore observations;
  EPD::LatestPerceptionResultStore results;
  EXPECT_EQ(results.wait_for_result_after(1, 10ms), nullptr);
  observation(observations, 1);
  const auto two = observation(observations, 2);
  results.publish(result(*two));
  ASSERT_NE(results.wait_for_result_after(1, 10ms), nullptr);
  EXPECT_EQ(results.wait_for_result_after(1, 10ms)->source_observation_id, 2U);
}

TEST(PerceptionResultStore, ShutdownWakesWaiter)
{
  EPD::LatestPerceptionResultStore results;
  auto waiter = std::async(
    std::launch::async, [&results]() {
      return results.wait_for_result_after(1, 5s);
    });
  EXPECT_EQ(waiter.wait_for(20ms), std::future_status::timeout);
  results.shutdown();
  ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(waiter.get(), nullptr);
  EXPECT_EQ(results.metrics().current_waiters, 0U);
}

TEST(PerceptionResultStore, FreshZeroDetectionResultIsSuccessful)
{
  EPD::LatestObservationStore observations;
  EPD::LatestPerceptionResultStore results;
  const auto input = observation(observations, 1);
  auto empty = result(*input);
  empty.localization.header = input->rgb()->header;
  ASSERT_TRUE(results.publish(std::move(empty)));
  const auto output = results.latest_after(0);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(output->success);
  EXPECT_TRUE(output->localization.objects.empty());
}

TEST(PerceptionResultStore, LocalizationAndTrackingPayloadShareSourceTruth)
{
  EPD::LatestObservationStore observations;
  EPD::LatestPerceptionResultStore results;
  const auto input = observation(observations, 7);
  auto completed = result(*input);
  completed.localization.header = input->rgb()->header;
  completed.tracking.header = input->rgb()->header;
  results.publish(std::move(completed));
  const auto output = results.latest();
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->localization.header.stamp, output->sensor_stamp);
  EXPECT_EQ(output->tracking.header.stamp, output->sensor_stamp);
  EXPECT_EQ(output->localization.header.frame_id, output->frame_id);
  EXPECT_EQ(output->tracking.header.frame_id, output->frame_id);
}

TEST(PerceptionResultStore, ConsecutiveWaitsNeedSuccessivelyFreshResults)
{
  EPD::LatestObservationStore observations;
  EPD::LatestPerceptionResultStore results;
  const auto one = observation(observations, 1);
  const auto two = observation(observations, 2);
  results.publish(result(*one));
  ASSERT_EQ(results.wait_for_result_after(0, 10ms)->source_observation_id, 1U);
  EXPECT_EQ(results.wait_for_result_after(1, 10ms), nullptr);
  results.publish(result(*two));
  ASSERT_EQ(results.wait_for_result_after(1, 10ms)->source_observation_id, 2U);
}

TEST(P7Handshake, ConsumerBusyOrAbsentDoesNotStopObservationAndInferenceProgress)
{
  EPD::LatestObservationStore observations;
  EPD::LatestInferenceScheduler scheduler;
  EPD::LatestPerceptionResultStore results;

  for (int id = 1; id <= 3; ++id) {
    const auto input = observation(observations, id);
    scheduler.submit(input);
    const auto work = scheduler.wait_for_next(10ms);
    ASSERT_NE(work, nullptr);
    results.publish(result(*work));
    scheduler.complete(*work, 1);
  }

  EXPECT_EQ(observations.latest_id(), 3U);
  EXPECT_EQ(scheduler.metrics().inference_completed, 3U);
  EXPECT_EQ(results.latest()->source_observation_id, 3U);
}

TEST(P7Handshake, FreshRequestAfterBusyCannotReuseItsBaseline)
{
  EPD::LatestObservationStore observations;
  EPD::LatestPerceptionResultStore results;
  const auto baseline_input = observation(observations, 1);
  results.publish(result(*baseline_input));
  const uint64_t baseline = observations.latest_id();

  EXPECT_EQ(results.latest_after(baseline), nullptr);
  const auto fresh_input = observation(observations, 2);
  results.publish(result(*fresh_input));
  ASSERT_NE(results.latest_after(baseline), nullptr);
  EXPECT_GT(results.latest_after(baseline)->source_observation_id, baseline);
}

TEST(P7Handshake, EpdRestartHasAnEmptyResultTimeline)
{
  EPD::LatestObservationStore old_observations;
  EPD::LatestPerceptionResultStore old_results;
  const auto old_input = observation(old_observations, 1);
  old_results.publish(result(*old_input));

  EPD::LatestObservationStore restarted_observations;
  EPD::LatestPerceptionResultStore restarted_results;
  EXPECT_EQ(restarted_results.latest(), nullptr);
  const uint64_t restart_baseline = restarted_observations.latest_id();
  const auto new_input = observation(restarted_observations, 2);
  restarted_results.publish(result(*new_input));
  ASSERT_NE(restarted_results.latest_after(restart_baseline), nullptr);
  EXPECT_EQ(restarted_results.latest_after(restart_baseline)->sensor_stamp.sec, 2);
}

TEST(P7Handshake, EmdRestartDoesNotAlterEpdOwnedTimeline)
{
  EPD::LatestObservationStore observations;
  EPD::LatestPerceptionResultStore results;
  results.publish(result(*observation(observations, 1)));
  const uint64_t before_consumer_restart = observations.latest_id();

  results.publish(result(*observation(observations, 2)));
  EXPECT_GT(observations.latest_id(), before_consumer_restart);
  EXPECT_EQ(results.latest()->source_observation_id, 2U);
}
