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

#ifndef EPD_UTILS_LIB__INFERENCE_SCHEDULER_HPP_
#define EPD_UTILS_LIB__INFERENCE_SCHEDULER_HPP_

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "epd_msgs/msg/epd_object_localization.hpp"
#include "epd_msgs/msg/epd_object_tracking.hpp"
#include "epd_utils_lib/observation.hpp"

namespace EPD
{
struct InferenceMetrics
{
  uint64_t observations_published{0};
  uint64_t inference_started{0};
  uint64_t inference_completed{0};
  uint64_t inference_failed{0};
  uint64_t last_consumed_observation_id{0};
  uint64_t last_completed_observation_id{0};
  uint64_t observations_skipped_before_inference{0};
  uint64_t duplicate_or_regressed_submissions{0};
  uint64_t duplicate_processing_attempts{0};
  uint64_t backlog_high_water_mark{0};
  uint64_t last_submitted_observation_id{0};
  int64_t last_latency_ms{0};
  int64_t minimum_latency_ms{0};
  int64_t maximum_latency_ms{0};
  int64_t total_latency_ms{0};
  bool worker_busy{false};
  std::size_t backlog_size{0};
  builtin_interfaces::msg::Time newest_observation_stamp;
  builtin_interfaces::msg::Time newest_result_stamp;
};

class LatestInferenceScheduler final
{
public:
  void submit(const Observation::ConstSharedPtr & observation)
  {
    if (!observation) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        return;
      }
      ++metrics_.observations_published;
      if (observation->observation_id() <= metrics_.last_submitted_observation_id) {
        ++metrics_.duplicate_or_regressed_submissions;
        return;
      }
      metrics_.last_submitted_observation_id = observation->observation_id();
      metrics_.newest_observation_stamp = observation->sensor_stamp();
      latest_pending_ = observation;
      metrics_.backlog_size = 1;
      metrics_.backlog_high_water_mark = std::max<uint64_t>(
        metrics_.backlog_high_water_mark, metrics_.backlog_size);
    }
    cv_.notify_one();
  }

  template<typename Rep, typename Period>
  Observation::ConstSharedPtr wait_for_next(
    const std::chrono::duration<Rep, Period> & timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_for(
      lock, timeout, [this]() {
        return shutdown_ || (latest_pending_ &&
        latest_pending_->observation_id() > metrics_.last_consumed_observation_id);
      });
    if (shutdown_ || !latest_pending_) {
      return nullptr;
    }
    const uint64_t id = latest_pending_->observation_id();
    if (id <= metrics_.last_consumed_observation_id) {
      ++metrics_.duplicate_processing_attempts;
      latest_pending_.reset();
      metrics_.backlog_size = 0;
      return nullptr;
    }
    if (id > metrics_.last_consumed_observation_id + 1) {
      metrics_.observations_skipped_before_inference +=
        id - metrics_.last_consumed_observation_id - 1;
    }
    metrics_.last_consumed_observation_id = id;
    ++metrics_.inference_started;
    metrics_.worker_busy = true;
    auto next = latest_pending_;
    latest_pending_.reset();
    metrics_.backlog_size = 0;
    return next;
  }

  void complete(const Observation & observation, int64_t latency_ms)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (observation.observation_id() != metrics_.last_consumed_observation_id) {
      ++metrics_.duplicate_processing_attempts;
      return;
    }
    ++metrics_.inference_completed;
    metrics_.last_completed_observation_id = observation.observation_id();
    metrics_.newest_result_stamp = observation.sensor_stamp();
    record_latency(latency_ms);
    metrics_.worker_busy = false;
  }

  void fail(const Observation & observation, int64_t latency_ms)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (observation.observation_id() != metrics_.last_consumed_observation_id) {
      ++metrics_.duplicate_processing_attempts;
      return;
    }
    ++metrics_.inference_failed;
    record_latency(latency_ms);
    metrics_.worker_busy = false;
  }

  InferenceMetrics metrics() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_;
  }

  std::size_t retained_observation_count() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_pending_ ? 1U : 0U;
  }

  void shutdown()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
      latest_pending_.reset();
      metrics_.backlog_size = 0;
    }
    cv_.notify_all();
  }

private:
  void record_latency(int64_t latency_ms)
  {
    metrics_.last_latency_ms = latency_ms;
    metrics_.total_latency_ms += latency_ms;
    if (metrics_.inference_completed + metrics_.inference_failed == 1) {
      metrics_.minimum_latency_ms = latency_ms;
      metrics_.maximum_latency_ms = latency_ms;
    } else {
      metrics_.minimum_latency_ms = std::min(metrics_.minimum_latency_ms, latency_ms);
      metrics_.maximum_latency_ms = std::max(metrics_.maximum_latency_ms, latency_ms);
    }
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  Observation::ConstSharedPtr latest_pending_;
  InferenceMetrics metrics_;
  bool shutdown_{false};
};

struct PerceptionResult
{
  uint64_t source_observation_id{0};
  builtin_interfaces::msg::Time sensor_stamp;
  std::string frame_id;
  bool success{false};
  epd_msgs::msg::EPDObjectLocalization localization;
  epd_msgs::msg::EPDObjectTracking tracking;
  std::chrono::steady_clock::time_point completed_at;
};

struct PerceptionResultStoreMetrics
{
  uint64_t results_completed{0};
  uint64_t latest_completed_result_observation_id{0};
  uint64_t result_store_regressions{0};
  uint64_t duplicate_result_publish{0};
  std::size_t current_waiters{0};
  bool shutdown{false};
};

class LatestPerceptionResultStore final
{
public:
  bool publish(PerceptionResult result)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {return false;}
      if (latest_ && result.source_observation_id < latest_->source_observation_id) {
        ++metrics_.result_store_regressions;
        return false;
      }
      if (latest_ && result.source_observation_id == latest_->source_observation_id) {
        ++metrics_.duplicate_result_publish;
        return false;
      }
      result.completed_at = std::chrono::steady_clock::now();
      latest_ = std::make_shared<const PerceptionResult>(std::move(result));
      ++metrics_.results_completed;
      metrics_.latest_completed_result_observation_id = latest_->source_observation_id;
    }
    cv_.notify_all();
    return true;
  }

  std::shared_ptr<const PerceptionResult> latest() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
  }

  std::shared_ptr<const PerceptionResult> latest_after(uint64_t baseline) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_ && latest_->source_observation_id > baseline ? latest_ : nullptr;
  }

  std::size_t retained_result_count() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_ ? 1U : 0U;
  }

  std::shared_ptr<const PerceptionResult> wait_for_result_after(
    uint64_t baseline, std::chrono::nanoseconds timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    ++metrics_.current_waiters;
    cv_.wait_for(
      lock, timeout, [this, baseline]() {
        return shutdown_ || (latest_ && latest_->source_observation_id > baseline);
      });
    --metrics_.current_waiters;
    return !shutdown_ && latest_ && latest_->source_observation_id > baseline ? latest_ : nullptr;
  }

  void shutdown()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
      metrics_.shutdown = true;
    }
    cv_.notify_all();
  }

  PerceptionResultStoreMetrics metrics() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::shared_ptr<const PerceptionResult> latest_;
  PerceptionResultStoreMetrics metrics_;
  bool shutdown_{false};
};
}  // namespace EPD

#endif  // EPD_UTILS_LIB__INFERENCE_SCHEDULER_HPP_
