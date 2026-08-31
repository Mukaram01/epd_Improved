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

#ifndef EPD_UTILS_LIB__OBSERVATION_HPP_
#define EPD_UTILS_LIB__OBSERVATION_HPP_

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "builtin_interfaces/msg/time.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace EPD
{
struct SynchronizationMetadata
{
  bool synchronized{false};
  bool exact_sensor_stamp{false};
  int64_t maximum_skew_ns{0};
  bool source_healthy{true};
  bool tf_available{false};
};

class Observation final
{
public:
  using ConstSharedPtr = std::shared_ptr<const Observation>;

  Observation(
    uint64_t observation_id,
    std::string camera_id,
    sensor_msgs::msg::Image::ConstSharedPtr rgb,
    sensor_msgs::msg::Image::ConstSharedPtr aligned_depth,
    sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info,
    SynchronizationMetadata synchronization)
  : observation_id_(observation_id),
    sensor_stamp_(rgb ? rgb->header.stamp : builtin_interfaces::msg::Time()),
    camera_id_(std::move(camera_id)),
    frame_id_(rgb ? rgb->header.frame_id : std::string()),
    rgb_(std::move(rgb)),
    aligned_depth_(std::move(aligned_depth)),
    camera_info_(std::move(camera_info)),
    synchronization_(synchronization)
  {}

  uint64_t observation_id() const {return observation_id_;}
  const builtin_interfaces::msg::Time & sensor_stamp() const {return sensor_stamp_;}
  const std::string & camera_id() const {return camera_id_;}
  const std::string & frame_id() const {return frame_id_;}
  const sensor_msgs::msg::Image::ConstSharedPtr & rgb() const {return rgb_;}
  const sensor_msgs::msg::Image::ConstSharedPtr & aligned_depth() const
  {
    return aligned_depth_;
  }
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info() const
  {
    return camera_info_;
  }
  const SynchronizationMetadata & synchronization() const {return synchronization_;}

private:
  const uint64_t observation_id_;
  const builtin_interfaces::msg::Time sensor_stamp_;
  const std::string camera_id_;
  const std::string frame_id_;
  const sensor_msgs::msg::Image::ConstSharedPtr rgb_;
  const sensor_msgs::msg::Image::ConstSharedPtr aligned_depth_;
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_;
  const SynchronizationMetadata synchronization_;
};

class LatestObservationStore final
{
public:
  Observation::ConstSharedPtr publish(
    const std::string & camera_id,
    sensor_msgs::msg::Image::ConstSharedPtr rgb,
    sensor_msgs::msg::Image::ConstSharedPtr aligned_depth,
    sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info,
    const SynchronizationMetadata & synchronization)
  {
    Observation::ConstSharedPtr observation;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {return nullptr;}
      observation = std::make_shared<const Observation>(
        ++last_observation_id_, camera_id, std::move(rgb), std::move(aligned_depth),
        std::move(camera_info), synchronization);
      latest_ = observation;
    }
    cv_.notify_all();
    return observation;
  }

  Observation::ConstSharedPtr latest() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
  }

  Observation::ConstSharedPtr latest_after(uint64_t observation_id) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_ && latest_->observation_id() > observation_id ? latest_ : nullptr;
  }

  template<typename Rep, typename Period>
  Observation::ConstSharedPtr wait_for_newer(
    uint64_t observation_id,
    const std::chrono::duration<Rep, Period> & timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_for(
      lock, timeout, [this, observation_id]() {
        return shutdown_ || (latest_ && latest_->observation_id() > observation_id);
      });
    return latest_ && latest_->observation_id() > observation_id ? latest_ : nullptr;
  }

  uint64_t latest_id() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_ ? latest_->observation_id() : 0;
  }

  std::size_t retained_observation_count() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_ ? 1U : 0U;
  }

  void shutdown()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    cv_.notify_all();
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  Observation::ConstSharedPtr latest_;
  uint64_t last_observation_id_{0};
  bool shutdown_{false};
};

template<typename OutputT>
void preserve_observation_header(OutputT & output, const Observation & observation)
{
  if (observation.rgb()) {
    output.header = observation.rgb()->header;
  }
}
}  // namespace EPD

#endif  // EPD_UTILS_LIB__OBSERVATION_HPP_
