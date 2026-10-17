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

#ifndef EPD_UTILS_LIB__TIME_UTILS_HPP_
#define EPD_UTILS_LIB__TIME_UTILS_HPP_

#include <algorithm>
#include <cmath>

#include "builtin_interfaces/msg/time.hpp"
#include "rclcpp/time.hpp"

namespace EPD
{
inline double safeAgeMilliseconds(
  const rclcpp::Time & current, const rclcpp::Time & sample) noexcept
{
  if (sample.nanoseconds() == 0 || current.get_clock_type() != sample.get_clock_type()) {
    return -1.0;
  }
  const double age_ms = static_cast<double>(current.nanoseconds() - sample.nanoseconds()) / 1e6;
  return std::isfinite(age_ms) ? std::max(0.0, age_ms) : -1.0;
}

inline double sensorStampAgeMilliseconds(
  const rclcpp::Time & current, const builtin_interfaces::msg::Time & sensor_stamp) noexcept
{
  if (sensor_stamp.sec == 0 && sensor_stamp.nanosec == 0) {
    return -1.0;
  }
  return safeAgeMilliseconds(
    current, rclcpp::Time(sensor_stamp, current.get_clock_type()));
}
}  // namespace EPD

#endif  // EPD_UTILS_LIB__TIME_UTILS_HPP_
