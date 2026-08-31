// Copyright 2026 Advanced Remanufacturing and Technology Centre
// Licensed under the Apache License, Version 2.0

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
