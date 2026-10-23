// Copyright 2026 Easy Perception Deployment contributors
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

#include "gtest/gtest.h"
#include "epd_utils_lib/time_utils.hpp"

TEST(TimeUtils, SameClockSubtractionWorks)
{
  const rclcpp::Time now(5'000'000'000LL, RCL_ROS_TIME);
  const rclcpp::Time sample(3'500'000'000LL, RCL_ROS_TIME);
  EXPECT_DOUBLE_EQ(EPD::safeAgeMilliseconds(now, sample), 1500.0);
}

TEST(TimeUtils, MixedClockReturnsUnavailableWithoutThrowing)
{
  const rclcpp::Time now(5'000'000'000LL, RCL_ROS_TIME);
  const rclcpp::Time sample(3'500'000'000LL, RCL_SYSTEM_TIME);
  EXPECT_NO_THROW(EXPECT_DOUBLE_EQ(EPD::safeAgeMilliseconds(now, sample), -1.0));
}

TEST(TimeUtils, SensorSourceStampRemainsUnchanged)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 3;
  stamp.nanosec = 250'000'000U;
  const auto original = stamp;
  const rclcpp::Time now(5'000'000'000LL, RCL_ROS_TIME);
  EXPECT_DOUBLE_EQ(EPD::sensorStampAgeMilliseconds(now, stamp), 1750.0);
  EXPECT_EQ(stamp.sec, original.sec);
  EXPECT_EQ(stamp.nanosec, original.nanosec);
}

TEST(TimeUtils, ValidAgeIsFiniteAndNonnegative)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 6;
  const rclcpp::Time now(5'000'000'000LL, RCL_ROS_TIME);
  const double age = EPD::sensorStampAgeMilliseconds(now, stamp);
  EXPECT_TRUE(std::isfinite(age));
  EXPECT_GE(age, 0.0);
}
