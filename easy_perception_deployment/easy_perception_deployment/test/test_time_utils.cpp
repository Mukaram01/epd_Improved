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
