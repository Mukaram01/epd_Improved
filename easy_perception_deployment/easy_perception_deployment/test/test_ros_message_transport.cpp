// Copyright 2026 Workcell Studio contributors
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

#include <cstring>
#include <stdexcept>

#include "gtest/gtest.h"
#include "epd_utils_lib/ros_message_transport.hpp"

namespace
{

std_msgs::msg::Header sourceHeader()
{
  std_msgs::msg::Header header;
  header.frame_id = "camera_color_optical_frame";
  header.stamp.sec = 123;
  header.stamp.nanosec = 456789U;
  return header;
}

TEST(RosMessageTransport, PackedMono8PreservesPixelBytesAndSensorHeader)
{
  cv::Mat mask(2, 3, CV_8UC1, cv::Scalar(0));
  mask.at<uint8_t>(0, 1) = 255;
  mask.at<uint8_t>(1, 2) = 127;
  const auto msg = EPD::makeSegmentedMaskImage(mask, sourceHeader());

  ASSERT_NE(msg, nullptr);
  EXPECT_EQ(msg->encoding, sensor_msgs::image_encodings::MONO8);
  EXPECT_EQ(msg->width, 3U);
  EXPECT_EQ(msg->height, 2U);
  EXPECT_EQ(msg->step, 3U);
  ASSERT_EQ(msg->data.size(), 6U);
  EXPECT_EQ(msg->data[1], 255U);
  EXPECT_EQ(msg->data[5], 127U);
  EXPECT_EQ(msg->header.frame_id, sourceHeader().frame_id);
  EXPECT_EQ(msg->header.stamp, sourceHeader().stamp);
}

TEST(RosMessageTransport, NonContiguousRoiPacksOnlySelectedPixels)
{
  cv::Mat large(4, 7, CV_8UC1, cv::Scalar(99));
  cv::Mat roi = large(cv::Rect(2, 1, 3, 2));
  ASSERT_FALSE(roi.isContinuous());
  roi.at<uint8_t>(0, 0) = 1;
  roi.at<uint8_t>(0, 1) = 2;
  roi.at<uint8_t>(0, 2) = 3;
  roi.at<uint8_t>(1, 0) = 4;
  roi.at<uint8_t>(1, 1) = 5;
  roi.at<uint8_t>(1, 2) = 6;

  const auto msg = EPD::makeSegmentedMaskImage(roi, sourceHeader());
  EXPECT_EQ(msg->step, 3U);
  const std::vector<uint8_t> expected{1, 2, 3, 4, 5, 6};
  EXPECT_EQ(msg->data, expected);
}

TEST(RosMessageTransport, Mono16UsesTwoBytesPerPixel)
{
  cv::Mat mask(2, 2, CV_16UC1, cv::Scalar(0));
  mask.at<uint16_t>(0, 1) = 65535U;
  const auto msg = EPD::makeSegmentedMaskImage(mask, sourceHeader());
  EXPECT_EQ(msg->encoding, sensor_msgs::image_encodings::MONO16);
  EXPECT_EQ(msg->step, 4U);
  EXPECT_EQ(msg->data.size(), 8U);
  uint16_t recovered = 0;
  std::memcpy(&recovered, msg->data.data() + sizeof(uint16_t), sizeof(uint16_t));
  EXPECT_EQ(recovered, 65535U);
}

TEST(RosMessageTransport, FloatMaskRetainsTypeAndStride)
{
  cv::Mat mask(2, 2, CV_32FC1, cv::Scalar(0.5F));
  const auto msg = EPD::makeSegmentedMaskImage(mask, sourceHeader());
  EXPECT_EQ(msg->encoding, sensor_msgs::image_encodings::TYPE_32FC1);
  EXPECT_EQ(msg->step, 8U);
  EXPECT_EQ(msg->data.size(), 16U);
}

TEST(RosMessageTransport, EmptyAndUnsupportedMasksAreRejected)
{
  EXPECT_THROW(
    EPD::makeSegmentedMaskImage(cv::Mat(), sourceHeader()), std::invalid_argument);
  cv::Mat colour_mask(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
  EXPECT_THROW(
    EPD::makeSegmentedMaskImage(colour_mask, sourceHeader()), std::invalid_argument);
}

TEST(RosMessageTransport, CloudUsesExactObservationFrameAndStamp)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.frame_id = "stale_frame";
  cloud.header.stamp.sec = 1;
  EPD::setSegmentedCloudHeader(cloud, sourceHeader());
  EXPECT_EQ(cloud.header.frame_id, sourceHeader().frame_id);
  EXPECT_EQ(cloud.header.stamp, sourceHeader().stamp);
}

}  // namespace
