// Copyright 2026 Workcell Studio contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

#ifndef EPD_UTILS_LIB__ROS_MESSAGE_TRANSPORT_HPP_
#define EPD_UTILS_LIB__ROS_MESSAGE_TRANSPORT_HPP_

#include <stdexcept>
#include <string>

#include "opencv2/core.hpp"
#if __has_include("cv_bridge/cv_bridge.hpp")
#include "cv_bridge/cv_bridge.hpp"
#else
#include "cv_bridge/cv_bridge.h"
#endif
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/header.hpp"

namespace EPD
{

// Preserve the actual mask depth. Labelling an 8-bit or float mask as mono16
// changes the advertised row stride and makes consumers misread its bytes.
inline sensor_msgs::msg::Image::SharedPtr makeSegmentedMaskImage(
  const cv::Mat & mask, const std_msgs::msg::Header & source_header)
{
  if (mask.empty()) {
    throw std::invalid_argument("Cannot publish an empty segmented mask");
  }

  std::string encoding;
  switch (mask.type()) {
    case CV_8UC1:
      encoding = sensor_msgs::image_encodings::MONO8;
      break;
    case CV_16UC1:
      encoding = sensor_msgs::image_encodings::MONO16;
      break;
    case CV_32FC1:
      encoding = sensor_msgs::image_encodings::TYPE_32FC1;
      break;
    default:
      throw std::invalid_argument("Unsupported segmented mask pixel type");
  }

  // cv::Mat ROI views may have padded/non-contiguous rows; publish packed data.
  const cv::Mat packed = mask.isContinuous() ? mask : mask.clone();
  return cv_bridge::CvImage(source_header, encoding, packed).toImageMsg();
}

// PCL conversion does not guarantee the original sensor stamp or optical
// frame; the cloud is derived from the same synchronized RGB-D observation.
inline void setSegmentedCloudHeader(
  sensor_msgs::msg::PointCloud2 & cloud,
  const std_msgs::msg::Header & source_header)
{
  cloud.header = source_header;
}

}  // namespace EPD

#endif  // EPD_UTILS_LIB__ROS_MESSAGE_TRANSPORT_HPP_
