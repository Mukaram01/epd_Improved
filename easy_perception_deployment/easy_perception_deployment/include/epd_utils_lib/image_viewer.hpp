// Copyright 2022 Advanced Remanufacturing and Technology Centre
// Copyright 2022 ROS-Industrial Consortium Asia Pacific Team
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

#ifndef EPD_UTILS_LIB__IMAGE_VIEWER_HPP_
#define EPD_UTILS_LIB__IMAGE_VIEWER_HPP_

#include <vector>
#include <string>
#include <memory>
#include <mutex>

#if __has_include("cv_bridge/cv_bridge.hpp")
#include "cv_bridge/cv_bridge.hpp"
#else
#include "cv_bridge/cv_bridge.h"
#endif
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include "opencv2/opencv.hpp"

/*! \class ImageViewer
    \brief An ImageViewer class object.
    The ImageViewer class object inherits from the rclcpp::Node object to
    provide a localized way of viewing the output inference visualization
    results.

    Frames are buffered in the subscription callback and rendered by a
    dedicated 30 Hz wall timer.  This decouples OpenCV GUI event processing
    (cv::imshow / cv::waitKey) from the ROS 2 executor threads, preventing
    the display window from freezing when frames arrive at irregular intervals
    or when inference is slow.
*/
class ImageViewer : public rclcpp::Node
{
public:
  /*! \brief A Constructor function*/
  ImageViewer();

private:
  /*! \brief A subscriber member variable to receive images.*/
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_1_;
  /*! \brief A subscriber member variable to receive remote calls to shutdown.*/
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_2_;

  /*! \brief Timer that drives the OpenCV display loop at ~30 Hz.*/
  rclcpp::TimerBase::SharedPtr display_timer_;

  /*! \brief Mutex protecting latest_frame_ and frame_available_.*/
  std::mutex frame_mutex_;
  /*! \brief The most recently received (and decoded) frame.*/
  cv::Mat latest_frame_;
  /*! \brief Set to true when a new frame is waiting to be shown.*/
  bool frame_available_{false};

  /*! \brief A Mutator function that sets the appropriate image encodings for
  displaying input images.
  */
  int encoding2mat_type(const std::string & encoding) const;
  /*! \brief A ROS2 callback function utilized by sub_1.
  Decodes the incoming image message and stores it; does NOT call cv::imshow.
  */
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  /*! \brief Timer callback: calls cv::imshow + cv::waitKey to refresh the
  display window at a stable cadence.
  */
  void display_callback();
};

ImageViewer::ImageViewer()
: Node("image_viewer")
{
  cv::namedWindow("image_viewer", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("image_viewer", 0, 375);
  cv::waitKey(1);

  // Keep only the latest frame in the subscription queue so the display
  // always shows the most recent inference result.
  auto qos = rclcpp::SensorDataQoS().keep_last(1);

  sub_1_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/image_viewer/image_input",
    qos, std::bind(&ImageViewer::image_callback, this, std::placeholders::_1));

  // Drive the OpenCV GUI at ~30 Hz independently of the subscription rate.
  using namespace std::chrono_literals;
  display_timer_ = this->create_wall_timer(
    33ms, std::bind(&ImageViewer::display_callback, this));
}

int ImageViewer::encoding2mat_type(const std::string & encoding) const
{
  if (encoding == "mono8") {
    return CV_8UC1;
  } else if (encoding == "bgr8") {
    return CV_8UC3;
  } else if (encoding == sensor_msgs::image_encodings::MONO16) {
    return CV_16UC1;
  } else if (encoding == "rgba8") {
    return CV_8UC4;
  } else if (encoding == "bgra8") {
    return CV_8UC4;
  } else if (encoding == "32FC1") {
    return CV_32FC1;
  } else if (encoding == "rgb8") {
    return CV_8UC3;
  } else {
    throw std::runtime_error("Unsupported encoding type");
  }
}

void ImageViewer::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  cv::Mat frame(
    msg->height, msg->width, this->encoding2mat_type(msg->encoding),
    const_cast<unsigned char *>(msg->data.data()), msg->step);

  if (msg->encoding == "rgb8") {
    cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
  }

  // Deep-copy so the decoded pixels outlive the ROS message buffer.
  std::lock_guard<std::mutex> lock(frame_mutex_);
  latest_frame_ = frame.clone();
  frame_available_ = true;
}

void ImageViewer::display_callback()
{
  cv::Mat frame_to_show;
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (frame_available_) {
      frame_to_show = latest_frame_.clone();
      frame_available_ = false;
    }
  }

  if (!frame_to_show.empty()) {
    cv::imshow("image_viewer", frame_to_show);
  }
  // Always pump OpenCV GUI events so the window stays responsive.
  cv::waitKey(1);
}

#endif  // EPD_UTILS_LIB__IMAGE_VIEWER_HPP_
