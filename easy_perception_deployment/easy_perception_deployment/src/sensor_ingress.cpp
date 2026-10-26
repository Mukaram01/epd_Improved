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
//
// Lightweight camera ingress isolation for Easy Perception Deployment.

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "epd_utils_lib/depth_registration.hpp"
#include "epd_utils_lib/log_format.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

class SensorIngress : public rclcpp::Node
{
public:
  SensorIngress()
  : Node("epd_sensor_ingress")
  {
    const auto rgb_in = declare_parameter<std::string>(
      "rgb_input_topic", "/camera/camera/color/image_raw");
    const auto depth_in = declare_parameter<std::string>(
      "depth_input_topic", "/camera/camera/aligned_depth_to_color/image_raw");
    const auto info_in = declare_parameter<std::string>(
      "camera_info_input_topic", "/camera/camera/color/camera_info");
    const auto rgb_out = declare_parameter<std::string>(
      "rgb_output_topic", "/easy_perception_deployment/ingress/color/image_raw");
    const auto depth_out = declare_parameter<std::string>(
      "depth_output_topic", "/easy_perception_deployment/ingress/aligned_depth/image_raw");
    const auto info_out = declare_parameter<std::string>(
      "camera_info_output_topic", "/easy_perception_deployment/ingress/color/camera_info");
    // "camera": depth_input_topic is already aligned to colour (aligned_depth_to_color) and is
    // forwarded untouched. "register": depth_input_topic is in the depth optical frame
    // (depth/image_rect_raw); the ingress reprojects it into the colour optical frame using
    // both CameraInfos and the camera's own depth->colour TF, and pairs it with the colour frame.
    depth_alignment_ = declare_parameter<std::string>("depth_alignment", "camera");
    if (depth_alignment_ != "camera" && depth_alignment_ != "register") {
      throw std::invalid_argument("depth_alignment must be 'camera' or 'register'");
    }
    const auto depth_info_in = declare_parameter<std::string>(
      "depth_camera_info_input_topic", "/camera/camera/depth/camera_info");
    const double max_skew_s = declare_parameter<double>("max_rgbd_pair_skew_s", 0.020);
    if (!(max_skew_s > 0.0 && max_skew_s < 0.1)) {
      throw std::invalid_argument("max_rgbd_pair_skew_s must be in (0, 0.1)");
    }
    pairer_ = std::make_unique<EPD::RgbdPairer>(static_cast<int64_t>(max_skew_s * 1e9));

    // Camera-facing readers match the proven RealSense RELIABLE KEEP_LAST(1)
    // publisher. Internal delivery is latest-only (KEEP_LAST(1): a slow reader
    // never backpressures the ingress). It is RELIABLE by default: best-effort
    // delivery of ~1 MB images loses UDP fragments, which on LENLEG (WSL2,
    // FastDDS, localhost) delivered ~1.5 of 14 camera frames per second.
    const auto internal_reliability = declare_parameter<std::string>(
      "internal_qos_reliability", "reliable");
    if (internal_reliability != "reliable" && internal_reliability != "best_effort") {
      throw std::invalid_argument("internal_qos_reliability must be 'reliable' or 'best_effort'");
    }
    auto camera_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    auto internal_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile();
    if (internal_reliability == "reliable") {
      internal_qos.reliable();
    } else {
      internal_qos.best_effort();
    }

    rgb_pub_ = create_publisher<sensor_msgs::msg::Image>(rgb_out, internal_qos);
    depth_pub_ = create_publisher<sensor_msgs::msg::Image>(depth_out, internal_qos);
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(info_out, internal_qos);
    health_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "/easy_perception_deployment/sensor_ingress/health", 1);

    if (depth_alignment_ == "register") {
      tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
      depth_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        depth_info_in, camera_qos,
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
          ++depth_info_count_;
          latest_depth_info_ = message;
        });
    }

    rgb_sub_ = create_subscription<sensor_msgs::msg::Image>(
      rgb_in, camera_qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
        ++rgb_count_;
        last_rgb_stamp_ = rclcpp::Time(message->header.stamp);
        last_rgb_receive_ = now();
        if (pairing()) {
          dispatch(pairer_->addColor(message));
        } else {
          rgb_pub_->publish(*message);
        }
      });
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      depth_in, camera_qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
        ++depth_count_;
        last_depth_stamp_ = rclcpp::Time(message->header.stamp);
        last_depth_receive_ = now();
        if (pairing()) {
          dispatch(pairer_->addDepth(message));
        } else {
          depth_pub_->publish(*message);
        }
      });
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      info_in, camera_qos,
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
        ++info_count_;
        last_info_stamp_ = rclcpp::Time(message->header.stamp);
        last_info_receive_ = now();
        if (pairing()) {
          dispatch(pairer_->addColorInfo(message));
        } else {
          info_pub_->publish(*message);
        }
      });

    health_timer_ = create_wall_timer(
      std::chrono::seconds(5), std::bind(&SensorIngress::publish_health, this));

    RCLCPP_INFO(
      get_logger(),
      "Isolated sensor ingress active: RELIABLE KEEP_LAST(1) camera readers, "
      "%s KEEP_LAST(1) internal publishers; depth_alignment=%s", internal_reliability.c_str(),
      depth_alignment_.c_str());
  }

private:
  bool pairing() const {return depth_alignment_ == "register";}

  // Register the paired depth into the colour optical frame and publish the triplet with the
  // colour sensor stamp (the same convention as the camera's aligned_depth_to_color). Nothing
  // is published when the extrinsics or intrinsics are unavailable or invalid.
  void dispatch(const std::optional<EPD::RgbdPairer::Triplet> & triplet)
  {
    if (!triplet) {return;}
    const auto & color = *triplet->color;
    const auto & depth = *triplet->depth;
    if (!latest_depth_info_) {
      return fail("waiting_for_depth_camera_info");
    }
    if (color.header.frame_id.empty() ||
      color.header.frame_id != triplet->color_info->header.frame_id)
    {
      return fail("color_camera_info_frame_mismatch");
    }
    if (color.width != triplet->color_info->width || color.height != triplet->color_info->height) {
      return fail("color_camera_info_size_mismatch");
    }
    EPD::RigidTransform color_from_depth;
    try {
      // RealSense publishes depth->colour extrinsics as a static transform.
      const auto tf = tf_buffer_->lookupTransform(
        color.header.frame_id, depth.header.frame_id, tf2::TimePointZero);
      const auto & q = tf.transform.rotation;
      const tf2::Matrix3x3 m(tf2::Quaternion(q.x, q.y, q.z, q.w));
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {color_from_depth.rotation[r * 3 + c] = m[r][c];}
      }
      color_from_depth.translation = {
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z};
    } catch (const tf2::TransformException & exc) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "Depth->colour extrinsics %s<-%s unavailable: %s",
        color.header.frame_id.c_str(), depth.header.frame_id.c_str(), exc.what());
      return fail("depth_to_color_extrinsics_unavailable");
    }
    auto registered = std::make_unique<sensor_msgs::msg::Image>();
    const auto error = EPD::registerDepthToColor(
      depth, *latest_depth_info_, *triplet->color_info, color_from_depth, *registered);
    if (!error.empty()) {
      return fail(error);
    }
    registered->header.stamp = color.header.stamp;
    ++registered_count_;
    last_registration_error_.clear();
    last_pair_skew_ns_ = triplet->skew_ns;
    max_pair_skew_ns_ = std::max<uint64_t>(max_pair_skew_ns_, std::llabs(triplet->skew_ns));
    rgb_pub_->publish(color);
    info_pub_->publish(*triplet->color_info);
    depth_pub_->publish(std::move(registered));
  }

  void fail(const std::string & reason)
  {
    ++registration_failures_;
    if (reason != last_registration_error_) {
      RCLCPP_ERROR(get_logger(), "Depth registration refused a frame: %s", reason.c_str());
    }
    last_registration_error_ = reason;
  }

  static diagnostic_msgs::msg::KeyValue value(const std::string & key, const std::string & text)
  {
    diagnostic_msgs::msg::KeyValue item;
    item.key = key;
    item.value = text;
    return item;
  }

  static diagnostic_msgs::msg::KeyValue value(const std::string & key, int64_t number)
  {
    diagnostic_msgs::msg::KeyValue item;
    item.key = key;
    item.value = std::to_string(number);
    return item;
  }

  static diagnostic_msgs::msg::KeyValue value(const std::string & key, uint64_t number)
  {
    diagnostic_msgs::msg::KeyValue item;
    item.key = key;
    item.value = std::to_string(number);
    return item;
  }

  static diagnostic_msgs::msg::KeyValue value(const std::string & key, double number)
  {
    diagnostic_msgs::msg::KeyValue item;
    item.key = key;
    item.value = std::to_string(number);
    return item;
  }

  void publish_health()
  {
    const auto current = now();
    const auto age = [&current](uint64_t count, const rclcpp::Time & received) {
        return count == 0 ? -1.0 : (current - received).seconds();
      };
    const double rgb_age = age(rgb_count_, last_rgb_receive_);
    const double depth_age = age(depth_count_, last_depth_receive_);
    const double info_age = age(info_count_, last_info_receive_);
    bool healthy = rgb_age >= 0.0 && rgb_age < 2.0 &&
      depth_age >= 0.0 && depth_age < 2.0 && info_age >= 0.0 && info_age < 2.0;
    // In register mode the output, not just the inputs, must advance.
    const bool registering = pairing() && registered_count_ > last_health_registered_;
    if (pairing()) {healthy = healthy && registering;}

    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "EPD sensor ingress";
    status.hardware_id = "camera_ingress";
    status.level = healthy ? diagnostic_msgs::msg::DiagnosticStatus::OK :
      diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    status.message = healthy ? "camera streams advancing" :
      (pairing() && !last_registration_error_.empty() ?
      "depth registration refused frames: " + last_registration_error_ :
      "one or more camera streams stale");
    status.values = {
      value("rgb_count", rgb_count_), value("depth_count", depth_count_),
      value("camera_info_count", info_count_), value("rgb_age_s", rgb_age),
      value("depth_age_s", depth_age), value("camera_info_age_s", info_age),
      value("rgb_sensor_stamp_ns", static_cast<uint64_t>(last_rgb_stamp_.nanoseconds())),
      value("depth_sensor_stamp_ns", static_cast<uint64_t>(last_depth_stamp_.nanoseconds())),
      value("camera_info_sensor_stamp_ns", static_cast<uint64_t>(last_info_stamp_.nanoseconds())),
      value("depth_alignment", depth_alignment_)};
    if (pairing()) {
      status.values.push_back(value("depth_camera_info_count", depth_info_count_));
      status.values.push_back(value("registered_depth_count", registered_count_));
      status.values.push_back(value("superseded_color_frames", pairer_->superseded()));
      status.values.push_back(value("unpaired_color_frames", pairer_->unpaired()));
      status.values.push_back(value("registration_failures", registration_failures_));
      status.values.push_back(value("last_rgbd_pair_skew_ns", last_pair_skew_ns_));
      status.values.push_back(value("max_rgbd_pair_skew_ns", max_pair_skew_ns_));
      status.values.push_back(value("last_registration_error", last_registration_error_));
      last_health_registered_ = registered_count_;
    }

    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = current;
    array.status.push_back(std::move(status));
    health_pub_->publish(array);

    RCLCPP_DEBUG(
      get_logger(), "Ingress counts: RGB=%llu depth=%llu info=%llu",
      static_cast<EPD::LogULongLong>(rgb_count_),
      static_cast<EPD::LogULongLong>(depth_count_),
      static_cast<EPD::LogULongLong>(info_count_));
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr health_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_;
  rclcpp::TimerBase::SharedPtr health_timer_;
  std::string depth_alignment_;
  std::unique_ptr<EPD::RgbdPairer> pairer_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr latest_depth_info_;
  uint64_t depth_info_count_{0};
  uint64_t registered_count_{0};
  uint64_t last_health_registered_{0};
  uint64_t registration_failures_{0};
  int64_t last_pair_skew_ns_{0};
  uint64_t max_pair_skew_ns_{0};
  std::string last_registration_error_;
  uint64_t rgb_count_{0};
  uint64_t depth_count_{0};
  uint64_t info_count_{0};
  rclcpp::Time last_rgb_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_depth_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_info_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_rgb_receive_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_depth_receive_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_info_receive_{0, 0, RCL_ROS_TIME};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorIngress>());
  rclcpp::shutdown();
  return 0;
}
