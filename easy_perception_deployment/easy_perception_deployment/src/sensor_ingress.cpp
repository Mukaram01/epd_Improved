// Copyright 2026
// Lightweight camera ingress isolation for Easy Perception Deployment.

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

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

    // Camera-facing readers match the proven RealSense RELIABLE KEEP_LAST(1)
    // publisher. Internal delivery is latest-only and must never backpressure
    // the lightweight ingress process.
    auto camera_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    auto internal_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();

    rgb_pub_ = create_publisher<sensor_msgs::msg::Image>(rgb_out, internal_qos);
    depth_pub_ = create_publisher<sensor_msgs::msg::Image>(depth_out, internal_qos);
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(info_out, internal_qos);
    health_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "/easy_perception_deployment/sensor_ingress/health", 1);

    rgb_sub_ = create_subscription<sensor_msgs::msg::Image>(
      rgb_in, camera_qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
        ++rgb_count_;
        last_rgb_stamp_ = rclcpp::Time(message->header.stamp);
        last_rgb_receive_ = now();
        rgb_pub_->publish(*message);
      });
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      depth_in, camera_qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
        ++depth_count_;
        last_depth_stamp_ = rclcpp::Time(message->header.stamp);
        last_depth_receive_ = now();
        depth_pub_->publish(*message);
      });
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      info_in, camera_qos,
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
        ++info_count_;
        last_info_stamp_ = rclcpp::Time(message->header.stamp);
        last_info_receive_ = now();
        info_pub_->publish(*message);
      });

    health_timer_ = create_wall_timer(
      std::chrono::seconds(5), std::bind(&SensorIngress::publish_health, this));

    RCLCPP_INFO(
      get_logger(),
      "Isolated sensor ingress active: RELIABLE KEEP_LAST(1) camera readers, "
      "BEST_EFFORT KEEP_LAST(1) internal publishers");
  }

private:
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
    const bool healthy = rgb_age >= 0.0 && rgb_age < 2.0 &&
      depth_age >= 0.0 && depth_age < 2.0 && info_age >= 0.0 && info_age < 2.0;

    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "EPD sensor ingress";
    status.hardware_id = "camera_ingress";
    status.level = healthy ? diagnostic_msgs::msg::DiagnosticStatus::OK :
      diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    status.message = healthy ? "camera streams advancing" : "one or more camera streams stale";
    status.values = {
      value("rgb_count", rgb_count_), value("depth_count", depth_count_),
      value("camera_info_count", info_count_), value("rgb_age_s", rgb_age),
      value("depth_age_s", depth_age), value("camera_info_age_s", info_age),
      value("rgb_sensor_stamp_ns", static_cast<uint64_t>(last_rgb_stamp_.nanoseconds())),
      value("depth_sensor_stamp_ns", static_cast<uint64_t>(last_depth_stamp_.nanoseconds())),
      value("camera_info_sensor_stamp_ns", static_cast<uint64_t>(last_info_stamp_.nanoseconds()))};

    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = current;
    array.status.push_back(std::move(status));
    health_pub_->publish(array);

    RCLCPP_DEBUG(
      get_logger(), "Ingress counts: RGB=%llu depth=%llu info=%llu",
      static_cast<unsigned long long>(rgb_count_),
      static_cast<unsigned long long>(depth_count_),
      static_cast<unsigned long long>(info_count_));
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr health_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::TimerBase::SharedPtr health_timer_;
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
