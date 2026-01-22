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

#ifndef EPD_UTILS_LIB__EASY_PERCEPTION_DEPLOYMENT_HPP_
#define EPD_UTILS_LIB__EASY_PERCEPTION_DEPLOYMENT_HPP_

#include <chrono>
#include <algorithm>
#include <cctype>
#include <string>
#include <memory>
#include <functional>
#include <stdexcept>  // FIX: for std::runtime_error
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <fstream>

// OpenCV LIB
#include "opencv2/opencv.hpp"

// ROS2 LIB
#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/region_of_interest.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/image_encodings.hpp"  // FIX: for sensor_msgs::image_encodings::TYPE_16UC1
#include "geometry_msgs/msg/point.hpp"
#include "image_transport/image_transport.hpp"
#include "image_transport/subscriber_filter.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

// EPD_UTILS LIB
#include "epd_utils_lib/epd_container.hpp"
#include "epd_msgs/msg/epd_image_classification.hpp"
#include "epd_msgs/msg/epd_object_detection.hpp"
#include "epd_msgs/msg/epd_object_localization.hpp"
#include "epd_msgs/msg/epd_object_tracking.hpp"
#include "epd_msgs/msg/epd_performance.hpp"
#include "epd_msgs/msg/localized_object.hpp"
#include "epd_msgs/srv/perception.hpp"
#include "epd_utils_lib/usecase_config.hpp"
#include "epd_utils_lib/message_utils.hpp"

#include "pcl_conversions/pcl_conversions.h"

/*! \class EasyPerceptionDeployment
    \brief An EasyPerceptionDeployment class object.
    This class object inherits rclcpp::Node object and acts the main bridge
    between the ROS2 interface and the underlying ort_cpp_lib library that is
    based on ONNXRuntime Library.
    The node now uses a background worker thread to run ONNX Runtime inference.
    ROS callbacks only enqueue the latest incoming frames, and the worker thread
    drains the most recent data for each mode (image, localization, or tracking)
    with mutex/condition_variable synchronization to avoid races with shared
    state such as frame buffers and ORT session initialization.
*/
class EasyPerceptionDeployment : public rclcpp::Node
{
public:
  /*! \brief A Constructor function*/
  EasyPerceptionDeployment(void);
  ~EasyPerceptionDeployment(void);
  /*! \brief A function that abstracts processing of input image in image_callback.*/
  void process_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  /*! \brief A function that abstracts processing of input image in localize_callback.*/
  void process_localize_callback(
    const sensor_msgs::msg::Image::SharedPtr msg,
    const sensor_msgs::msg::Image::SharedPtr depth_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr camera_info);
  /*! \brief A function that abstracts processing of input image in tracking_callback.*/
  void process_tracking_callback(
    const sensor_msgs::msg::Image::SharedPtr msg,
    const sensor_msgs::msg::Image::SharedPtr depth_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr camera_info);

private:
  /*! \brief A subscriber member variable to receive 2D RGB images to receive.*/
  image_transport::Subscriber image_sub;

  /*! \brief An alias definition for SyncPolicy that is used below for sync_ object.*/
  typedef message_filters::sync_policies::ApproximateTime
    <sensor_msgs::msg::Image, sensor_msgs::msg::Image,
      sensor_msgs::msg::CameraInfo> SyncPolicy;

  /*! \brief A policy-synchronized subscriber member variable
  to receive rectified 2D RGB images.
  */
  image_transport::SubscriberFilter localize_image_rgb;
  /*! \brief A policy-synchronized subscriber member variable
  to receive rectified 2D Depth images.
  */
  image_transport::SubscriberFilter localize_image_depth;
  /*! \brief A policy-synchronized subscriber member variable
  to receive camera information.
  */
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> localize_cam_info;
  /*! \brief A Synchronizer policy member variable.*/
  message_filters::Synchronizer<SyncPolicy> sync_;

  /*! \brief A publisher member variable to output visualization of inference
  results*/
  image_transport::Publisher visual_pub;
  /*! \brief A publisher member variable to output Precision-Level 1 (P1)
  specific inference output suitable for external agents.*/
  rclcpp::Publisher<epd_msgs::msg::EPDImageClassification>::SharedPtr p1_pub;
  /*! \brief A publisher member variable to output Precision-Level 2 (P2)
  specific inference output suitable for external agents.*/
  rclcpp::Publisher<epd_msgs::msg::EPDObjectDetection>::SharedPtr p2_pub;
  /*! \brief A publisher member variable to output Precision-Level 3 (P3)
  specific inference output suitable for external agents.*/
  rclcpp::Publisher<epd_msgs::msg::EPDObjectDetection>::SharedPtr p3_pub;
  /*! \brief A publisher member variable to output Precision-Level 3 (P3)
  specific inference output suitable for external agents.*/
  rclcpp::Publisher<epd_msgs::msg::EPDObjectLocalization>::SharedPtr localize_pub;

  rclcpp::Publisher<epd_msgs::msg::EPDObjectTracking>::SharedPtr tracking_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_pub;
  rclcpp::Publisher<epd_msgs::msg::EPDPerformance>::SharedPtr performance_pub_;

  rclcpp::Service<epd_msgs::srv::Perception>::SharedPtr srv_;
  /*! \brief A singular EPDContainer object that deploys a user-defined
  ONNX model as an inference enginer using onnxruntime.
  */
  mutable EPD::EPDContainer ortAgent_;

  void localize_callback(
    const sensor_msgs::msg::Image::SharedPtr msg,
    const sensor_msgs::msg::Image::SharedPtr depth_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr camera_info);

  void tracking_callback(
    const sensor_msgs::msg::Image::SharedPtr msg,
    const sensor_msgs::msg::Image::SharedPtr depth_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr camera_info);

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void image_worker_loop();
  void publishPerformanceMetrics(
    const std_msgs::msg::Header & header,
    const std::string & pipeline,
    int64_t preprocess_ms,
    int64_t inference_ms,
    int64_t postprocess_ms,
    int64_t publish_ms,
    int64_t total_ms);
  void handlePickStatus(const std_msgs::msg::String::SharedPtr msg);

  void hasCameraChanged(
    const int img_height,
    const int img_width) const;

  void checkOrtAgentIsInitialized(
    const int img_height,
    const int img_width) const;

  void subscribeImageInput();
  void subscribeLocalizeInputs();
  std::string resolveDepthTransport(const std::string & transport) const;

  std::string rgb_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string image_transport_;
  std::string depth_transport_;
  rmw_qos_profile_t sensor_qos_profile_;
  bool image_input_active_{false};
  void process_image_work(const sensor_msgs::msg::Image::SharedPtr msg);
  void process_localize_work(
    const sensor_msgs::msg::Image::SharedPtr msg,
    const sensor_msgs::msg::Image::SharedPtr depth_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr camera_info);
  void process_tracking_work(
    const sensor_msgs::msg::Image::SharedPtr msg,
    const sensor_msgs::msg::Image::SharedPtr depth_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr camera_info);
  void worker_loop();

  std::mutex data_mutex_;
  std::condition_variable data_cv_;
  std::thread worker_thread_;
  bool worker_stop_{false};
  bool image_pending_{false};
  bool localize_pending_{false};
  bool tracking_pending_{false};
  sensor_msgs::msg::Image::SharedPtr latest_image_;
  sensor_msgs::msg::Image::SharedPtr latest_depth_image_;
  sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info_;

  std::mutex ort_mutex_;
  std::mutex performance_log_mutex_;
  std::string performance_log_path_;
  std::string performance_log_format_;
  bool performance_log_header_written_{false};
  std::atomic<uint64_t> picks_attempted_{0};
  std::atomic<uint64_t> picks_failed_{0};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr pick_status_sub_;
};

EasyPerceptionDeployment::EasyPerceptionDeployment(void)
: Node("easy_perception_deployment"),
  sync_(SyncPolicy(10), localize_image_rgb, localize_image_depth, localize_cam_info),
  sensor_qos_profile_(rclcpp::SensorDataQoS().get_rmw_qos_profile())
{
  rclcpp::PublisherOptions publisher_options;
  publisher_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  const auto image_qos = rclcpp::SensorDataQoS().keep_last(1).best_effort();
  const auto camera_info_qos = rclcpp::SensorDataQoS().keep_last(1);

  // FIX: Humble requires declare_parameter<T>(name, default)
  this->declare_parameter<double>("camera_to_plane_distance_mm", 1000.0);
  this->declare_parameter<std::string>("rgb_topic", "/camera/color/image_raw");
  this->declare_parameter<std::string>("depth_topic", "/camera/depth/image_rect_raw");
  this->declare_parameter<std::string>("camera_info_topic", "/camera/color/camera_info");
  this->declare_parameter<std::string>("image_transport", ortAgent_.image_transport);

  rgb_topic_ = this->get_parameter("rgb_topic").as_string();
  depth_topic_ = this->get_parameter("depth_topic").as_string();
  camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();
  image_transport_ = this->get_parameter("image_transport").as_string();
  std::transform(
    image_transport_.begin(),
    image_transport_.end(),
    image_transport_.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  if (image_transport_.empty() ||
    (image_transport_ != "raw" && image_transport_ != "compressed" &&
    image_transport_ != "compresseddepth"))
  {
    RCLCPP_WARN(
      this->get_logger(),
      "Invalid image_transport '%s'. Falling back to 'raw'.",
      image_transport_.c_str());
    image_transport_ = "raw";
  }
  depth_transport_ = resolveDepthTransport(image_transport_);

  subscribeImageInput();
  // Creating Subscriber to get Input Image.
  image_sub = this->create_subscription<sensor_msgs::msg::Image>(
    "/easy_perception_deployment/image_input",
    image_qos,
    std::bind(&EasyPerceptionDeployment::image_callback, this, std::placeholders::_1),
    subscription_options);

  // Creating Publisher to output Visualizable P2 and P3 Detection Results.
  visual_pub = image_transport::create_publisher(
    this,
    "/easy_perception_deployment/image_output",
    rclcpp::QoS(10).get_rmw_qos_profile());

  // Creating Publisher to output Action P1 Detection Results.
  p1_pub = this->create_publisher<epd_msgs::msg::EPDImageClassification>(
    "/easy_perception_deployment/epd_p1_output",
    10,
    publisher_options);

  // Creating Publisher to output Action P2 Detection Results.
  p2_pub = this->create_publisher<epd_msgs::msg::EPDObjectDetection>(
    "/easy_perception_deployment/epd_p2_output",
    10,
    publisher_options);

  // Creating Publisher to output Action P3 Detection Results.
  p3_pub = this->create_publisher<epd_msgs::msg::EPDObjectDetection>(
    "/easy_perception_deployment/epd_p3_output",
    10,
    publisher_options);

  // Creating Publisher to output Action P3 and Localization Detection Results.
  localize_pub = this->create_publisher<epd_msgs::msg::EPDObjectLocalization>(
    "/easy_perception_deployment/epd_localize_output",
    10,
    publisher_options);

  // Creating Publisher to output Action P3 and Tracking Detection Results.
  tracking_pub = this->create_publisher<epd_msgs::msg::EPDObjectTracking>(
    "/easy_perception_deployment/epd_tracking_output",
    10,
    publisher_options);

  // Creating Publisher to output 3D poses of localized/tracked objects.
  pose_pub = this->create_publisher<geometry_msgs::msg::PoseArray>(
    "/easy_perception_deployment/epd_pose_output",
    10,
    publisher_options);

  performance_pub_ = this->create_publisher<epd_msgs::msg::EPDPerformance>(
    "/easy_perception_deployment/epd_performance",
    10,
    publisher_options);

  performance_log_path_ = ortAgent_.performance_log_path;
  performance_log_format_ = ortAgent_.performance_log_format;

  if (!ortAgent_.pick_status_topic.empty()) {
    pick_status_sub_ = this->create_subscription<std_msgs::msg::String>(
      ortAgent_.pick_status_topic,
      10,
      std::bind(&EasyPerceptionDeployment::handlePickStatus, this, std::placeholders::_1));
  }

  // If useCaseMode is detected to be Localization or Tracking,
  // Subscribe to all synchronized ROS2 topics.
  if (ortAgent_.useCaseMode == 3) {
    subscribeLocalizeInputs();
    sync_.registerCallback(&EasyPerceptionDeployment::localize_callback, this);
    if (image_input_active_) {
      image_sub.shutdown();
      image_input_active_ = false;
    }
  } else if (ortAgent_.useCaseMode == 4) {
    subscribeLocalizeInputs();
    sync_.registerCallback(&EasyPerceptionDeployment::tracking_callback, this);
    if (image_input_active_) {
      image_sub.shutdown();
      image_input_active_ = false;
    }
  } else {
    localize_image_rgb.unsubscribe();
    localize_image_depth.unsubscribe();
    localize_cam_info.unsubscribe();
  }

  // FIX: Humble requires declare_parameter<T>(name, default)
  this->declare_parameter<double>("camera_to_plane_distance_mm", 1000.0);
  this->declare_parameter<std::string>("rgb_topic", "/camera/color/image_raw");
  this->declare_parameter<std::string>("depth_topic", "/camera/depth/image_rect_raw");
  this->declare_parameter<std::string>("camera_info_topic", "/camera/color/camera_info");

  const std::string rgb_topic = this->get_parameter("rgb_topic").as_string();
  const std::string depth_topic = this->get_parameter("depth_topic").as_string();
  const std::string camera_info_topic = this->get_parameter("camera_info_topic").as_string();

  localize_image_rgb.subscribe(
    this,
    rgb_topic,
    image_qos.get_rmw_qos_profile(),  // Required for ROS 2 Humble message_filters::Subscriber API.
    subscription_options);
  localize_image_depth.subscribe(
    this,
    depth_topic,
    image_qos.get_rmw_qos_profile(),  // Required for ROS 2 Humble message_filters::Subscriber API.
    subscription_options);
  localize_cam_info.subscribe(
    this,
    camera_info_topic,
    camera_info_qos.get_rmw_qos_profile(),  // Required for ROS 2 Humble message_filters::Subscriber API.
    subscription_options);

  auto handle_emd_request =
    [this](
    const std::shared_ptr<epd_msgs::srv::Perception::Request> request,
    std::shared_ptr<epd_msgs::srv::Perception::Response> response) -> void
    {
      (void)request;
      RCLCPP_INFO(this->get_logger(), "[ RECEIVED ] - EMD Grasp-Planner Request");
      response->success = true;

      {
        std::lock_guard<std::mutex> ort_guard(ort_mutex_);
        response->tracking_enabled = (ortAgent_.useCaseMode == 4);
      }

      int use_case_mode = 0;
      {
        std::lock_guard<std::mutex> ort_guard(ort_mutex_);
        use_case_mode = ortAgent_.useCaseMode;
      }

      if (ortAgent_.useCaseMode == 3) {
        subscribeLocalizeInputs();
        sync_.registerCallback(&EasyPerceptionDeployment::localize_callback, this);
        if (image_input_active_) {
          image_sub.shutdown();
          image_input_active_ = false;
        }
      } else if (ortAgent_.useCaseMode == 4) {
        subscribeLocalizeInputs();
      if (use_case_mode == 3) {
        localize_image_rgb.subscribe();
        localize_image_depth.subscribe();
        localize_cam_info.subscribe();
        sync_.registerCallback(&EasyPerceptionDeployment::localize_callback, this);
      } else if (use_case_mode == 4) {
        localize_image_rgb.subscribe();
        localize_image_depth.subscribe();
        localize_cam_info.subscribe();
        sync_.registerCallback(&EasyPerceptionDeployment::tracking_callback, this);
        if (image_input_active_) {
          image_sub.shutdown();
          image_input_active_ = false;
        }
      } else {
        if (!image_input_active_) {
          subscribeImageInput();
        if (!image_sub) {
          image_sub = this->create_subscription<sensor_msgs::msg::Image>(
            "/easy_perception_deployment/image_input",
            rclcpp::SensorDataQoS().keep_last(1).best_effort(),
            std::bind(&EasyPerceptionDeployment::image_callback, this, std::placeholders::_1),
            subscription_options);
        }
      }

      {
        std::lock_guard<std::mutex> ort_guard(ort_mutex_);
        ortAgent_.requestAddressed = false;
      }
    };

  srv_ = this->create_service<epd_msgs::srv::Perception>(
    "epd_perception_service",
    handle_emd_request);

  // Log all session_config and usecase_config configurations for user to check on system boot
  RCLCPP_INFO(this->get_logger(), "[-ONNX Model-] - %s", ortAgent_.onnx_model_path.c_str());
  RCLCPP_INFO(this->get_logger(), "[-Label List-] - %s", ortAgent_.class_label_path.c_str());
  RCLCPP_INFO(this->get_logger(), "[-Precision Level-] - %d", ortAgent_.precision_level);
  RCLCPP_INFO(this->get_logger(), "[-Image Transport-] - %s", image_transport_.c_str());
  if (!performance_log_path_.empty()) {
    RCLCPP_INFO(
      this->get_logger(),
      "[-Performance Log-] - %s (%s)",
      performance_log_path_.c_str(),
      performance_log_format_.c_str());
  }

  if (ortAgent_.isVisualize()) {
    RCLCPP_INFO(this->get_logger(), "[-Mode-] - VISUALISE");
  } else {
    RCLCPP_INFO(this->get_logger(), "[-Mode-] - ACTION");
  }

  switch (ortAgent_.useCaseMode) {
    case EPD::CLASSIFICATION_MODE:
      RCLCPP_INFO(this->get_logger(), "[-Use Case-] - EPD::CLASSIFICATION_MODE");
      break;
    case EPD::COUNTING_MODE:
      RCLCPP_INFO(this->get_logger(), "[-Use Case-] - EPD::COUNTING_MODE");
      break;
    case EPD::COLOR_MATCHING_MODE:
      RCLCPP_INFO(this->get_logger(), "[-Use Case-] - EPD::COLOR_MATCHING_MODE");
      break;
    case EPD::LOCALISATION_MODE:
      RCLCPP_INFO(this->get_logger(), "[-Use Case-] - EPD::LOCALISATION_MODE");
      RCLCPP_INFO(this->get_logger(), "[- Input RGB Image Topic -] - %s", rgb_topic_.c_str());
      RCLCPP_INFO(
        this->get_logger(),
        "[- Input Depth Image Topic -] - %s",
        depth_topic_.c_str());
      RCLCPP_INFO(
        this->get_logger(),
        "[- Camera Info Topic -] - %s",
        camera_info_topic_.c_str());
      break;
    case EPD::TRACKING_MODE:
      RCLCPP_INFO(this->get_logger(), "[-Use Case-] - EPD::TRACKING_MODE");
      break;
  }

  worker_thread_ = std::thread(&EasyPerceptionDeployment::worker_loop, this);
}

EasyPerceptionDeployment::~EasyPerceptionDeployment(void)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    worker_stop_ = true;
  }
  data_cv_.notify_all();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

std::string EasyPerceptionDeployment::resolveDepthTransport(const std::string & transport) const
{
  if (transport == "compressed") {
    return "compressedDepth";
  }
  if (transport == "compresseddepth") {
    return "compressedDepth";
  }
  return transport;
}

void EasyPerceptionDeployment::subscribeImageInput()
{
  image_sub = image_transport::create_subscription(
    this,
    "/easy_perception_deployment/image_input",
    std::bind(&EasyPerceptionDeployment::image_callback, this, std::placeholders::_1),
    image_transport_,
    sensor_qos_profile_);
  image_input_active_ = true;
}

void EasyPerceptionDeployment::subscribeLocalizeInputs()
{
  rclcpp::SubscriptionOptions subscription_options;
  subscription_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  localize_image_rgb.subscribe(
    this,
    rgb_topic_,
    image_transport_,
    sensor_qos_profile_);
  localize_image_depth.subscribe(
    this,
    depth_topic_,
    depth_transport_,
    sensor_qos_profile_);
  localize_cam_info.subscribe(
    this,
    camera_info_topic_,
    sensor_qos_profile_,
    subscription_options);
}

void EasyPerceptionDeployment::hasCameraChanged(const int img_height, const int img_width) const
{
  // FIX: should trigger if EITHER dimension changed (not only when both changed)
  if (ortAgent_.getWidth() != img_width || ortAgent_.getHeight() != img_height) {
    throw std::runtime_error("Input camera changed. Please restart.");
  }
}

void EasyPerceptionDeployment::checkOrtAgentIsInitialized(
  const int img_height,
  const int img_width) const
{
  if (!ortAgent_.isInit()) {
    ortAgent_.setFrameDimension(img_width, img_height);
    ortAgent_.initORTSessionHandler();
    ortAgent_.setInitBoolean(true);
  } else {
    hasCameraChanged(img_height, img_width);
  }
}

void EasyPerceptionDeployment::handlePickStatus(const std_msgs::msg::String::SharedPtr msg)
{
  std::string status = msg->data;
  std::transform(status.begin(), status.end(), status.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  const bool is_attempt =
    status.find("attempt") != std::string::npos ||
    status.find("success") != std::string::npos ||
    status.find("fail") != std::string::npos;
  if (is_attempt) {
    ++picks_attempted_;
  }
  if (status.find("fail") != std::string::npos) {
    ++picks_failed_;
  }
}

void EasyPerceptionDeployment::publishPerformanceMetrics(
  const std_msgs::msg::Header & header,
  const std::string & pipeline,
  int64_t preprocess_ms,
  int64_t inference_ms,
  int64_t postprocess_ms,
  int64_t publish_ms,
  int64_t total_ms)
{
  epd_msgs::msg::EPDPerformance metrics;
  metrics.header = header;
  metrics.pipeline = pipeline;
  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    metrics.precision_level = ortAgent_.precision_level;
  }
  metrics.preprocess_ms = preprocess_ms;
  metrics.inference_ms = inference_ms;
  metrics.postprocess_ms = postprocess_ms;
  metrics.publish_ms = publish_ms;
  metrics.total_ms = total_ms;
  metrics.picks_attempted = picks_attempted_.load();
  metrics.picks_failed = picks_failed_.load();

  performance_pub_->publish(metrics);

  if (performance_log_path_.empty()) {
    return;
  }

  std::lock_guard<std::mutex> log_guard(performance_log_mutex_);
  std::ofstream log_stream(performance_log_path_, std::ios::app);
  if (!log_stream.is_open()) {
    RCLCPP_WARN(
      this->get_logger(),
      "Failed to open performance log file: %s",
      performance_log_path_.c_str());
    return;
  }

  if (performance_log_format_ == "csv") {
    if (!performance_log_header_written_) {
      std::ifstream check_stream(performance_log_path_);
      if (check_stream.good()) {
        check_stream.peek();
        if (check_stream.good() && !check_stream.eof()) {
          performance_log_header_written_ = true;
        }
      }
      if (!performance_log_header_written_) {
        log_stream
          << "stamp_sec,stamp_nanosec,pipeline,precision_level,"
          << "preprocess_ms,inference_ms,postprocess_ms,publish_ms,total_ms,"
          << "picks_attempted,picks_failed\n";
        performance_log_header_written_ = true;
      }
    }
    log_stream
      << header.stamp.sec << ","
      << header.stamp.nanosec << ","
      << pipeline << ","
      << metrics.precision_level << ","
      << preprocess_ms << ","
      << inference_ms << ","
      << postprocess_ms << ","
      << publish_ms << ","
      << total_ms << ","
      << metrics.picks_attempted << ","
      << metrics.picks_failed << "\n";
    return;
  }

  std::string pipeline_escaped = pipeline;
  size_t pos = 0;
  while ((pos = pipeline_escaped.find('\"', pos)) != std::string::npos) {
    pipeline_escaped.insert(pos, "\\");
    pos += 2;
  }

  log_stream << "{"
             << "\"stamp_sec\":" << header.stamp.sec << ","
             << "\"stamp_nanosec\":" << header.stamp.nanosec << ","
             << "\"pipeline\":\"" << pipeline_escaped << "\","
             << "\"precision_level\":" << metrics.precision_level << ","
             << "\"preprocess_ms\":" << preprocess_ms << ","
             << "\"inference_ms\":" << inference_ms << ","
             << "\"postprocess_ms\":" << postprocess_ms << ","
             << "\"publish_ms\":" << publish_ms << ","
             << "\"total_ms\":" << total_ms << ","
             << "\"picks_attempted\":" << metrics.picks_attempted << ","
             << "\"picks_failed\":" << metrics.picks_failed
             << "}\n";
}

void EasyPerceptionDeployment::process_localize_callback(
  const sensor_msgs::msg::Image::SharedPtr msg,
  const sensor_msgs::msg::Image::SharedPtr depth_msg,
  const sensor_msgs::msg::CameraInfo::SharedPtr camera_info)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_image_ = msg;
    latest_depth_image_ = depth_msg;
    latest_camera_info_ = camera_info;
    localize_pending_ = true;
  }
  data_cv_.notify_one();
}

void EasyPerceptionDeployment::localize_callback(
  const sensor_msgs::msg::Image::SharedPtr msg,
  const sensor_msgs::msg::Image::SharedPtr depth_msg,
  const sensor_msgs::msg::CameraInfo::SharedPtr camera_info)
{
  this->process_localize_callback(msg, depth_msg, camera_info);
}

void EasyPerceptionDeployment::process_tracking_callback(
  const sensor_msgs::msg::Image::SharedPtr msg,
  const sensor_msgs::msg::Image::SharedPtr depth_msg,
  const sensor_msgs::msg::CameraInfo::SharedPtr camera_info)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_image_ = msg;
    latest_depth_image_ = depth_msg;
    latest_camera_info_ = camera_info;
    tracking_pending_ = true;
  }
  data_cv_.notify_one();
}

void EasyPerceptionDeployment::tracking_callback(
  const sensor_msgs::msg::Image::SharedPtr msg,
  const sensor_msgs::msg::Image::SharedPtr depth_msg,
  const sensor_msgs::msg::CameraInfo::SharedPtr camera_info)
{
  this->process_tracking_callback(msg, depth_msg, camera_info);
}

void EasyPerceptionDeployment::process_image_callback(
  const sensor_msgs::msg::Image::SharedPtr msg)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_image_ = msg;
    image_pending_ = true;
  }
  data_cv_.notify_one();
}

void EasyPerceptionDeployment::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  this->process_image_callback(msg);
}

void EasyPerceptionDeployment::process_localize_work(
  const sensor_msgs::msg::Image::SharedPtr msg,
  const sensor_msgs::msg::Image::SharedPtr depth_msg,
  const sensor_msgs::msg::CameraInfo::SharedPtr camera_info)
{
  const auto t_start = std::chrono::high_resolution_clock::now();
  std::unique_lock<std::mutex> ort_lock(ort_mutex_);
  if (ortAgent_.requestAddressed) {
    return;
  }
  ort_lock.unlock();

  const double camera_to_plane_distance_mm =
    this->get_parameter("camera_to_plane_distance_mm").as_double();

  if (msg->height == 0) {
    RCLCPP_WARN(this->get_logger(), "Input image empty. Discarding.");
    return;
  }

  cv_bridge::CvImageConstPtr imgptr;
  if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
    imgptr = cv_bridge::toCvShare(msg);
  } else {
    imgptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  cv::Mat img = imgptr->image;

  cv_bridge::CvImageConstPtr depth_imageptr;
  if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    depth_imageptr = cv_bridge::toCvShare(depth_msg);
  } else {
    depth_imageptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  cv::Mat depth_img = depth_imageptr->image;

  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    checkOrtAgentIsInitialized(img.rows, img.cols);
  }

  const auto t_preprocess_end = std::chrono::high_resolution_clock::now();

  EPD::EPDObjectLocalization result;
  const auto t_infer_start = std::chrono::high_resolution_clock::now();
  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    result = ortAgent_.p3_ort_session->infer(
      img,
      depth_img,
      *camera_info,
      camera_to_plane_distance_mm);
  }
  const auto t_infer_end = std::chrono::high_resolution_clock::now();

  const auto t_postprocess_start = t_infer_end;
  cv::Mat resultImg;
  sensor_msgs::msg::Image::SharedPtr visual_output_msg;

  bool visualize = false;
  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    visualize = ortAgent_.isVisualize();
  }

  if (visualize) {
    EPD::EPDObjectTracking converted_result(result.data_size);
    converted_result.object_ids.clear();
    for (size_t i = 0; i < result.data_size; i++) {
      converted_result.objects.emplace_back(result.objects[i]);
    }

    // FIX: don't redeclare resultImg (avoid shadowing)
    {
      std::lock_guard<std::mutex> ort_guard(ort_mutex_);
      resultImg = ortAgent_.visualize(converted_result, img);
    }

    visual_output_msg =
      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", resultImg).toImageMsg();
  }

  epd_msgs::msg::EPDObjectLocalization output_msg;

  output_msg.header = msg->header;
  output_msg.frame_width = img.cols;
  output_msg.frame_height = img.rows;
  output_msg.depth_image = *depth_msg;
  output_msg.depth_image.header = depth_msg->header;

  output_msg.ppx = camera_info->k.at(2);
  output_msg.fx  = camera_info->k.at(0);
  output_msg.ppy = camera_info->k.at(5);
  output_msg.fy  = camera_info->k.at(4);

  for (size_t i = 0; i < result.data_size; i++) {
    epd_msgs::msg::LocalizedObject object;
    object.name = result.objects[i].name;
    object.roi = result.objects[i].roi;

    sensor_msgs::msg::Image::SharedPtr mask_ptr = cv_bridge::CvImage(
      std_msgs::msg::Header(), "mono16", result.objects[i].mask).toImageMsg();
    mask_ptr->header.stamp = msg->header.stamp;
    mask_ptr->header.frame_id = msg->header.frame_id;
    object.segmented_binary_mask = *mask_ptr;

    object.centroid = result.objects[i].centroid;
    object.length   = result.objects[i].length;
    object.breadth  = result.objects[i].breadth;
    object.height   = result.objects[i].height;
    object.axis     = result.objects[i].axis;

    sensor_msgs::msg::PointCloud2 output_segmented_pcl;
    pcl::toROSMsg(result.objects[i].segmented_pcl, output_segmented_pcl);
    object.segmented_pcl = output_segmented_pcl;

    output_msg.objects.push_back(object);
  }

  geometry_msgs::msg::PoseArray pose_array;
  pose_array.header = msg->header;
  for (size_t i = 0; i < result.data_size; i++) {
    geometry_msgs::msg::Pose pose;
    pose.position = result.objects[i].centroid;
    pose.orientation.w = 1.0;
    pose_array.poses.push_back(pose);
  }

  const auto t_postprocess_end = std::chrono::high_resolution_clock::now();
  const auto process_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_postprocess_end - t_start).count();
  output_msg.process_time = process_time_ms;
  const auto t_publish_start = std::chrono::high_resolution_clock::now();
  if (visual_output_msg) {
    visual_pub.publish(*visual_output_msg);
    visual_pub->publish(*visual_output_msg);
  }
  localize_pub->publish(output_msg);
  pose_pub->publish(pose_array);
  const auto t_publish_end = std::chrono::high_resolution_clock::now();

  const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_publish_end - t_start);
  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "[-FPS-]= %f\n",
    1000.0 / elapsed_time.count());

  const auto preprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_preprocess_end - t_start).count();
  const auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_infer_end - t_infer_start).count();
  const auto postprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_postprocess_end - t_postprocess_start).count();
  const auto publish_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_publish_end - t_publish_start).count();
  publishPerformanceMetrics(
    msg->header,
    "localize",
    preprocess_ms,
    inference_ms,
    postprocess_ms,
    publish_ms,
    elapsed_time.count());

  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    if (ortAgent_.isService()) {
      ortAgent_.requestAddressed = true;
    }
  }
}

void EasyPerceptionDeployment::process_tracking_work(
  const sensor_msgs::msg::Image::SharedPtr msg,
  const sensor_msgs::msg::Image::SharedPtr depth_msg,
  const sensor_msgs::msg::CameraInfo::SharedPtr camera_info)
{
  const auto t_start = std::chrono::high_resolution_clock::now();
  std::unique_lock<std::mutex> ort_lock(ort_mutex_);
  if (ortAgent_.requestAddressed) {
    return;
  }
  ort_lock.unlock();

  const double camera_to_plane_distance_mm =
    this->get_parameter("camera_to_plane_distance_mm").as_double();

  if (msg->height == 0) {
    RCLCPP_WARN(this->get_logger(), "Input image empty. Discarding.");
    return;
  }

  cv_bridge::CvImageConstPtr imgptr;
  if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
    imgptr = cv_bridge::toCvShare(msg);
  } else {
    imgptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  cv::Mat img = imgptr->image;

  cv_bridge::CvImageConstPtr depth_imageptr;
  if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    depth_imageptr = cv_bridge::toCvShare(depth_msg);
  } else {
    depth_imageptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  cv::Mat depth_img = depth_imageptr->image;

  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    checkOrtAgentIsInitialized(img.rows, img.cols);
  }

  const auto t_preprocess_end = std::chrono::high_resolution_clock::now();

  EPD::EPDObjectTracking result;
  const auto t_infer_start = std::chrono::high_resolution_clock::now();
  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    result = ortAgent_.p3_ort_session->infer(
      img,
      depth_img,
      *camera_info,
      camera_to_plane_distance_mm,
      ortAgent_.tracker_type,
      ortAgent_.trackers,
      ortAgent_.tracker_logs,
      ortAgent_.tracker_results);
  }
  const auto t_infer_end = std::chrono::high_resolution_clock::now();

  const auto t_postprocess_start = t_infer_end;
  cv::Mat resultImg;
  sensor_msgs::msg::Image::SharedPtr visual_output_msg;

  bool visualize = false;
  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    visualize = ortAgent_.isVisualize();
  }

  if (visualize) {
    {
      std::lock_guard<std::mutex> ort_guard(ort_mutex_);
      resultImg = ortAgent_.visualize(result, img);
    }

    visual_output_msg =
      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", resultImg).toImageMsg();
  }

  epd_msgs::msg::EPDObjectTracking output_msg;

  output_msg.header = msg->header;
  output_msg.frame_width = img.cols;
  output_msg.frame_height = img.rows;
  output_msg.depth_image = *depth_msg;
  output_msg.depth_image.header = depth_msg->header;

  output_msg.ppx = camera_info->k.at(2);
  output_msg.fx  = camera_info->k.at(0);
  output_msg.ppy = camera_info->k.at(5);
  output_msg.fy  = camera_info->k.at(4);

  for (size_t i = 0; i < result.data_size; i++) {
    epd_msgs::msg::LocalizedObject object;
    object.name = result.objects[i].name;
    object.roi = result.objects[i].roi;

    sensor_msgs::msg::Image::SharedPtr mask_ptr = cv_bridge::CvImage(
      std_msgs::msg::Header(), "mono16", result.objects[i].mask).toImageMsg();
    mask_ptr->header.stamp = msg->header.stamp;
    mask_ptr->header.frame_id = msg->header.frame_id;
    object.segmented_binary_mask = *mask_ptr;

    object.centroid = result.objects[i].centroid;
    object.length   = result.objects[i].length;
    object.breadth  = result.objects[i].breadth;
    object.height   = result.objects[i].height;
    object.axis     = result.objects[i].axis;

    sensor_msgs::msg::PointCloud2 output_segmented_pcl;
    pcl::toROSMsg(result.objects[i].segmented_pcl, output_segmented_pcl);
    object.segmented_pcl = output_segmented_pcl;

    output_msg.object_ids.push_back(result.object_ids[i]);
    output_msg.objects.push_back(object);
  }

  geometry_msgs::msg::PoseArray pose_array;
  pose_array.header = msg->header;
  for (size_t i = 0; i < result.data_size; i++) {
    geometry_msgs::msg::Pose pose;
    pose.position = result.objects[i].centroid;
    pose.orientation.w = 1.0;
    pose_array.poses.push_back(pose);
  }

  const auto t_postprocess_end = std::chrono::high_resolution_clock::now();
  const auto process_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_postprocess_end - t_start).count();
  output_msg.process_time = process_time_ms;
  const auto t_publish_start = std::chrono::high_resolution_clock::now();
  if (visual_output_msg) {
    visual_pub.publish(*visual_output_msg);
    visual_pub->publish(*visual_output_msg);
  }
  tracking_pub->publish(output_msg);
  pose_pub->publish(pose_array);
  const auto t_publish_end = std::chrono::high_resolution_clock::now();

  const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_publish_end - t_start);
  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "[-FPS-]= %f\n",
    1000.0 / elapsed_time.count());

  const auto preprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_preprocess_end - t_start).count();
  const auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_infer_end - t_infer_start).count();
  const auto postprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_postprocess_end - t_postprocess_start).count();
  const auto publish_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_publish_end - t_publish_start).count();
  publishPerformanceMetrics(
    msg->header,
    "tracking",
    preprocess_ms,
    inference_ms,
    postprocess_ms,
    publish_ms,
    elapsed_time.count());

  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    if (ortAgent_.isService()) {
      ortAgent_.requestAddressed = true;
    }
  }
}

void EasyPerceptionDeployment::process_image_work(
  const sensor_msgs::msg::Image::SharedPtr msg)
{
  const auto t_start = std::chrono::high_resolution_clock::now();
  if (msg->height == 0) {
    RCLCPP_WARN(this->get_logger(), "Input image empty. Discarding.");
    return;
  }

  cv_bridge::CvImageConstPtr imgptr;
  if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
    imgptr = cv_bridge::toCvShare(msg);
  } else {
    imgptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  cv::Mat img = imgptr->image;

  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    checkOrtAgentIsInitialized(img.rows, img.cols);
  }

  const auto t_preprocess_end = std::chrono::high_resolution_clock::now();
  auto t_infer_start = t_preprocess_end;
  auto t_infer_end = t_preprocess_end;
  auto t_postprocess_start = t_preprocess_end;
  auto t_postprocess_end = t_preprocess_end;
  auto t_publish_start = t_preprocess_end;
  auto t_publish_end = t_preprocess_end;

  cv::Mat resultImg;
  int precision_level = 0;
  bool visualize = false;
  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    precision_level = ortAgent_.precision_level;
    visualize = ortAgent_.isVisualize();
  }

  switch (precision_level) {
    case 2:
      {
        EPD::EPDObjectDetection result;
        t_infer_start = std::chrono::high_resolution_clock::now();
        {
          std::lock_guard<std::mutex> ort_guard(ort_mutex_);
          result = ortAgent_.p2_ort_session->infer(img);
          EPD::activateUseCase(
            img,
            result.bboxes,
            result.classIndices,
            result.scores,
            result.masks,
            ortAgent_.classNames,
            ortAgent_.useCaseMode,
            ortAgent_.countClassNames,
            ortAgent_.template_color_path,
            ortAgent_.color_match_histogram_metric);
        }
        t_infer_end = std::chrono::high_resolution_clock::now();
        t_postprocess_start = t_infer_end;

        EPD::EPDObjectDetection output_obj(result.bboxes.size());
        output_obj.bboxes = result.bboxes;
        output_obj.classIndices = result.classIndices;
        output_obj.scores = result.scores;

        t_postprocess_end = std::chrono::high_resolution_clock::now();
        t_publish_start = t_postprocess_end;

        if (visualize) {
          {
            std::lock_guard<std::mutex> ort_guard(ort_mutex_);
            resultImg = ortAgent_.visualize(output_obj, img);
          }
          sensor_msgs::msg::Image::SharedPtr output_msg =
            cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", resultImg).toImageMsg();
          visual_pub.publish(*output_msg);
        } else {
          epd_msgs::msg::EPDObjectDetection output_msg;
          output_msg.header = msg->header;
          for (size_t i = 0; i < output_obj.data_size; i++) {
            output_msg.class_indices.push_back(output_obj.classIndices[i]);
            output_msg.scores.push_back(output_obj.scores[i]);

            sensor_msgs::msg::RegionOfInterest roi;
            roi.x_offset = output_obj.bboxes[i][0];
            roi.y_offset = output_obj.bboxes[i][1];
            roi.width = output_obj.bboxes[i][2] - output_obj.bboxes[i][0];
            roi.height = output_obj.bboxes[i][3] - output_obj.bboxes[i][1];
            roi.do_rectify = false;
            output_msg.bboxes.push_back(roi);
          }
          p2_pub->publish(output_msg);
          visual_pub->publish(*output_msg);
        }

        epd_msgs::msg::EPDObjectDetection output_msg;
        output_msg.header = msg->header;
        for (size_t i = 0; i < output_obj.data_size; i++) {
          output_msg.class_indices.push_back(output_obj.classIndices[i]);
          output_msg.scores.push_back(output_obj.scores[i]);

          sensor_msgs::msg::RegionOfInterest roi;
          roi.x_offset = output_obj.bboxes[i][0];
          roi.y_offset = output_obj.bboxes[i][1];
          roi.width = output_obj.bboxes[i][2] - output_obj.bboxes[i][0];
          roi.height = output_obj.bboxes[i][3] - output_obj.bboxes[i][1];
          roi.do_rectify = false;
          output_msg.bboxes.push_back(roi);
        }
        p2_pub->publish(output_msg);
        t_publish_end = std::chrono::high_resolution_clock::now();
        break;
      }
    case 3:
      {
        EPD::EPDObjectDetection result;
        t_infer_start = std::chrono::high_resolution_clock::now();
        {
          std::lock_guard<std::mutex> ort_guard(ort_mutex_);
          result = ortAgent_.p3_ort_session->infer(img);
          EPD::activateUseCase(
            img,
            result.bboxes,
            result.classIndices,
            result.scores,
            result.masks,
            ortAgent_.classNames,
            ortAgent_.useCaseMode,
            ortAgent_.countClassNames,
            ortAgent_.template_color_path,
            ortAgent_.color_match_histogram_metric);
        }
        t_infer_end = std::chrono::high_resolution_clock::now();
        t_postprocess_start = t_infer_end;

        EPD::EPDObjectDetection output_obj(result.bboxes.size());
        output_obj.bboxes = result.bboxes;
        output_obj.classIndices = result.classIndices;
        output_obj.scores = result.scores;
        output_obj.masks = result.masks;

        t_postprocess_end = std::chrono::high_resolution_clock::now();
        t_publish_start = t_postprocess_end;

        if (visualize) {
          {
            std::lock_guard<std::mutex> ort_guard(ort_mutex_);
            resultImg = ortAgent_.visualize(output_obj, img);
          }
          sensor_msgs::msg::Image::SharedPtr output_msg =
            cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", resultImg).toImageMsg();
          visual_pub.publish(*output_msg);
        } else {
          epd_msgs::msg::EPDObjectDetection output_msg;
          output_msg.header = msg->header;
          for (size_t i = 0; i < output_obj.data_size; i++) {
            output_msg.class_indices.push_back(output_obj.classIndices[i]);
            output_msg.scores.push_back(output_obj.scores[i]);

            sensor_msgs::msg::RegionOfInterest roi;
            roi.x_offset = output_obj.bboxes[i][0];
            roi.y_offset = output_obj.bboxes[i][1];
            roi.width = output_obj.bboxes[i][2] - output_obj.bboxes[i][0];
            roi.height = output_obj.bboxes[i][3] - output_obj.bboxes[i][1];
            roi.do_rectify = false;
            output_msg.bboxes.push_back(roi);

            sensor_msgs::msg::Image::SharedPtr mask =
              cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", output_obj.masks[i]).toImageMsg();
            mask->header.stamp = msg->header.stamp;
            mask->header.frame_id = msg->header.frame_id;
            output_msg.masks.push_back(*mask);
          }
          p3_pub->publish(output_msg);
          visual_pub->publish(*output_msg);
        }

        epd_msgs::msg::EPDObjectDetection output_msg;
        output_msg.header = msg->header;
        for (size_t i = 0; i < output_obj.data_size; i++) {
          output_msg.class_indices.push_back(output_obj.classIndices[i]);
          output_msg.scores.push_back(output_obj.scores[i]);

          sensor_msgs::msg::RegionOfInterest roi;
          roi.x_offset = output_obj.bboxes[i][0];
          roi.y_offset = output_obj.bboxes[i][1];
          roi.width = output_obj.bboxes[i][2] - output_obj.bboxes[i][0];
          roi.height = output_obj.bboxes[i][3] - output_obj.bboxes[i][1];
          roi.do_rectify = false;
          output_msg.bboxes.push_back(roi);

          sensor_msgs::msg::Image::SharedPtr mask =
            cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", output_obj.masks[i]).toImageMsg();
          mask->header.stamp = msg->header.stamp;
          mask->header.frame_id = msg->header.frame_id;
          output_msg.masks.push_back(*mask);
        }
        p3_pub->publish(output_msg);
        t_publish_end = std::chrono::high_resolution_clock::now();
        break;
      }
    default:
      RCLCPP_ERROR(
        this->get_logger(),
        "Unsupported precision level %u for image classification/detection.",
        ortAgent_.precision_level);
      t_publish_end = std::chrono::high_resolution_clock::now();
      break;
  }

  const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_publish_end - t_start);
  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "[-FPS-]= %f\n",
    1000.0 / elapsed_time.count());

  const auto preprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_preprocess_end - t_start).count();
  const auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_infer_end - t_infer_start).count();
  const auto postprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_postprocess_end - t_postprocess_start).count();
  const auto publish_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    t_publish_end - t_publish_start).count();
  publishPerformanceMetrics(
    msg->header,
    "image",
    preprocess_ms,
    inference_ms,
    postprocess_ms,
    publish_ms,
    elapsed_time.count());
}

void EasyPerceptionDeployment::worker_loop()
{
  while (rclcpp::ok()) {
    sensor_msgs::msg::Image::SharedPtr image_msg;
    sensor_msgs::msg::Image::SharedPtr depth_msg;
    sensor_msgs::msg::CameraInfo::SharedPtr camera_info;
    bool do_localize = false;
    bool do_tracking = false;
    bool do_image = false;

    {
      std::unique_lock<std::mutex> lock(data_mutex_);
      data_cv_.wait(lock, [this]() {
        return worker_stop_ || image_pending_ || localize_pending_ || tracking_pending_;
      });

      if (worker_stop_) {
        return;
      }

      if (localize_pending_) {
        image_msg = latest_image_;
        depth_msg = latest_depth_image_;
        camera_info = latest_camera_info_;
        localize_pending_ = false;
        do_localize = true;
      } else if (tracking_pending_) {
        image_msg = latest_image_;
        depth_msg = latest_depth_image_;
        camera_info = latest_camera_info_;
        tracking_pending_ = false;
        do_tracking = true;
      } else if (image_pending_) {
        image_msg = latest_image_;
        image_pending_ = false;
        do_image = true;
      }
    }

    if (do_localize) {
      if (image_msg && depth_msg && camera_info) {
        process_localize_work(image_msg, depth_msg, camera_info);
      }
      continue;
    }

    if (do_tracking) {
      if (image_msg && depth_msg && camera_info) {
        process_tracking_work(image_msg, depth_msg, camera_info);
      }
      continue;
    }

    if (do_image && image_msg) {
      process_image_work(image_msg);
    }
  }
}

#endif  // EPD_UTILS_LIB__EASY_PERCEPTION_DEPLOYMENT_HPP_
