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
#include <cmath>
#include <string>
#include <memory>
#include <functional>
#include <stdexcept>  // FIX: for std::runtime_error
#include <thread>
#include <mutex>
#include <condition_variable>

#include <Eigen/Dense>

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
#include "message_filters/connection.h"

// EPD_UTILS LIB
#include "epd_utils_lib/epd_container.hpp"
#include "epd_msgs/msg/epd_image_classification.hpp"
#include "epd_msgs/msg/epd_object_detection.hpp"
#include "epd_msgs/msg/epd_object_localization.hpp"
#include "epd_msgs/msg/epd_object_tracking.hpp"
#include "epd_msgs/msg/localized_object.hpp"
#include "epd_msgs/srv/perception.hpp"
#include "epd_utils_lib/usecase_config.hpp"
#include "epd_utils_lib/message_utils.hpp"

#include "pcl_conversions/pcl_conversions.h"
#include "pcl/common/centroid.h"
#include "pcl/common/eigen.h"

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
  void process_image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
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
  image_transport::Subscriber depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  /*! \brief An alias definition for SyncPolicy that is used below for sync_ object.*/
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo>;

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

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void image_worker_loop();

  void hasCameraChanged(
    const int img_height,
    const int img_width) const;

  void checkOrtAgentIsInitialized(
    const int img_height,
    const int img_width) const;

  void subscribeImageInput();
  void subscribeLocalizeInputs();
  void subscribeDetectionDepthInputs();
  void enableDetectionInputs();
  void disableDetectionInputs();
  void enableLocalizeInputs(const int use_case_mode);
  void disableLocalizeInputs();
  std::string resolveDepthTransport(const std::string & transport) const;

  std::string rgb_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string image_transport_;
  std::string depth_transport_;
  rmw_qos_profile_t sensor_qos_profile_;
  bool image_input_active_{false};
  bool depth_input_active_{false};
  bool localize_input_active_{false};
  int sync_callback_mode_{-1};
  message_filters::Connection sync_connection_;
  void process_image_work(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void process_localize_work(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr camera_info);
  void process_tracking_work(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr camera_info);
  void worker_loop();
  geometry_msgs::msg::Pose buildObjectPose(
    const geometry_msgs::msg::Point & centroid,
    const geometry_msgs::msg::Vector3 & axis,
    const pcl::PointCloud<pcl::PointXYZ> & segmented_pcl) const;
  geometry_msgs::msg::Quaternion buildOrientationFromAxisOrPcl(
    const geometry_msgs::msg::Vector3 & axis,
    const pcl::PointCloud<pcl::PointXYZ> & segmented_pcl) const;

  std::mutex data_mutex_;
  std::condition_variable data_cv_;
  std::thread worker_thread_;
  bool worker_stop_{false};
  bool image_pending_{false};
  bool localize_pending_{false};
  bool tracking_pending_{false};
  sensor_msgs::msg::Image::ConstSharedPtr latest_image_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_depth_image_;
  sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info_;

  std::mutex ort_mutex_;
};

EasyPerceptionDeployment::EasyPerceptionDeployment(void)
: Node("easy_perception_deployment"),
  sync_(SyncPolicy(10), localize_image_rgb, localize_image_depth, localize_cam_info),
  sensor_qos_profile_(rclcpp::SensorDataQoS().get_rmw_qos_profile())
{
  rclcpp::PublisherOptions publisher_options;
  publisher_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
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

  if (ortAgent_.publish_detection_segmentation &&
    ortAgent_.useCaseMode <= EPD::COLOR_MATCHING_MODE)
  {
    enableDetectionInputs();
  }

  subscribeImageInput();

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

  // If useCaseMode is detected to be Localization or Tracking,
  // Subscribe to all synchronized ROS2 topics.
  if (ortAgent_.useCaseMode == EPD::LOCALISATION_MODE ||
    ortAgent_.useCaseMode == EPD::TRACKING_MODE)
  {
    enableLocalizeInputs(ortAgent_.useCaseMode);
  } else {
    disableLocalizeInputs();
  }

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
        response->tracking_enabled = (ortAgent_.useCaseMode == EPD::TRACKING_MODE);
      }

      int use_case_mode = 0;
      {
        std::lock_guard<std::mutex> ort_guard(ort_mutex_);
        use_case_mode = ortAgent_.useCaseMode;
      }

      if (use_case_mode == EPD::LOCALISATION_MODE ||
        use_case_mode == EPD::TRACKING_MODE)
      {
        enableLocalizeInputs(use_case_mode);
        disableDetectionInputs();
      } else {
        disableLocalizeInputs();
        if (!image_input_active_) {
          subscribeImageInput();
        }
        if (ortAgent_.publish_detection_segmentation &&
          use_case_mode <= EPD::COLOR_MATCHING_MODE)
        {
          enableDetectionInputs();
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
    sensor_qos_profile_,
    subscription_options);
  localize_image_depth.subscribe(
    this,
    depth_topic_,
    depth_transport_,
    sensor_qos_profile_,
    subscription_options);
  localize_cam_info.subscribe(
    this,
    camera_info_topic_,
    sensor_qos_profile_,
    subscription_options);
}

void EasyPerceptionDeployment::subscribeDetectionDepthInputs()
{
  if (depth_input_active_) {
    return;
  }

  depth_sub_ = image_transport::create_subscription(
    this,
    depth_topic_,
    std::bind(&EasyPerceptionDeployment::depth_callback, this, std::placeholders::_1),
    depth_transport_,
    sensor_qos_profile_);
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic_,
    rclcpp::SensorDataQoS().keep_last(1),
    std::bind(&EasyPerceptionDeployment::camera_info_callback, this, std::placeholders::_1));
  depth_input_active_ = true;
}

void EasyPerceptionDeployment::enableDetectionInputs()
{
  if (!depth_input_active_) {
    subscribeDetectionDepthInputs();
  }
}

void EasyPerceptionDeployment::disableDetectionInputs()
{
  if (!depth_input_active_) {
    return;
  }

  depth_sub_.shutdown();
  camera_info_sub_.reset();
  depth_input_active_ = false;
}

void EasyPerceptionDeployment::enableLocalizeInputs(const int use_case_mode)
{
  disableDetectionInputs();

  if (image_input_active_) {
    image_sub.shutdown();
    image_input_active_ = false;
  }

  if (!localize_input_active_) {
    subscribeLocalizeInputs();
    localize_input_active_ = true;
  }

  if (sync_callback_mode_ == use_case_mode) {
    return;
  }

  sync_connection_.disconnect();
  if (use_case_mode == EPD::LOCALISATION_MODE) {
    sync_connection_ =
      sync_.registerCallback(&EasyPerceptionDeployment::localize_callback, this);
  } else if (use_case_mode == EPD::TRACKING_MODE) {
    sync_connection_ =
      sync_.registerCallback(&EasyPerceptionDeployment::tracking_callback, this);
  }
  sync_callback_mode_ = use_case_mode;
}

void EasyPerceptionDeployment::disableLocalizeInputs()
{
  sync_connection_.disconnect();
  sync_callback_mode_ = -1;

  if (!localize_input_active_) {
    return;
  }

  localize_image_rgb.unsubscribe();
  localize_image_depth.unsubscribe();
  localize_cam_info.unsubscribe();
  localize_input_active_ = false;
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
  const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_image_ = msg;
    image_pending_ = true;
  }
  data_cv_.notify_one();
}

void EasyPerceptionDeployment::image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  this->process_image_callback(msg);
}

void EasyPerceptionDeployment::depth_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_depth_image_ = msg;
  }
}

void EasyPerceptionDeployment::camera_info_callback(
  const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_camera_info_ = msg;
  }
}

void EasyPerceptionDeployment::process_localize_work(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
  const sensor_msgs::msg::CameraInfo::SharedPtr camera_info)
{
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

  auto begin = std::chrono::high_resolution_clock::now();

  EPD::EPDObjectLocalization result([&]() {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    auto inference_result = ortAgent_.p3_ort_session->infer(
      img,
      depth_img,
      *camera_info,
      camera_to_plane_distance_mm);
    const size_t input_size = inference_result.data_size;
    EPD::EPDObjectLocalization initialized_result(input_size);
    initialized_result = std::move(inference_result);
    return initialized_result;
  }());

  cv::Mat resultImg;

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

    sensor_msgs::msg::Image::SharedPtr output_msg =
      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", resultImg).toImageMsg();
    visual_pub.publish(*output_msg);
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

  output_msg.objects.reserve(result.data_size);

  geometry_msgs::msg::PoseArray pose_array;
  pose_array.header = msg->header;
  pose_array.poses.reserve(result.data_size);

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
    object.pose = buildObjectPose(
      object.centroid,
      object.axis,
      result.objects[i].segmented_pcl);
    pose_array.poses.push_back(object.pose);

    output_msg.objects.push_back(object);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "[-FPS-]= %f\n",
    1000.0 / elapsedTime.count());

  output_msg.process_time = elapsedTime.count();
  localize_pub->publish(output_msg);
  pose_pub->publish(pose_array);

  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    if (ortAgent_.isService()) {
      ortAgent_.requestAddressed = true;
    }
  }
}

void EasyPerceptionDeployment::process_tracking_work(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
  const sensor_msgs::msg::CameraInfo::SharedPtr camera_info)
{
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

  auto begin = std::chrono::high_resolution_clock::now();

  EPD::EPDObjectTracking result([&]() {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    auto inference_result = ortAgent_.p3_ort_session->infer(
      img,
      depth_img,
      *camera_info,
      camera_to_plane_distance_mm,
      ortAgent_.tracker_type,
      ortAgent_.trackers,
      ortAgent_.tracker_logs,
      ortAgent_.tracker_results);
    const size_t input_size = inference_result.data_size;
    EPD::EPDObjectTracking initialized_result(input_size);
    initialized_result = std::move(inference_result);
    return initialized_result;
  }());

  cv::Mat resultImg;

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

    sensor_msgs::msg::Image::SharedPtr output_msg =
      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", resultImg).toImageMsg();
    visual_pub.publish(*output_msg);
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

  output_msg.object_ids.reserve(result.data_size);
  output_msg.objects.reserve(result.data_size);

  geometry_msgs::msg::PoseArray pose_array;
  pose_array.header = msg->header;
  pose_array.poses.reserve(result.data_size);

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
    object.pose = buildObjectPose(
      object.centroid,
      object.axis,
      result.objects[i].segmented_pcl);
    pose_array.poses.push_back(object.pose);

    output_msg.object_ids.push_back(result.object_ids[i]);
    output_msg.objects.push_back(object);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "[-FPS-]= %f\n",
    1000.0 / elapsedTime.count());

  output_msg.process_time = elapsedTime.count();
  tracking_pub->publish(output_msg);
  pose_pub->publish(pose_array);

  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    if (ortAgent_.isService()) {
      ortAgent_.requestAddressed = true;
    }
  }
}

geometry_msgs::msg::Pose EasyPerceptionDeployment::buildObjectPose(
  const geometry_msgs::msg::Point & centroid,
  const geometry_msgs::msg::Vector3 & axis,
  const pcl::PointCloud<pcl::PointXYZ> & segmented_pcl) const
{
  geometry_msgs::msg::Pose pose;
  pose.position = centroid;
  pose.orientation = buildOrientationFromAxisOrPcl(axis, segmented_pcl);
  return pose;
}

geometry_msgs::msg::Quaternion EasyPerceptionDeployment::buildOrientationFromAxisOrPcl(
  const geometry_msgs::msg::Vector3 & axis,
  const pcl::PointCloud<pcl::PointXYZ> & segmented_pcl) const
{
  Eigen::Vector3f axis_vec(axis.x, axis.y, axis.z);
  if (axis_vec.norm() < 1e-6f && !segmented_pcl.empty()) {
    Eigen::Vector4f centerpoint;
    Eigen::Vector3f eigenvalues;
    Eigen::Matrix3f eigenvectors;
    Eigen::Matrix3f covariance_matrix;

    pcl::compute3DCentroid(segmented_pcl, centerpoint);
    pcl::computeCovarianceMatrix(segmented_pcl, centerpoint, covariance_matrix);
    pcl::eigen33(covariance_matrix, eigenvectors, eigenvalues);

    axis_vec = Eigen::Vector3f(
      eigenvectors.col(2)(0),
      eigenvectors.col(2)(1),
      eigenvectors.col(2)(2));
  }

  if (axis_vec.norm() < 1e-6f) {
    geometry_msgs::msg::Quaternion identity;
    identity.w = 1.0;
    return identity;
  }

  Eigen::Vector3f x_axis = axis_vec.normalized();
  Eigen::Vector3f z_reference(0.0f, 0.0f, 1.0f);
  if (std::abs(x_axis.dot(z_reference)) > 0.95f) {
    z_reference = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
  }

  Eigen::Vector3f y_axis = z_reference.cross(x_axis);
  if (y_axis.norm() < 1e-6f) {
    geometry_msgs::msg::Quaternion identity;
    identity.w = 1.0;
    return identity;
  }
  y_axis.normalize();
  Eigen::Vector3f z_axis = x_axis.cross(y_axis);
  z_axis.normalize();

  Eigen::Matrix3f rotation;
  rotation.col(0) = x_axis;
  rotation.col(1) = y_axis;
  rotation.col(2) = z_axis;

  Eigen::Quaternionf quat(rotation);
  quat.normalize();

  geometry_msgs::msg::Quaternion orientation;
  orientation.x = quat.x();
  orientation.y = quat.y();
  orientation.z = quat.z();
  orientation.w = quat.w();
  return orientation;
}

void EasyPerceptionDeployment::process_image_work(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
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

  auto begin = std::chrono::high_resolution_clock::now();

  cv::Mat resultImg;
  int precision_level = 0;
  bool visualize = false;
  bool publish_segmentation = false;
  sensor_msgs::msg::Image::ConstSharedPtr depth_msg;
  sensor_msgs::msg::CameraInfo::SharedPtr camera_info;
  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    precision_level = ortAgent_.precision_level;
    visualize = ortAgent_.isVisualize();
    publish_segmentation = ortAgent_.publish_detection_segmentation;
  }
  if (publish_segmentation) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    depth_msg = latest_depth_image_;
    camera_info = latest_camera_info_;
  }

  switch (precision_level) {
    case 2:
      {
        EPD::EPDObjectDetection result(0);
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

        EPD::EPDObjectDetection output_obj(result.data_size);
        output_obj.bboxes = result.bboxes;
        output_obj.classIndices = result.classIndices;
        output_obj.scores = result.scores;

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
        }

        break;
      }
    case 3:
      {
        EPD::EPDObjectDetection result(0);
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

        EPD::EPDObjectDetection output_obj(result.data_size);
        output_obj.bboxes = result.bboxes;
        output_obj.classIndices = result.classIndices;
        output_obj.scores = result.scores;
        output_obj.masks = result.masks;

        std::vector<sensor_msgs::msg::PointCloud2> segmented_pcls;
        bool has_segmented_pcls = false;
        bool depth_ready = publish_segmentation && depth_msg && camera_info;
        cv::Mat depth_img;
        bool depth_is_float = false;
        if (depth_ready) {
          if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
            depth_msg->encoding == sensor_msgs::image_encodings::MONO16)
          {
            cv_bridge::CvImageConstPtr depth_imageptr =
              cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
            depth_img = depth_imageptr->image;
            depth_is_float = false;
          } else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
            cv_bridge::CvImageConstPtr depth_imageptr =
              cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
            depth_img = depth_imageptr->image;
            depth_is_float = true;
          } else {
            RCLCPP_WARN_THROTTLE(
              this->get_logger(),
              *this->get_clock(),
              2000,
              "Unsupported depth encoding '%s' for detection segmentation.",
              depth_msg->encoding.c_str());
            depth_ready = false;
          }
        }

        if (depth_ready && (depth_img.rows != img.rows || depth_img.cols != img.cols)) {
          RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "Depth image size (%dx%d) does not match RGB image size (%dx%d). Skipping segmentation.",
            depth_img.cols,
            depth_img.rows,
            img.cols,
            img.rows);
          depth_ready = false;
        }

        if (depth_ready) {
          const double camera_to_plane_distance_mm =
            this->get_parameter("camera_to_plane_distance_mm").as_double();
          const double max_depth_m = camera_to_plane_distance_mm * 0.001;
          const float ppx = static_cast<float>(camera_info->k.at(2));
          const float fx = static_cast<float>(camera_info->k.at(0));
          const float ppy = static_cast<float>(camera_info->k.at(5));
          const float fy = static_cast<float>(camera_info->k.at(4));

          segmented_pcls.reserve(output_obj.data_size);
          for (size_t i = 0; i < output_obj.data_size; i++) {
            const auto & bbox = output_obj.bboxes[i];
            const int left = std::clamp(bbox[0], 0, img.cols);
            const int top = std::clamp(bbox[1], 0, img.rows);
            const int right = std::clamp(bbox[2], 0, img.cols);
            const int bottom = std::clamp(bbox[3], 0, img.rows);
            const int width = right - left;
            const int height = bottom - top;

            sensor_msgs::msg::PointCloud2 cloud_msg;
            if (left != bbox[0] || top != bbox[1] || right != bbox[2] || bottom != bbox[3]) {
              RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Clamped bbox for segmentation: [%d, %d, %d, %d] -> [%d, %d, %d, %d]",
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                left,
                top,
                right,
                bottom);
            }
            if (width <= 0 || height <= 0) {
              RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Skipping invalid bbox for segmentation after clamping: [%d, %d, %d, %d]",
                left,
                top,
                right,
                bottom);
              segmented_pcls.push_back(cloud_msg);
              continue;
            }

            cv::Mat resized_mask;
            cv::resize(output_obj.masks[i], resized_mask, cv::Size(width, height));
            cv::Mat mask_binary = resized_mask > 0.5;
            cv::Mat mask_u8;
            mask_binary.convertTo(mask_u8, CV_8U);

            pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud(
              new pcl::PointCloud<pcl::PointXYZ>);
            segmented_cloud->header.frame_id = depth_msg->header.frame_id;
            segmented_cloud->is_dense = true;

            for (int row = 0; row < height; row++) {
              for (int col = 0; col < width; col++) {
                if (mask_u8.at<uchar>(row, col) == 0) {
                  continue;
                }

                const int pixel_x = left + col;
                const int pixel_y = top + row;

                float z = 0.0f;
                if (depth_is_float) {
                  z = depth_img.at<float>(pixel_y, pixel_x);
                } else {
                  z = static_cast<float>(depth_img.at<uint16_t>(pixel_y, pixel_x)) * 0.001f;
                }

                if (std::abs(z) < 0.0001f || (max_depth_m > 0.0 && z > max_depth_m)) {
                  continue;
                }

                const float x = (static_cast<float>(pixel_x) - ppx) / fx * z;
                const float y = (static_cast<float>(pixel_y) - ppy) / fy * z;
                segmented_cloud->points.emplace_back(x, y, z);
              }
            }

            pcl::toROSMsg(*segmented_cloud, cloud_msg);
            cloud_msg.header = depth_msg->header;
            segmented_pcls.push_back(cloud_msg);
          }
          has_segmented_pcls = true;
        }

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

            if (publish_segmentation) {
              sensor_msgs::msg::Image::SharedPtr mask =
                cv_bridge::CvImage(
                  std_msgs::msg::Header(),
                  "32FC1",
                  output_obj.masks[i]).toImageMsg();
              mask->header.stamp = msg->header.stamp;
              mask->header.frame_id = msg->header.frame_id;
              output_msg.masks.push_back(*mask);
            }
            if (has_segmented_pcls && i < segmented_pcls.size()) {
              output_msg.segmented_pcls.push_back(segmented_pcls[i]);
            }
          }
          p3_pub->publish(output_msg);
        }

        break;
      }
    default:
      RCLCPP_ERROR(
        this->get_logger(),
        "Unsupported precision level %u for image classification/detection.",
        ortAgent_.precision_level);
      break;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "[-FPS-]= %f\n",
    1000.0 / elapsedTime.count());
}

void EasyPerceptionDeployment::worker_loop()
{
  while (rclcpp::ok()) {
    sensor_msgs::msg::Image::ConstSharedPtr image_msg;
    sensor_msgs::msg::Image::ConstSharedPtr depth_msg;
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
