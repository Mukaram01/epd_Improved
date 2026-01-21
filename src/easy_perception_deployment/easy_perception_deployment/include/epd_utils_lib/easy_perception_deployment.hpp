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
#include <string>
#include <memory>
#include <functional>
#include <stdexcept>  // FIX: for std::runtime_error
#include <thread>
#include <mutex>
#include <condition_variable>

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
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;

  /*! \brief An alias definition for SyncPolicy that is used below for sync_ object.*/
  typedef message_filters::sync_policies::ApproximateTime
    <sensor_msgs::msg::Image, sensor_msgs::msg::Image,
      sensor_msgs::msg::CameraInfo> SyncPolicy;

  /*! \brief A policy-synchronized subscriber member variable
  to receive rectified 2D RGB images.
  */
  message_filters::Subscriber<sensor_msgs::msg::Image> localize_image_rgb;
  /*! \brief A policy-synchronized subscriber member variable
  to receive rectified 2D Depth images.
  */
  message_filters::Subscriber<sensor_msgs::msg::Image> localize_image_depth;
  /*! \brief A policy-synchronized subscriber member variable
  to receive camera information.
  */
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> localize_cam_info;
  /*! \brief A Synchronizer policy member variable.*/
  message_filters::Synchronizer<SyncPolicy> sync_;

  /*! \brief A publisher member variable to output visualization of inference
  results*/
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visual_pub;
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

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void image_worker_loop();

  void hasCameraChanged(
    const int img_height,
    const int img_width) const;

  void checkOrtAgentIsInitialized(
    const int img_height,
    const int img_width) const;

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
};

EasyPerceptionDeployment::EasyPerceptionDeployment(void)
: Node("easy_perception_deployment"),
  localize_image_rgb(this, "/camera/color/image_raw"),
  localize_image_depth(this, "/camera/depth/image_rect_raw"),
  localize_cam_info(this, "/camera/color/camera_info"),
  sync_(SyncPolicy(10), localize_image_rgb, localize_image_depth, localize_cam_info)
{
  rclcpp::SubscriptionOptions subscription_options;
  subscription_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions publisher_options;
  publisher_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  const auto image_qos = rclcpp::SensorDataQoS().keep_last(1).best_effort();
  const auto camera_info_qos = rclcpp::SensorDataQoS().keep_last(1);

  // Creating Subscriber to get Input Image.
  image_sub = this->create_subscription<sensor_msgs::msg::Image>(
    "/easy_perception_deployment/image_input",
    image_qos,
    std::bind(&EasyPerceptionDeployment::image_callback, this, std::placeholders::_1),
    subscription_options);

  // Creating Publisher to output Visualizable P2 and P3 Detection Results.
  visual_pub = this->create_publisher<sensor_msgs::msg::Image>(
    "/easy_perception_deployment/image_output",
    10,
    publisher_options);

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
  if (ortAgent_.useCaseMode == 3) {
    localize_image_rgb.subscribe();
    localize_image_depth.subscribe();
    localize_cam_info.subscribe();
    sync_.registerCallback(&EasyPerceptionDeployment::localize_callback, this);
    image_sub.reset();
  } else if (ortAgent_.useCaseMode == 4) {
    localize_image_rgb.subscribe();
    localize_image_depth.subscribe();
    localize_cam_info.subscribe();
    sync_.registerCallback(&EasyPerceptionDeployment::tracking_callback, this);
    image_sub.reset();
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
    [this, subscription_options](
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
      } else {
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
      RCLCPP_INFO(this->get_logger(), "[- Input RGB Image Topic -] - %s", rgb_topic.c_str());
      RCLCPP_INFO(
        this->get_logger(),
        "[- Input Depth Image Topic -] - %s",
        depth_topic.c_str());
      RCLCPP_INFO(
        this->get_logger(),
        "[- Camera Info Topic -] - %s",
        camera_info_topic.c_str());
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

  EPD::EPDObjectLocalization result;
  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    result = ortAgent_.p3_ort_session->infer(
      img,
      depth_img,
      *camera_info,
      camera_to_plane_distance_mm);
  }

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
    visual_pub->publish(*output_msg);
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

    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header = msg->header;
    for (size_t i = 0; i < result.data_size; i++) {
      geometry_msgs::msg::Pose pose;
      pose.position = result.objects[i].centroid;
      pose.orientation.w = 1.0;
      pose_array.poses.push_back(pose);
    }
    pose_pub->publish(pose_array);
  }

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

  EPD::EPDObjectTracking result;
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
    visual_pub->publish(*output_msg);
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

    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header = msg->header;
    for (size_t i = 0; i < result.data_size; i++) {
      geometry_msgs::msg::Pose pose;
      pose.position = result.objects[i].centroid;
      pose.orientation.w = 1.0;
      pose_array.poses.push_back(pose);
    }
    pose_pub->publish(pose_array);
  }

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
  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    precision_level = ortAgent_.precision_level;
    visualize = ortAgent_.isVisualize();
  }

  switch (precision_level) {
    case 2:
      {
        EPD::EPDObjectDetection result;
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

        EPD::EPDObjectDetection output_obj(result.bboxes.size());
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
        break;
      }
    case 3:
      {
        EPD::EPDObjectDetection result;
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

        EPD::EPDObjectDetection output_obj(result.bboxes.size());
        output_obj.bboxes = result.bboxes;
        output_obj.classIndices = result.classIndices;
        output_obj.scores = result.scores;
        output_obj.masks = result.masks;

        if (visualize) {
          {
            std::lock_guard<std::mutex> ort_guard(ort_mutex_);
            resultImg = ortAgent_.visualize(output_obj, img);
          }
          sensor_msgs::msg::Image::SharedPtr output_msg =
            cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", resultImg).toImageMsg();
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
