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
#include <sstream>
#include <memory>
#include <functional>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <Eigen/Dense>

// OpenCV LIB
#include "opencv2/opencv.hpp"

// ROS2 LIB
#if __has_include("cv_bridge/cv_bridge.hpp")
#include "cv_bridge/cv_bridge.hpp"
#else
#include "cv_bridge/cv_bridge.h"
#endif
#include <map>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/region_of_interest.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "image_transport/image_transport.hpp"
#include "image_transport/subscriber_filter.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
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
#include "epd_utils_lib/observation.hpp"
#include "epd_utils_lib/inference_scheduler.hpp"
#include "epd_utils_lib/geometry_quality.hpp"
#include "epd_utils_lib/time_utils.hpp"
#include "epd_utils_lib/temporal_tracker.hpp"

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

  // Depth-enabled localization/tracking uses a bounded exact-timestamp
  // matcher. RealSense aligned RGB, depth and CameraInfo carry the same
  // sensor timestamp, so no ApproximateTime policy is required here.
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr localize_image_rgb_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr localize_image_depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr localize_cam_info_sub_;

  std::mutex localize_sync_mutex_;
  std::map<int64_t, sensor_msgs::msg::Image::SharedPtr> localize_rgb_cache_;
  std::map<int64_t, sensor_msgs::msg::Image::SharedPtr> localize_depth_cache_;
  std::map<int64_t, sensor_msgs::msg::CameraInfo::SharedPtr> localize_info_cache_;
  static constexpr std::size_t kLocalizeSyncCacheSize = 8;

  uint64_t localize_rgb_callback_count_{0};
  uint64_t localize_depth_callback_count_{0};
  uint64_t localize_info_callback_count_{0};
  uint64_t localize_triplet_dispatch_count_{0};

  int64_t localize_last_rgb_key_{0};
  int64_t localize_last_depth_key_{0};
  int64_t localize_last_info_key_{0};

  static int64_t stampKey(const builtin_interfaces::msg::Time & stamp)
  {
    return static_cast<int64_t>(stamp.sec) * 1000000000LL +
           static_cast<int64_t>(stamp.nanosec);
  }

  void tryDispatchLocalizedTriplet(int64_t key);
  void pruneLocalizeSyncCaches();

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
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    inference_diagnostics_pub_;

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
  void rgb_input_watchdog_callback();
  void inference_diagnostics_callback();
  void image_worker_loop();

  void hasCameraChanged(
    const int img_height,
    const int img_width) const;

  void ensureOrtAgentInitialized(
    const int img_height,
    const int img_width);

  void subscribeImageInput();
  void subscribeLocalizeInputs();
  void subscribeLocalizeNoDepth(const unsigned int use_case_mode);
  void subscribeDetectionDepthInputs();
  void enableDetectionInputs();
  void disableDetectionInputs();
  void enableLocalizeInputs(const unsigned int use_case_mode);
  void disableLocalizeInputs();
  std::string resolveDepthTransport(const std::string & transport) const;

  std::string rgb_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string image_transport_;
  std::string depth_transport_;
  double rgb_input_watchdog_timeout_s_{5.0};
  int slow_frame_warn_ms_{1000};
  double max_processing_fps_{0.0};
  double dropped_frame_log_period_s_{5.0};
  rmw_qos_profile_t sensor_qos_profile_;
  bool use_depth_{true};
  bool service_mode_{false};
  bool image_input_active_{false};
  bool depth_input_active_{false};
  bool localize_input_active_{false};
  bool localize_nodepth_active_{false};
  int sync_callback_mode_{-1};
  image_transport::Subscriber localize_rgb_nodepth_;
  void process_image_work(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void process_localize_work(
    const EPD::Observation::ConstSharedPtr & observation);
  void process_tracking_work(
    const EPD::Observation::ConstSharedPtr & observation);
  EPD::EPDObjectDetection applyDetectionFilters(
    const EPD::EPDObjectDetection & raw,
    float confidence_threshold,
    int max_detections) const;
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
  sensor_msgs::msg::Image::ConstSharedPtr latest_image_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_depth_image_;
  sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info_;
  EPD::LatestObservationStore observation_store_;
  EPD::LatestInferenceScheduler inference_scheduler_;
  EPD::LatestPerceptionResultStore perception_result_store_;
  std::string camera_id_{"camera"};
  rclcpp::Time last_rgb_frame_time_;
  rclcpp::Time last_synchronized_observation_time_;
  rclcpp::Time last_processed_frame_time_;
  rclcpp::Time last_drop_stats_log_time_;
  bool has_received_rgb_frame_{false};
  bool has_received_synchronized_observation_{false};
  bool has_processed_frame_{false};
  bool rgb_stream_missing_{false};
  uint64_t dropped_frames_overwritten_{0};
  uint64_t dropped_frames_rate_limited_{0};
  uint64_t logged_dropped_frames_overwritten_{0};
  uint64_t logged_dropped_frames_rate_limited_{0};
  rclcpp::TimerBase::SharedPtr rgb_input_watchdog_timer_;
  rclcpp::TimerBase::SharedPtr inference_diagnostics_timer_;
  rclcpp::Time metrics_start_time_;
  EPD::GeometryThresholds geometry_thresholds_;
  std::atomic<uint64_t> detections_total_{0};
  std::atomic<uint64_t> geometry_valid_total_{0};
  std::atomic<uint64_t> geometry_degraded_total_{0};
  std::atomic<uint64_t> geometry_invalid_total_{0};
  std::atomic<uint64_t> invalid_intrinsics_total_{0};
  std::atomic<uint64_t> invalid_mask_total_{0};
  std::atomic<uint64_t> insufficient_depth_total_{0};
  std::atomic<uint64_t> empty_cloud_total_{0};
  std::atomic<uint64_t> nonfinite_geometry_total_{0};
  std::atomic<int64_t> latest_valid_geometry_stamp_ns_{0};
  std::unique_ptr<EPD::TemporalTracker> temporal_tracker_;
  std::mutex temporal_tracker_mutex_;

  std::mutex ort_mutex_;

  std::mutex service_request_mutex_;
  uint64_t service_request_baseline_{0};
  bool service_request_active_{false};
  std::atomic<uint64_t> service_requests_{0};
  std::atomic<uint64_t> service_success_{0};
  std::atomic<uint64_t> service_timeout_{0};
  std::atomic<uint64_t> service_shutdown_abort_{0};
  std::atomic<uint64_t> service_baseline_observation_id_{0};
  std::atomic<uint64_t> last_service_result_observation_id_{0};
  rclcpp::CallbackGroup::SharedPtr service_callback_group_;
  rclcpp::CallbackGroup::SharedPtr sensor_callback_group_;
  rclcpp::CallbackGroup::SharedPtr localize_input_callback_group_;
};

EasyPerceptionDeployment::EasyPerceptionDeployment(void)
: Node("easy_perception_deployment"),
  sensor_qos_profile_(rclcpp::SensorDataQoS().get_rmw_qos_profile())
{
  service_mode_ = ortAgent_.isService();
  rclcpp::PublisherOptions publisher_options;
  publisher_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  service_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  // message_filters::Synchronizer is fed by three independent subscriptions.
  // Serialize those callbacks so ApproximateTime never mutates its queues from
  // multiple executor threads concurrently.  The service remains in its own
  // callback group, so waiting for inference cannot starve sensor delivery.
  sensor_callback_group_ =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // RGB/depth/CameraInfo callbacks are independent and the exact timestamp
  // matcher protects shared state with localize_sync_mutex_.
  localize_input_callback_group_ =
    this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  this->declare_parameter<double>("camera_to_plane_distance_mm", 1000.0);
  this->declare_parameter<std::string>("rgb_topic", "/camera/camera/color/image_raw");
  this->declare_parameter<std::string>("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw");
  this->declare_parameter<std::string>("camera_info_topic", "/camera/camera/color/camera_info");
  this->declare_parameter<std::string>("camera_id", "camera");
  this->declare_parameter<std::string>("image_transport", ortAgent_.image_transport);
  this->declare_parameter<std::string>("image_output_qos_reliability", "best_effort");
  this->declare_parameter<int>("image_output_qos_depth", 1);
  this->declare_parameter<bool>("use_depth", true);
  this->declare_parameter<double>("service_timeout_s", 10.0);
  this->declare_parameter<double>("rgb_input_watchdog_timeout_s", 5.0);
  this->declare_parameter<int>("slow_frame_warn_ms", 1000);
  this->declare_parameter<double>("max_processing_fps", 0.0);
  this->declare_parameter<int>("geometry_minimum_mask_pixels", 16);
  this->declare_parameter<int>("geometry_minimum_depth_pixels", 12);
  this->declare_parameter<double>("geometry_minimum_valid_depth_ratio", 0.20);
  this->declare_parameter<int>("geometry_minimum_cloud_points", 12);
  this->declare_parameter<double>("tracking_minimum_iou", 0.20);
  this->declare_parameter<double>("tracking_maximum_roi_distance_px", 100.0);
  this->declare_parameter<double>("tracking_maximum_3d_distance_m", 0.25);
  this->declare_parameter<int>("tracking_confirmation_hits", 2);
  this->declare_parameter<int>("tracking_maximum_missed_observations", 3);
  this->declare_parameter<int>("tracking_maximum_active_tracks", 64);
  // Test/development ingress may select an existing production use case without
  // rewriting the operator's persistent GUI configuration.
  this->declare_parameter<int>("usecase_mode_override", -1);
  this->declare_parameter<std::string>("tracker_type_override", "");

  rgb_topic_ = this->get_parameter("rgb_topic").as_string();
  depth_topic_ = this->get_parameter("depth_topic").as_string();
  camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();
  camera_id_ = this->get_parameter("camera_id").as_string();
  const int usecase_mode_override =
    static_cast<int>(this->get_parameter("usecase_mode_override").as_int());
  if (usecase_mode_override >= static_cast<int>(EPD::CLASSIFICATION_MODE) &&
    usecase_mode_override <= static_cast<int>(EPD::TRACKING_MODE))
  {
    ortAgent_.useCaseMode = static_cast<unsigned int>(usecase_mode_override);
  }
  const auto tracker_type_override = this->get_parameter("tracker_type_override").as_string();
  if (!tracker_type_override.empty()) {ortAgent_.tracker_type = tracker_type_override;}
  image_transport_ = this->get_parameter("image_transport").as_string();
  std::string image_output_qos_reliability =
    this->get_parameter("image_output_qos_reliability").as_string();
  int image_output_qos_depth = this->get_parameter("image_output_qos_depth").as_int();
  use_depth_ = this->get_parameter("use_depth").as_bool();
  rgb_input_watchdog_timeout_s_ = this->get_parameter("rgb_input_watchdog_timeout_s").as_double();
  slow_frame_warn_ms_ = this->get_parameter("slow_frame_warn_ms").as_int();
  max_processing_fps_ = this->get_parameter("max_processing_fps").as_double();
  geometry_thresholds_.minimum_mask_pixels = static_cast<size_t>(std::max<int64_t>(
      1, this->get_parameter("geometry_minimum_mask_pixels").as_int()));
  geometry_thresholds_.minimum_depth_pixels = static_cast<size_t>(std::max<int64_t>(
      1, this->get_parameter("geometry_minimum_depth_pixels").as_int()));
  geometry_thresholds_.minimum_valid_depth_ratio = std::clamp(
    this->get_parameter("geometry_minimum_valid_depth_ratio").as_double(), 0.0, 1.0);
  geometry_thresholds_.minimum_cloud_points = static_cast<size_t>(std::max<int64_t>(
      1, this->get_parameter("geometry_minimum_cloud_points").as_int()));
  EPD::TrackerThresholds tracker_thresholds;
  tracker_thresholds.minimum_iou = std::clamp(
    this->get_parameter("tracking_minimum_iou").as_double(), 0.0, 1.0);
  tracker_thresholds.maximum_roi_centroid_distance_px = std::max(
    0.0, this->get_parameter("tracking_maximum_roi_distance_px").as_double());
  tracker_thresholds.maximum_3d_distance_m = std::max(
    0.0, this->get_parameter("tracking_maximum_3d_distance_m").as_double());
  tracker_thresholds.confirmation_hits = static_cast<uint32_t>(std::max<int64_t>(
      1, this->get_parameter("tracking_confirmation_hits").as_int()));
  tracker_thresholds.maximum_missed_observations = static_cast<uint32_t>(std::max<int64_t>(
      0, this->get_parameter("tracking_maximum_missed_observations").as_int()));
  tracker_thresholds.maximum_active_tracks = static_cast<size_t>(std::max<int64_t>(
      1, this->get_parameter("tracking_maximum_active_tracks").as_int()));
  temporal_tracker_ = std::make_unique<EPD::TemporalTracker>(tracker_thresholds);
  if (slow_frame_warn_ms_ < 0) {
    RCLCPP_WARN(
      this->get_logger(),
      "Invalid slow_frame_warn_ms '%d'. Falling back to 1000.",
      slow_frame_warn_ms_);
    slow_frame_warn_ms_ = 1000;
  }
  if (max_processing_fps_ < 0.0) {
    RCLCPP_WARN(
      this->get_logger(),
      "Invalid max_processing_fps '%.3f'. Falling back to 0.0 (disabled).",
      max_processing_fps_);
    max_processing_fps_ = 0.0;
  }
  std::transform(
    image_output_qos_reliability.begin(),
    image_output_qos_reliability.end(),
    image_output_qos_reliability.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  if (image_output_qos_reliability != "best_effort" && image_output_qos_reliability != "reliable") {
    RCLCPP_WARN(
      this->get_logger(),
      "Invalid image_output_qos_reliability '%s'. Falling back to 'best_effort'.",
      image_output_qos_reliability.c_str());
    image_output_qos_reliability = "best_effort";
  }
  if (image_output_qos_depth < 1) {
    RCLCPP_WARN(
      this->get_logger(),
      "Invalid image_output_qos_depth '%d'. Falling back to 1.",
      image_output_qos_depth);
    image_output_qos_depth = 1;
  }
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
  last_rgb_frame_time_ = this->now();
  last_synchronized_observation_time_ = this->now();
  last_processed_frame_time_ = this->now();
  last_drop_stats_log_time_ = this->now();
  metrics_start_time_ = this->now();

  if (ortAgent_.publish_detection_segmentation &&
    ortAgent_.useCaseMode <= EPD::COLOR_MATCHING_MODE)
  {
    enableDetectionInputs();
  }

  subscribeImageInput();
  if (rgb_input_watchdog_timeout_s_ > 0.0) {
    rgb_input_watchdog_timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&EasyPerceptionDeployment::rgb_input_watchdog_callback, this),
      sensor_callback_group_);
  } else {
    RCLCPP_INFO(
      this->get_logger(),
      "RGB input watchdog disabled because rgb_input_watchdog_timeout_s is %.3f.",
      rgb_input_watchdog_timeout_s_);
  }

  // Creating Publisher to output Visualizable P2 and P3 Detection Results.
  // By default use BEST_EFFORT QoS with depth 1 so the latest annotated frame
  // is forwarded immediately; old frames are dropped rather than queued.
  // This prevents backpressure from stalling inference when the display
  // consumer is slower than the camera rate. Users can override reliability
  // and depth through ROS parameters for visualization tools that require
  // RELIABLE delivery.
  auto image_output_qos = rclcpp::QoS(rclcpp::KeepLast(static_cast<size_t>(image_output_qos_depth)));
  if (image_output_qos_reliability == "reliable") {
    image_output_qos.reliable();
  } else {
    image_output_qos.best_effort();
  }
  image_output_qos.durability_volatile();
  RCLCPP_INFO(
    this->get_logger(),
    "Image output QoS selected: reliability='%s', depth=%d, durability='volatile'.",
    image_output_qos_reliability.c_str(),
    image_output_qos_depth);
  visual_pub = image_transport::create_publisher(
    this,
    "/easy_perception_deployment/image_output",
    image_output_qos.get_rmw_qos_profile());

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
    rclcpp::SensorDataQoS(),
    publisher_options);

  // Creating Publisher to output Action P3 and Tracking Detection Results.
  tracking_pub = this->create_publisher<epd_msgs::msg::EPDObjectTracking>(
    "/easy_perception_deployment/epd_tracking_output",
    rclcpp::SensorDataQoS(),
    publisher_options);

  // Creating Publisher to output 3D poses of localized/tracked objects.
  pose_pub = this->create_publisher<geometry_msgs::msg::PoseArray>(
    "/easy_perception_deployment/epd_pose_output",
    10,
    publisher_options);
  inference_diagnostics_pub_ =
    this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "/easy_perception_deployment/inference_diagnostics", rclcpp::QoS(1));
  inference_diagnostics_timer_ = this->create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&EasyPerceptionDeployment::inference_diagnostics_callback, this),
    sensor_callback_group_);

  // If useCaseMode is detected to be Localization or Tracking,
  // Subscribe to all synchronized ROS2 topics.
  if (ortAgent_.useCaseMode == EPD::LOCALISATION_MODE ||
    ortAgent_.useCaseMode == EPD::TRACKING_MODE)
  {
    enableLocalizeInputs(ortAgent_.useCaseMode);
  } else {
    disableLocalizeInputs();
  }

  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    if (ortAgent_.isService()) {
      ortAgent_.requestAddressed = true;
    }
  }

  auto handle_emd_request =
    [this](
    const std::shared_ptr<epd_msgs::srv::Perception::Request> request,
    std::shared_ptr<epd_msgs::srv::Perception::Response> response) -> void
    {
      std::unique_lock<std::mutex> request_lock(service_request_mutex_);
      service_requests_.fetch_add(1);
      RCLCPP_INFO(this->get_logger(), "[ RECEIVED ] - EMD Grasp-Planner Request");

      if (!request->trigger) {
        RCLCPP_WARN(this->get_logger(), "Service request ignored: trigger=false");
        response->success = false;
        std::lock_guard<std::mutex> ort_guard(ort_mutex_);
        response->tracking_enabled = (ortAgent_.useCaseMode == EPD::TRACKING_MODE);
        return;
      }

      {
        std::lock_guard<std::mutex> ort_guard(ort_mutex_);
        response->tracking_enabled = (ortAgent_.useCaseMode == EPD::TRACKING_MODE);
      }

      unsigned int use_case_mode = 0;
      {
        std::lock_guard<std::mutex> ort_guard(ort_mutex_);
        use_case_mode = ortAgent_.useCaseMode;
      }

      if (use_case_mode == EPD::LOCALISATION_MODE ||
        use_case_mode == EPD::TRACKING_MODE)
      {
        // A request consumes a fresh generation from the already-active
        // pipeline.  Only repair the lifecycle here if the mode really changed
        // or the inputs are unexpectedly inactive.
        if ((!use_depth_ && (!localize_nodepth_active_ ||
          sync_callback_mode_ != static_cast<int>(use_case_mode))) ||
          (use_depth_ && (!localize_input_active_ ||
          sync_callback_mode_ != static_cast<int>(use_case_mode))))
        {
          enableLocalizeInputs(use_case_mode);
        }
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

      if (use_case_mode == EPD::LOCALISATION_MODE ||
        use_case_mode == EPD::TRACKING_MODE)
      {
        const uint64_t request_baseline = observation_store_.latest_id();
        service_baseline_observation_id_.store(request_baseline);
        {
          std::lock_guard<std::mutex> data_guard(data_mutex_);
          service_request_baseline_ = request_baseline;
          service_request_active_ = true;
        }
        RCLCPP_INFO(
          this->get_logger(),
          "Waiting for observation_id newer than %llu",
          static_cast<unsigned long long>(request_baseline));
      }

      if (!(use_case_mode == EPD::LOCALISATION_MODE ||
        use_case_mode == EPD::TRACKING_MODE))
      {
        response->success = true;
        return;
      }

      const double timeout_s = this->get_parameter("service_timeout_s").as_double();
      const auto service_timeout = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(timeout_s));
      RCLCPP_INFO(this->get_logger(), "Waiting for EPD service result...");

      uint64_t request_baseline = 0;
      {
        std::lock_guard<std::mutex> data_guard(data_mutex_);
        request_baseline = service_request_baseline_;
      }

      const auto result = perception_result_store_.wait_for_result_after(
        request_baseline, service_timeout);

      if (!result) {
        {
          std::lock_guard<std::mutex> data_guard(data_mutex_);
          service_request_active_ = false;
        }
        if (perception_result_store_.metrics().shutdown) {
          service_shutdown_abort_.fetch_add(1);
        } else {
          service_timeout_.fetch_add(1);
        }
        response->success = false;
        response->epd_localization = epd_msgs::msg::EPDObjectLocalization();
        response->epd_tracking = epd_msgs::msg::EPDObjectTracking();
        RCLCPP_WARN(
          this->get_logger(),
          "Timed out waiting for fresh synchronized localization observation");
        return;
      }

      response->success = result->success;
      response->epd_localization = result->localization;
      response->epd_tracking = result->tracking;
      service_success_.fetch_add(result->success ? 1 : 0);
      last_service_result_observation_id_.store(result->source_observation_id);
      RCLCPP_INFO(
        this->get_logger(),
        "Accepted fresh observation_id %llu (baseline %llu)",
        static_cast<unsigned long long>(result->source_observation_id),
        static_cast<unsigned long long>(request_baseline));
      {
        std::lock_guard<std::mutex> data_guard(data_mutex_);
        service_request_active_ = false;
      }
    };

  srv_ = this->create_service<epd_msgs::srv::Perception>(
    "epd_perception_service",
    handle_emd_request,
    rmw_qos_profile_services_default,
    service_callback_group_);

  // Log all session_config and usecase_config configurations for user to check on system boot
  RCLCPP_INFO(this->get_logger(), "[-ONNX Model-] - %s", ortAgent_.onnx_model_path.c_str());
  RCLCPP_INFO(this->get_logger(), "[-Label List-] - %s", ortAgent_.class_label_path.c_str());
  RCLCPP_INFO(this->get_logger(), "[-Precision Level-] - %d", ortAgent_.precision_level);
  RCLCPP_INFO(this->get_logger(), "[-Image Transport-] - %s", image_transport_.c_str());
  RCLCPP_INFO(
    this->get_logger(), "[-Confidence Threshold-] - %.2f", ortAgent_.confidence_threshold);
  RCLCPP_INFO(this->get_logger(), "[-Max Detections-] - %d", ortAgent_.max_detections);

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
  observation_store_.shutdown();
  inference_scheduler_.shutdown();
  perception_result_store_.shutdown();
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
  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = sensor_callback_group_;
  image_sub = image_transport::create_subscription(
    this,
    "/easy_perception_deployment/image_input",
    std::bind(&EasyPerceptionDeployment::image_callback, this, std::placeholders::_1),
    image_transport_,
    sensor_qos_profile_,
    sub_options);
  image_input_active_ = true;
}

void EasyPerceptionDeployment::pruneLocalizeSyncCaches()
{
  auto prune = [](auto & cache) {
      while (cache.size() > kLocalizeSyncCacheSize) {
        cache.erase(cache.begin());
      }
    };

  prune(localize_rgb_cache_);
  prune(localize_depth_cache_);
  prune(localize_info_cache_);
}

void EasyPerceptionDeployment::tryDispatchLocalizedTriplet(const int64_t key)
{
  sensor_msgs::msg::Image::SharedPtr rgb;
  sensor_msgs::msg::Image::SharedPtr depth;
  sensor_msgs::msg::CameraInfo::SharedPtr info;

  {
    std::lock_guard<std::mutex> lock(localize_sync_mutex_);

    const auto rgb_it = localize_rgb_cache_.find(key);
    const auto depth_it = localize_depth_cache_.find(key);
    const auto info_it = localize_info_cache_.find(key);

    if (rgb_it == localize_rgb_cache_.end() ||
      depth_it == localize_depth_cache_.end() ||
      info_it == localize_info_cache_.end())
    {
      pruneLocalizeSyncCaches();
      return;
    }

    rgb = rgb_it->second;
    depth = depth_it->second;
    info = info_it->second;
    ++localize_triplet_dispatch_count_;

    localize_rgb_cache_.erase(localize_rgb_cache_.begin(), std::next(rgb_it));
    localize_depth_cache_.erase(localize_depth_cache_.begin(), std::next(depth_it));
    localize_info_cache_.erase(localize_info_cache_.begin(), std::next(info_it));
  }

  if (sync_callback_mode_ == static_cast<int>(EPD::LOCALISATION_MODE)) {
    localize_callback(rgb, depth, info);
  } else if (sync_callback_mode_ == static_cast<int>(EPD::TRACKING_MODE)) {
    tracking_callback(rgb, depth, info);
  }

}

void EasyPerceptionDeployment::subscribeLocalizeInputs()
{
  const auto qos = rclcpp::SensorDataQoS().keep_last(10);

  rclcpp::SubscriptionOptions options;
  options.callback_group = localize_input_callback_group_;

  localize_image_rgb_sub_ =
    this->create_subscription<sensor_msgs::msg::Image>(
    rgb_topic_,
    qos,
    [this](sensor_msgs::msg::Image::SharedPtr msg) {
      const int64_t key = stampKey(msg->header.stamp);
      {
        std::lock_guard<std::mutex> lock(localize_sync_mutex_);
        ++localize_rgb_callback_count_;
        localize_last_rgb_key_ = key;
        localize_rgb_cache_[key] = msg;
        pruneLocalizeSyncCaches();
      }
      tryDispatchLocalizedTriplet(key);
    },
    options);

  localize_image_depth_sub_ =
    this->create_subscription<sensor_msgs::msg::Image>(
    depth_topic_,
    qos,
    [this](sensor_msgs::msg::Image::SharedPtr msg) {
      const int64_t key = stampKey(msg->header.stamp);
      {
        std::lock_guard<std::mutex> lock(localize_sync_mutex_);
        ++localize_depth_callback_count_;
        localize_last_depth_key_ = key;
        localize_depth_cache_[key] = msg;
        pruneLocalizeSyncCaches();
      }
      tryDispatchLocalizedTriplet(key);
    },
    options);

  localize_cam_info_sub_ =
    this->create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic_,
    qos,
    [this](sensor_msgs::msg::CameraInfo::SharedPtr msg) {
      const int64_t key = stampKey(msg->header.stamp);
      {
        std::lock_guard<std::mutex> lock(localize_sync_mutex_);
        ++localize_info_callback_count_;
        localize_last_info_key_ = key;
        localize_info_cache_[key] = msg;
        pruneLocalizeSyncCaches();
      }
      tryDispatchLocalizedTriplet(key);
    },
    options);
}

void EasyPerceptionDeployment::subscribeLocalizeNoDepth(const unsigned int)
{
  // Shut down any previous no-depth subscription before re-subscribing.
  localize_rgb_nodepth_.shutdown();
  camera_info_sub_.reset();

  rclcpp::SubscriptionOptions cam_info_options;
  cam_info_options.callback_group = sensor_callback_group_;
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic_,
    rclcpp::SensorDataQoS().keep_last(1),
    std::bind(&EasyPerceptionDeployment::camera_info_callback, this, std::placeholders::_1),
    cam_info_options);

  RCLCPP_WARN(
    this->get_logger(),
    "use_depth=false: running without depth. "
    "3D coordinates will be unavailable; only 2D detection results are valid.");

  rclcpp::SubscriptionOptions rgb_nodepth_options;
  rgb_nodepth_options.callback_group = sensor_callback_group_;
  localize_rgb_nodepth_ = image_transport::create_subscription(
    this,
    rgb_topic_,
    [this](const sensor_msgs::msg::Image::ConstSharedPtr & msg) {
      bool has_cam = false;
      {
        std::lock_guard<std::mutex> lk(data_mutex_);
        if (latest_camera_info_) {
          EPD::SynchronizationMetadata metadata;
          metadata.synchronized = false;
          metadata.source_healthy = true;
          const auto observation = observation_store_.publish(
            camera_id_, msg, nullptr, latest_camera_info_, metadata);
          // Perception is continuously owned by EPD.  Service requests wait on
          // the completed-result timeline; they never start or stop inference.
          inference_scheduler_.submit(observation);
          has_cam = true;
        }
      }
      if (!has_cam) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Waiting for camera_info on '%s'...", camera_info_topic_.c_str());
        return;
      }
    },
    image_transport_,
    sensor_qos_profile_,
    rgb_nodepth_options);
}

void EasyPerceptionDeployment::subscribeDetectionDepthInputs()
{
  if (depth_input_active_) {
    return;
  }

  rclcpp::SubscriptionOptions depth_sub_options;
  depth_sub_options.callback_group = sensor_callback_group_;
  depth_sub_ = image_transport::create_subscription(
    this,
    depth_topic_,
    std::bind(&EasyPerceptionDeployment::depth_callback, this, std::placeholders::_1),
    depth_transport_,
    sensor_qos_profile_,
    depth_sub_options);

  rclcpp::SubscriptionOptions cam_info_options;
  cam_info_options.callback_group = sensor_callback_group_;
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic_,
    rclcpp::SensorDataQoS().keep_last(1),
    std::bind(&EasyPerceptionDeployment::camera_info_callback, this, std::placeholders::_1),
    cam_info_options);
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

void EasyPerceptionDeployment::enableLocalizeInputs(const unsigned int use_case_mode)
{
  disableDetectionInputs();

  if (image_input_active_) {
    image_sub.shutdown();
    image_input_active_ = false;
  }

  if (!use_depth_) {
    // No-depth path: subscribe RGB + camera_info only; no synchronizer needed.
    if (!localize_nodepth_active_ || sync_callback_mode_ != static_cast<int>(use_case_mode)) {
      subscribeLocalizeNoDepth(use_case_mode);
      localize_nodepth_active_ = true;
      sync_callback_mode_ = static_cast<int>(use_case_mode);
    }
    return;
  }

  if (!localize_input_active_) {
    sync_callback_mode_ = static_cast<int>(use_case_mode);
    subscribeLocalizeInputs();
    localize_input_active_ = true;
    return;
  }

  // Subscribers remain active when switching between localization/tracking.
  // Only the downstream dispatch mode changes.
  sync_callback_mode_ = static_cast<int>(use_case_mode);
}

void EasyPerceptionDeployment::disableLocalizeInputs()
{
  sync_callback_mode_ = -1;

  if (localize_nodepth_active_) {
    localize_rgb_nodepth_.shutdown();
    camera_info_sub_.reset();
    localize_nodepth_active_ = false;
  }

  if (!localize_input_active_) {
    return;
  }

  localize_image_rgb_sub_.reset();
  localize_image_depth_sub_.reset();
  localize_cam_info_sub_.reset();

  {
    std::lock_guard<std::mutex> lock(localize_sync_mutex_);
    localize_rgb_cache_.clear();
    localize_depth_cache_.clear();
    localize_info_cache_.clear();
  }

  localize_input_active_ = false;
}

void EasyPerceptionDeployment::hasCameraChanged(const int img_height, const int img_width) const
{
  if (ortAgent_.getWidth() != img_width || ortAgent_.getHeight() != img_height) {
    RCLCPP_FATAL(
      this->get_logger(),
      "Input camera resolution changed (%dx%d → %dx%d). "
      "Requesting graceful shutdown — please restart the node.",
      ortAgent_.getWidth(), ortAgent_.getHeight(), img_width, img_height);
    rclcpp::shutdown();
    throw std::runtime_error("Input camera changed. Please restart.");
  }
}

void EasyPerceptionDeployment::ensureOrtAgentInitialized(
  const int img_height,
  const int img_width)
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
  uint64_t previous_observation_id = observation_store_.latest_id();
  uint64_t current_observation_id = 0;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    EPD::SynchronizationMetadata metadata;
    metadata.synchronized = true;
    metadata.exact_sensor_stamp = true;
    metadata.maximum_skew_ns = 0;
    metadata.source_healthy = true;
    const auto observation = observation_store_.publish(
      camera_id_, msg, depth_msg, camera_info, metadata);
    current_observation_id = observation->observation_id();
    last_synchronized_observation_time_ = this->now();
    has_received_synchronized_observation_ = true;
    // The service is a fresh snapshot consumer, not the inference scheduler.
    inference_scheduler_.submit(observation);
  }
  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 10000,
    "Observation advanced %llu -> %llu (RGB stamp %d.%09u)",
    static_cast<unsigned long long>(previous_observation_id),
    static_cast<unsigned long long>(current_observation_id),
    msg->header.stamp.sec, msg->header.stamp.nanosec);
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
  uint64_t previous_observation_id = observation_store_.latest_id();
  uint64_t current_observation_id = 0;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    EPD::SynchronizationMetadata metadata;
    metadata.synchronized = true;
    metadata.exact_sensor_stamp = true;
    metadata.maximum_skew_ns = 0;
    metadata.source_healthy = true;
    const auto observation = observation_store_.publish(
      camera_id_, msg, depth_msg, camera_info, metadata);
    current_observation_id = observation->observation_id();
    last_synchronized_observation_time_ = this->now();
    has_received_synchronized_observation_ = true;
    // Tracking also remains live independently of EMD service activity.
    inference_scheduler_.submit(observation);
  }
  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 10000,
    "Observation advanced %llu -> %llu (RGB stamp %d.%09u)",
    static_cast<unsigned long long>(previous_observation_id),
    static_cast<unsigned long long>(current_observation_id),
    msg->header.stamp.sec, msg->header.stamp.nanosec);
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
  bool should_log_drop_stats = false;
  uint64_t overwritten_delta = 0;
  uint64_t rate_limited_delta = 0;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    const rclcpp::Time now = this->now();
    if (max_processing_fps_ > 0.0 && has_processed_frame_) {
      const double min_interval_s = 1.0 / max_processing_fps_;
      const double elapsed_s = (now - last_processed_frame_time_).seconds();
      if (elapsed_s < min_interval_s) {
        ++dropped_frames_rate_limited_;
        should_log_drop_stats =
          (now - last_drop_stats_log_time_).seconds() >= dropped_frame_log_period_s_;
        if (should_log_drop_stats) {
          overwritten_delta = dropped_frames_overwritten_ - logged_dropped_frames_overwritten_;
          rate_limited_delta =
            dropped_frames_rate_limited_ - logged_dropped_frames_rate_limited_;
          logged_dropped_frames_overwritten_ = dropped_frames_overwritten_;
          logged_dropped_frames_rate_limited_ = dropped_frames_rate_limited_;
          last_drop_stats_log_time_ = now;
        }
        return;
      }
    }
    if (image_pending_ && latest_image_) {
      ++dropped_frames_overwritten_;
    }
    latest_image_ = msg;
    image_pending_ = true;
    should_log_drop_stats =
      (now - last_drop_stats_log_time_).seconds() >= dropped_frame_log_period_s_;
    if (should_log_drop_stats) {
      overwritten_delta = dropped_frames_overwritten_ - logged_dropped_frames_overwritten_;
      rate_limited_delta = dropped_frames_rate_limited_ - logged_dropped_frames_rate_limited_;
      logged_dropped_frames_overwritten_ = dropped_frames_overwritten_;
      logged_dropped_frames_rate_limited_ = dropped_frames_rate_limited_;
      last_drop_stats_log_time_ = now;
    }
  }

  if (should_log_drop_stats && (overwritten_delta > 0 || rate_limited_delta > 0)) {
    RCLCPP_WARN(
      this->get_logger(),
      "Dropped frames in last %.1f s: overwritten_latest=%llu, rate_limited=%llu "
      "(totals: overwritten=%llu, rate_limited=%llu)",
      dropped_frame_log_period_s_,
      static_cast<unsigned long long>(overwritten_delta),
      static_cast<unsigned long long>(rate_limited_delta),
      static_cast<unsigned long long>(logged_dropped_frames_overwritten_),
      static_cast<unsigned long long>(logged_dropped_frames_rate_limited_));
  }
  data_cv_.notify_one();
}

void EasyPerceptionDeployment::image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    last_rgb_frame_time_ = this->now();
    has_received_rgb_frame_ = true;
  }
  this->process_image_callback(msg);
}

void EasyPerceptionDeployment::inference_diagnostics_callback()
{
  const auto metrics = inference_scheduler_.metrics();
  const auto now = this->now();
  const double elapsed_s = std::max(0.001, (now - metrics_start_time_).seconds());
  const auto stamp_age_ms = [&now](const builtin_interfaces::msg::Time & stamp) {
      return EPD::sensorStampAgeMilliseconds(now, stamp);
    };

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  status.name = "easy_perception_deployment/inference_worker";
  status.hardware_id = camera_id_;
  status.message = metrics.inference_failed == 0 ? "latest-only worker healthy" :
    "worker recovered from inference failure";
  const auto add = [&status](const std::string & key, const auto & value) {
      diagnostic_msgs::msg::KeyValue item;
      item.key = key;
      item.value = std::to_string(value);
      status.values.push_back(std::move(item));
    };
  add("latest_observation_id", observation_store_.latest_id());
  add("observations_published", metrics.observations_published);
  add("observation_rate_hz", metrics.observations_published / elapsed_s);
  add("inference_started", metrics.inference_started);
  add("inference_completed", metrics.inference_completed);
  add("inference_failed", metrics.inference_failed);
  add("last_consumed_observation_id", metrics.last_consumed_observation_id);
  add("last_completed_observation_id", metrics.last_completed_observation_id);
  add("observations_skipped_before_inference", metrics.observations_skipped_before_inference);
  add("inference_latency_ms", metrics.last_latency_ms);
  const uint64_t finished = metrics.inference_completed + metrics.inference_failed;
  add("inference_latency_min_ms", metrics.minimum_latency_ms);
  add("inference_latency_avg_ms", finished == 0 ? 0.0 :
    static_cast<double>(metrics.total_latency_ms) / static_cast<double>(finished));
  add("inference_latency_max_ms", metrics.maximum_latency_ms);
  add("inference_rate_hz", metrics.inference_completed / elapsed_s);
  add("newest_observation_age_ms", stamp_age_ms(metrics.newest_observation_stamp));
  add("newest_result_age_ms", stamp_age_ms(metrics.newest_result_stamp));
  add("worker_busy", metrics.worker_busy ? 1 : 0);
  add("backlog_size", metrics.backlog_size);
  add("backlog_high_water_mark", metrics.backlog_high_water_mark);
  add("duplicate_or_regressed_submissions", metrics.duplicate_or_regressed_submissions);
  add("duplicate_processing_attempts", metrics.duplicate_processing_attempts);
  const auto result_store_metrics = perception_result_store_.metrics();
  const auto latest_result = perception_result_store_.latest();
  add("latest_completed_result_observation_id",
    result_store_metrics.latest_completed_result_observation_id);
  add("results_completed", result_store_metrics.results_completed);
  add("service_requests", service_requests_.load());
  add("service_success", service_success_.load());
  add("service_timeout", service_timeout_.load());
  add("service_shutdown_abort", service_shutdown_abort_.load());
  add("service_baseline_observation_id", service_baseline_observation_id_.load());
  add("last_service_result_observation_id", last_service_result_observation_id_.load());
  add("result_age_ms", latest_result ? stamp_age_ms(latest_result->sensor_stamp) : -1.0);
  add("result_store_regressions", result_store_metrics.result_store_regressions);
  add("duplicate_result_publish", result_store_metrics.duplicate_result_publish);
  add("current_waiters", result_store_metrics.current_waiters);
  add("detections_total", detections_total_.load());
  add("geometry_valid_total", geometry_valid_total_.load());
  add("geometry_degraded_total", geometry_degraded_total_.load());
  add("geometry_invalid_total", geometry_invalid_total_.load());
  add("invalid_intrinsics_total", invalid_intrinsics_total_.load());
  add("empty_mask_total", invalid_mask_total_.load());
  add("insufficient_depth_total", insufficient_depth_total_.load());
  add("empty_cloud_total", empty_cloud_total_.load());
  add("nonfinite_geometry_total", nonfinite_geometry_total_.load());
  const int64_t valid_geometry_stamp_ns = latest_valid_geometry_stamp_ns_.load();
  add("latest_valid_geometry_age_ms", valid_geometry_stamp_ns == 0 ? -1.0 :
    EPD::safeAgeMilliseconds(
      now, rclcpp::Time(valid_geometry_stamp_ns, now.get_clock_type())));
  if (temporal_tracker_) {
    EPD::TrackerMetrics tracking;
    {
      std::lock_guard<std::mutex> tracker_guard(temporal_tracker_mutex_);
      tracking = temporal_tracker_->metrics();
    }
    add("tracks_created", tracking.tracks_created);
    add("tracks_confirmed", tracking.tracks_confirmed);
    std::ostringstream confirmed_ids;
    for (size_t index = 0; index < tracking.confirmed_track_ids.size(); ++index) {
      if (index != 0) {
        confirmed_ids << ',';
      }
      confirmed_ids << tracking.confirmed_track_ids[index];
    }
    diagnostic_msgs::msg::KeyValue confirmed_ids_item;
    confirmed_ids_item.key = "confirmed_track_ids";
    confirmed_ids_item.value = confirmed_ids.str();
    status.values.push_back(std::move(confirmed_ids_item));
    add("tracks_lost", tracking.tracks_lost);
    add("tracks_expired", tracking.tracks_expired);
    add("active_tracks", tracking.active_tracks);
    add("max_active_tracks", tracking.max_active_tracks);
    add("associations_matched", tracking.associations_matched);
    add("associations_rejected", tracking.associations_rejected);
    add("id_switches", tracking.id_switches);
    add("tracker_duplicate_update_attempts", tracking.duplicate_update_attempts);
    add("tracker_out_of_order_observations", tracking.out_of_order_observations);
    add("tracker_processing_latency_us", tracking.last_processing_latency_us);
    add("latest_track_observation_id", tracking.latest_track_observation_id);
    add("latest_track_observation_age_ms",
      EPD::sensorStampAgeMilliseconds(now, tracking.latest_track_stamp));
    add("geometry_valid_tracks", tracking.geometry_valid_tracks);
    add("two_d_only_tracks", tracking.two_d_only_tracks);
  }

  diagnostic_msgs::msg::DiagnosticArray message;
  message.header.stamp = now;
  message.status.push_back(std::move(status));
  inference_diagnostics_pub_->publish(message);

  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 10000,
    "Inference health: observations=%llu latest=%llu started=%llu completed=%llu "
    "failed=%llu skipped=%llu backlog=%zu busy=%s latency_ms=%lld",
    static_cast<unsigned long long>(metrics.observations_published),
    static_cast<unsigned long long>(metrics.last_submitted_observation_id),
    static_cast<unsigned long long>(metrics.inference_started),
    static_cast<unsigned long long>(metrics.inference_completed),
    static_cast<unsigned long long>(metrics.inference_failed),
    static_cast<unsigned long long>(metrics.observations_skipped_before_inference),
    metrics.backlog_size, metrics.worker_busy ? "true" : "false",
    static_cast<long long>(metrics.last_latency_ms));
}

void EasyPerceptionDeployment::rgb_input_watchdog_callback()
{
  if (rgb_input_watchdog_timeout_s_ <= 0.0) {
    return;
  }

  const rclcpp::Time now = this->now();
  if (localize_input_active_ || localize_nodepth_active_) {
    bool has_synchronized_observation = false;
    rclcpp::Time last_synchronized_observation_time = now;
    uint64_t observation_id = 0;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      has_synchronized_observation = has_received_synchronized_observation_;
      last_synchronized_observation_time = last_synchronized_observation_time_;
      observation_id = observation_store_.latest_id();
    }
    const double stalled_for_s = (now - last_synchronized_observation_time).seconds();
    if (!has_synchronized_observation || stalled_for_s > rgb_input_watchdog_timeout_s_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Synchronized observation has not advanced for %.1f seconds while localization "
        "inputs are active (generation %llu).",
        stalled_for_s, static_cast<unsigned long long>(observation_id));

      uint64_t rgb_count = 0;
      uint64_t depth_count = 0;
      uint64_t info_count = 0;
      uint64_t dispatch_count = 0;
      int64_t rgb_key = 0;
      int64_t depth_key = 0;
      int64_t info_key = 0;
      std::size_t rgb_cache_size = 0;
      std::size_t depth_cache_size = 0;
      std::size_t info_cache_size = 0;

      {
        std::lock_guard<std::mutex> sync_lock(localize_sync_mutex_);
        rgb_count = localize_rgb_callback_count_;
        depth_count = localize_depth_callback_count_;
        info_count = localize_info_callback_count_;
        dispatch_count = localize_triplet_dispatch_count_;
        rgb_key = localize_last_rgb_key_;
        depth_key = localize_last_depth_key_;
        info_key = localize_last_info_key_;
        rgb_cache_size = localize_rgb_cache_.size();
        depth_cache_size = localize_depth_cache_.size();
        info_cache_size = localize_info_cache_.size();
      }

      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Exact-sync diagnostics: callbacks rgb=%llu depth=%llu info=%llu; "
        "dispatched=%llu; cache rgb=%zu depth=%zu info=%zu; "
        "last_keys rgb=%lld depth=%lld info=%lld",
        static_cast<unsigned long long>(rgb_count),
        static_cast<unsigned long long>(depth_count),
        static_cast<unsigned long long>(info_count),
        static_cast<unsigned long long>(dispatch_count),
        rgb_cache_size,
        depth_cache_size,
        info_cache_size,
        static_cast<long long>(rgb_key),
        static_cast<long long>(depth_key),
        static_cast<long long>(info_key));

    }
    return;
  }

  if (!image_input_active_) {
    return;
  }

  bool has_received_rgb_frame = false;
  rclcpp::Time last_rgb_frame_time = now;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    has_received_rgb_frame = has_received_rgb_frame_;
    last_rgb_frame_time = last_rgb_frame_time_;
  }

  if (!has_received_rgb_frame) {
    rgb_stream_missing_ = true;
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      5000,
      "No RGB frame received yet on topic '%s' (transport '%s'). "
      "Watchdog timeout is %.1f seconds.",
      rgb_topic_.c_str(),
      image_transport_.c_str(),
      rgb_input_watchdog_timeout_s_);
    return;
  }

  const double missing_for_s = (now - last_rgb_frame_time).seconds();
  if (missing_for_s > rgb_input_watchdog_timeout_s_) {
    rgb_stream_missing_ = true;
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      5000,
      "No RGB frame received on topic '%s' (transport '%s') for %.1f seconds "
      "(watchdog timeout %.1f seconds).",
      rgb_topic_.c_str(),
      image_transport_.c_str(),
      missing_for_s,
      rgb_input_watchdog_timeout_s_);
    return;
  }

  if (rgb_stream_missing_) {
    rgb_stream_missing_ = false;
    RCLCPP_INFO_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      5000,
      "RGB frames resumed on topic '%s' (transport '%s').",
      rgb_topic_.c_str(),
      image_transport_.c_str());
  }
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
  const EPD::Observation::ConstSharedPtr & observation)
{
  if (!observation) {return;}
  const auto & msg = observation->rgb();
  const auto & depth_msg = observation->aligned_depth();
  const auto & camera_info = observation->camera_info();
  std::unique_lock<std::mutex> ort_lock(ort_mutex_);
  if (ortAgent_.requestAddressed && !service_mode_) {
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

  cv::Mat depth_img;
  if (depth_msg) {
    cv_bridge::CvImageConstPtr depth_imageptr;
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      depth_imageptr = cv_bridge::toCvShare(depth_msg);
    } else {
      depth_imageptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    }
    depth_img = depth_imageptr->image;
  } else {
    depth_img = cv::Mat::zeros(img.rows, img.cols, CV_16UC1);
  }

  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    ensureOrtAgentInitialized(img.rows, img.cols);
  }

  auto begin = std::chrono::high_resolution_clock::now();

  EPD::EPDObjectLocalization result = [&]() {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    return ortAgent_.p3_ort_session->infer(
      img,
      depth_img,
      *camera_info,
      camera_to_plane_distance_mm,
      ortAgent_.confidence_threshold);
  }();

  // Apply max_detections limit.
  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    const int max_det = ortAgent_.max_detections;
    if (max_det > 0 && static_cast<int>(result.size()) > max_det) {
      result.objects.resize(max_det);
    }
  }

  detections_total_.fetch_add(result.size());
  std::vector<EPD::LocalizedObject> valid_objects;
  valid_objects.reserve(result.size());
  for (auto & object : result.objects) {
    object.source_observation_id = observation->observation_id();
    object.source_sensor_stamp = observation->sensor_stamp();
    object.source_frame = observation->frame_id();
    const auto quality = EPD::validateLocalizedObject(
      object, static_cast<uint32_t>(img.cols), static_cast<uint32_t>(img.rows),
      camera_info->k.at(0), camera_info->k.at(4), camera_info->k.at(2), camera_info->k.at(5),
      geometry_thresholds_);
    if (quality == EPD::GeometryQuality::VALID) {
      geometry_valid_total_.fetch_add(1);
      latest_valid_geometry_stamp_ns_.store(rclcpp::Time(observation->sensor_stamp()).nanoseconds());
      valid_objects.push_back(std::move(object));
      continue;
    }
    if (quality == EPD::GeometryQuality::DEGRADED) {
      geometry_degraded_total_.fetch_add(1);
    } else {
      geometry_invalid_total_.fetch_add(1);
    }
    const uint32_t reasons = object.failure_reasons;
    if (reasons & EPD::reason(EPD::GeometryFailure::INVALID_INTRINSICS)) {
      invalid_intrinsics_total_.fetch_add(1);
    }
    if (reasons & EPD::reason(EPD::GeometryFailure::INVALID_MASK)) {
      invalid_mask_total_.fetch_add(1);
    }
    if (reasons & EPD::reason(EPD::GeometryFailure::INSUFFICIENT_DEPTH)) {
      insufficient_depth_total_.fetch_add(1);
    }
    if (reasons & EPD::reason(EPD::GeometryFailure::EMPTY_CLOUD)) {
      empty_cloud_total_.fetch_add(1);
    }
    if (reasons & EPD::reason(EPD::GeometryFailure::NONFINITE_GEOMETRY)) {
      nonfinite_geometry_total_.fetch_add(1);
    }
  }
  result.objects = std::move(valid_objects);

  cv::Mat resultImg;

  bool visualize = false;
  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    visualize = ortAgent_.isVisualize();
  }

  if (visualize) {
    EPD::EPDObjectTracking converted_result(result.size());
    converted_result.object_ids.clear();
    for (size_t i = 0; i < result.size(); i++) {
      converted_result.objects.emplace_back(result.objects[i]);
    }

    {
      std::lock_guard<std::mutex> ort_guard(ort_mutex_);
      resultImg = ortAgent_.visualize(converted_result, img);
    }

    sensor_msgs::msg::Image::SharedPtr output_msg =
      cv_bridge::CvImage(msg->header, "bgr8", resultImg).toImageMsg();
    visual_pub.publish(*output_msg);
  }

  epd_msgs::msg::EPDObjectLocalization output_msg;

  EPD::preserve_observation_header(output_msg, *observation);
  output_msg.frame_width = img.cols;
  output_msg.frame_height = img.rows;
  if (depth_msg) {
    output_msg.depth_image = *depth_msg;
    output_msg.depth_image.header = depth_msg->header;
  }

  output_msg.ppx = camera_info->k.at(2);
  output_msg.fx  = camera_info->k.at(0);
  output_msg.ppy = camera_info->k.at(5);
  output_msg.fy  = camera_info->k.at(4);

  output_msg.objects.reserve(result.size());

  geometry_msgs::msg::PoseArray pose_array;
  pose_array.header = msg->header;
  pose_array.poses.reserve(result.size());

  for (size_t i = 0; i < result.size(); i++) {
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
  const auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  const auto ms = elapsedTime.count();
  const double fps = (ms > 0) ? (1000.0 / static_cast<double>(ms)) : 0.0;
  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "[-FPS-]= %.2f (dt_ms=%lld)",
    fps,
    static_cast<long long>(ms));

  output_msg.process_time = ms;
  localize_pub->publish(output_msg);
  pose_pub->publish(pose_array);

  EPD::PerceptionResult completed_result;
  completed_result.source_observation_id = observation->observation_id();
  completed_result.sensor_stamp = observation->sensor_stamp();
  completed_result.frame_id = observation->frame_id();
  completed_result.success = true;
  completed_result.localization = output_msg;
  perception_result_store_.publish(std::move(completed_result));

}

void EasyPerceptionDeployment::process_tracking_work(
  const EPD::Observation::ConstSharedPtr & observation)
{
  if (!observation) {return;}
  const auto & msg = observation->rgb();
  const auto & depth_msg = observation->aligned_depth();
  const auto & camera_info = observation->camera_info();
  std::unique_lock<std::mutex> ort_lock(ort_mutex_);
  if (ortAgent_.requestAddressed && !service_mode_) {
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

  cv::Mat depth_img;
  if (depth_msg) {
    cv_bridge::CvImageConstPtr depth_imageptr;
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      depth_imageptr = cv_bridge::toCvShare(depth_msg);
    } else {
      depth_imageptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    }
    depth_img = depth_imageptr->image;
  } else {
    depth_img = cv::Mat::zeros(img.rows, img.cols, CV_16UC1);
  }

  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    ensureOrtAgentInitialized(img.rows, img.cols);
  }

  auto begin = std::chrono::high_resolution_clock::now();

  EPD::EPDObjectTracking result = [&]() {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    return ortAgent_.p3_ort_session->infer(
      img,
      depth_img,
      *camera_info,
      camera_to_plane_distance_mm,
      ortAgent_.tracker_type,
      ortAgent_.trackers,
      ortAgent_.tracker_logs,
      ortAgent_.tracker_results,
      ortAgent_.confidence_threshold);
  }();

  // Apply max_detections limit.
  {
    std::lock_guard<std::mutex> ort_guard(ort_mutex_);
    const int max_det = ortAgent_.max_detections;
    if (max_det > 0 && static_cast<int>(result.size()) > max_det) {
      result.objects.resize(max_det);
      result.object_ids.resize(max_det);
    }
  }

  detections_total_.fetch_add(result.size());
  std::vector<EPD::TrackingDetection> tracking_detections;
  tracking_detections.reserve(result.size());
  for (auto & object : result.objects) {
    object.source_observation_id = observation->observation_id();
    object.source_sensor_stamp = observation->sensor_stamp();
    object.source_frame = observation->frame_id();
    const auto quality = EPD::validateLocalizedObject(
      object, static_cast<uint32_t>(img.cols), static_cast<uint32_t>(img.rows),
      camera_info->k.at(0), camera_info->k.at(4), camera_info->k.at(2), camera_info->k.at(5),
      geometry_thresholds_);
    EPD::TrackingDetection detection;
    detection.name = object.name;
    detection.roi = object.roi;
    detection.centroid = object.centroid;
    detection.geometry_valid = quality == EPD::GeometryQuality::VALID;
    detection.detector_confidence = object.confidence;
    tracking_detections.push_back(std::move(detection));
    if (quality == EPD::GeometryQuality::VALID) {
      geometry_valid_total_.fetch_add(1);
      latest_valid_geometry_stamp_ns_.store(rclcpp::Time(observation->sensor_stamp()).nanoseconds());
    } else if (quality == EPD::GeometryQuality::DEGRADED) {
      geometry_degraded_total_.fetch_add(1);
    } else {
      geometry_invalid_total_.fetch_add(1);
    }
  }
  std::vector<EPD::TrackAssignment> track_assignments;
  {
    std::lock_guard<std::mutex> tracker_guard(temporal_tracker_mutex_);
    track_assignments = temporal_tracker_->update(
      observation->observation_id(), observation->sensor_stamp(), tracking_detections);
  }
  std::vector<EPD::LocalizedObject> valid_tracking_objects;
  std::vector<std::string> valid_tracking_ids;
  valid_tracking_objects.reserve(result.size());
  valid_tracking_ids.reserve(result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    if (result.objects[i].quality != EPD::GeometryQuality::VALID ||
      i >= track_assignments.size() || track_assignments[i].track_id == 0)
    {
      continue;
    }
    valid_tracking_ids.push_back(std::to_string(track_assignments[i].track_id));
    valid_tracking_objects.push_back(std::move(result.objects[i]));
  }
  result.objects = std::move(valid_tracking_objects);
  result.object_ids = std::move(valid_tracking_ids);

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
      cv_bridge::CvImage(msg->header, "bgr8", resultImg).toImageMsg();
    visual_pub.publish(*output_msg);
  }

  epd_msgs::msg::EPDObjectTracking output_msg;

  EPD::preserve_observation_header(output_msg, *observation);
  output_msg.frame_width = img.cols;
  output_msg.frame_height = img.rows;
  if (depth_msg) {
    output_msg.depth_image = *depth_msg;
    output_msg.depth_image.header = depth_msg->header;
  }

  output_msg.ppx = camera_info->k.at(2);
  output_msg.fx  = camera_info->k.at(0);
  output_msg.ppy = camera_info->k.at(5);
  output_msg.fy  = camera_info->k.at(4);

  output_msg.object_ids.reserve(result.size());
  output_msg.objects.reserve(result.size());

  geometry_msgs::msg::PoseArray pose_array;
  pose_array.header = msg->header;
  pose_array.poses.reserve(result.size());

  for (size_t i = 0; i < result.size(); i++) {
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
  const auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  const auto ms = elapsedTime.count();
  const double fps = (ms > 0) ? (1000.0 / static_cast<double>(ms)) : 0.0;
  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "[-FPS-]= %.2f (dt_ms=%lld)",
    fps,
    static_cast<long long>(ms));

  output_msg.process_time = ms;
  tracking_pub->publish(output_msg);
  pose_pub->publish(pose_array);

  EPD::PerceptionResult completed_result;
  completed_result.source_observation_id = observation->observation_id();
  completed_result.sensor_stamp = observation->sensor_stamp();
  completed_result.frame_id = observation->frame_id();
  completed_result.success = true;
  completed_result.tracking = output_msg;
  perception_result_store_.publish(std::move(completed_result));

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

// Filter detections by confidence_threshold and max_detections.
// Returns a new EPDObjectDetection containing only accepted detections.
EPD::EPDObjectDetection EasyPerceptionDeployment::applyDetectionFilters(
  const EPD::EPDObjectDetection & raw, float confidence_threshold, int max_detections) const
{
  EPD::EPDObjectDetection filtered(0);
  const bool has_masks = !raw.masks.empty();

  // Verify mask vector is consistent: either empty (no masks) or same size as bboxes.
  if (has_masks && raw.masks.size() != raw.size()) {
    throw std::runtime_error(
      "applyDetectionFilters: masks.size() (" + std::to_string(raw.masks.size()) +
      ") != bboxes.size() (" + std::to_string(raw.size()) +
      "). Inference output is inconsistent.");
  }

  for (size_t i = 0; i < raw.size(); i++) {
    if (raw.scores[i] < confidence_threshold) {
      continue;
    }
    if (max_detections > 0 &&
      static_cast<int>(filtered.bboxes.size()) >= max_detections)
    {
      break;
    }
    filtered.bboxes.push_back(raw.bboxes[i]);
    filtered.classIndices.push_back(raw.classIndices[i]);
    filtered.scores.push_back(raw.scores[i]);
    if (has_masks) {
      filtered.masks.push_back(raw.masks[i]);
    }
  }
  return filtered;
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
    ensureOrtAgentInitialized(img.rows, img.cols);
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
            ortAgent_.color_match_histogram_metric,
            ortAgent_.color_match_threshold);
        }

        EPD::EPDObjectDetection output_obj(result.size());
        output_obj.bboxes = result.bboxes;
        output_obj.classIndices = result.classIndices;
        output_obj.scores = result.scores;

        // Apply confidence threshold and max_detections filters.
        {
          std::lock_guard<std::mutex> ort_guard(ort_mutex_);
          output_obj = applyDetectionFilters(
            output_obj,
            ortAgent_.confidence_threshold,
            ortAgent_.max_detections);
        }

        if (visualize) {
          {
            std::lock_guard<std::mutex> ort_guard(ort_mutex_);
            resultImg = ortAgent_.visualize(output_obj, img);
          }
          sensor_msgs::msg::Image::SharedPtr output_msg =
            cv_bridge::CvImage(msg->header, "bgr8", resultImg).toImageMsg();
          visual_pub.publish(*output_msg);
        } else {
          epd_msgs::msg::EPDObjectDetection output_msg;
          output_msg.header = msg->header;
          for (size_t i = 0; i < output_obj.size(); i++) {
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
            ortAgent_.color_match_histogram_metric,
            ortAgent_.color_match_threshold);
        }

        EPD::EPDObjectDetection output_obj(result.size());
        output_obj.bboxes = result.bboxes;
        output_obj.classIndices = result.classIndices;
        output_obj.scores = result.scores;
        output_obj.masks = result.masks;

        // Apply confidence threshold and max_detections filters.
        {
          std::lock_guard<std::mutex> ort_guard(ort_mutex_);
          output_obj = applyDetectionFilters(
            output_obj,
            ortAgent_.confidence_threshold,
            ortAgent_.max_detections);
        }

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

          segmented_pcls.reserve(output_obj.size());
          for (size_t i = 0; i < output_obj.size(); i++) {
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
            cv_bridge::CvImage(msg->header, "bgr8", resultImg).toImageMsg();
          visual_pub.publish(*output_msg);
        } else {
          epd_msgs::msg::EPDObjectDetection output_msg;
          output_msg.header = msg->header;
          for (size_t i = 0; i < output_obj.size(); i++) {
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
  const auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  const auto ms = elapsedTime.count();
  const double fps = (ms > 0) ? (1000.0 / static_cast<double>(ms)) : 0.0;
  RCLCPP_INFO_THROTTLE(
    this->get_logger(),
    *this->get_clock(),
    2000,
    "[-FPS-]= %.2f (dt_ms=%lld)",
    fps,
    static_cast<long long>(ms));
  if (slow_frame_warn_ms_ > 0 && ms > static_cast<long long>(slow_frame_warn_ms_)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      5000,
      "Slow inference frame detected: %lld ms (threshold: %d ms).",
      static_cast<long long>(ms),
      slow_frame_warn_ms_);
  }
}

void EasyPerceptionDeployment::worker_loop()
{
  while (rclcpp::ok()) {
    unsigned int use_case_mode = 0;
    {
      std::lock_guard<std::mutex> ort_guard(ort_mutex_);
      use_case_mode = ortAgent_.useCaseMode;
    }

    if (use_case_mode == EPD::LOCALISATION_MODE || use_case_mode == EPD::TRACKING_MODE) {
      const auto observation = inference_scheduler_.wait_for_next(std::chrono::seconds(1));
      if (!observation) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (worker_stop_) {
          return;
        }
        continue;
      }
      const auto started_at = std::chrono::steady_clock::now();
      try {
        if (!observation->rgb() ||
          (!observation->aligned_depth() && use_depth_) || !observation->camera_info())
        {
          throw std::runtime_error("incomplete sensor data");
        }
        if (use_case_mode == EPD::LOCALISATION_MODE) {
          process_localize_work(observation);
        } else {
          process_tracking_work(observation);
        }
        const auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started_at).count();
        inference_scheduler_.complete(*observation, latency);
      } catch (const std::exception & e) {
        const auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started_at).count();
        inference_scheduler_.fail(*observation, latency);
        RCLCPP_ERROR(
          this->get_logger(), "Inference failed for observation_id=%llu: %s",
          static_cast<unsigned long long>(observation->observation_id()), e.what());
      } catch (...) {
        const auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started_at).count();
        inference_scheduler_.fail(*observation, latency);
        RCLCPP_ERROR(
          this->get_logger(), "Unknown inference failure for observation_id=%llu",
          static_cast<unsigned long long>(observation->observation_id()));
      }
      continue;
    }

    sensor_msgs::msg::Image::ConstSharedPtr image_msg;
    {
      std::unique_lock<std::mutex> lock(data_mutex_);
      data_cv_.wait(lock, [this]() {return worker_stop_ || image_pending_;});
      if (worker_stop_) {
        return;
      }
      image_msg = latest_image_;
      image_pending_ = false;
    }
    if (image_msg) {
      try {
        process_image_work(image_msg);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in process_image_work: %s", e.what());
      } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in process_image_work");
      }
      {
        std::lock_guard<std::mutex> lock(data_mutex_);
        last_processed_frame_time_ = this->now();
        has_processed_frame_ = true;
      }
    }
  }
}

#endif  // EPD_UTILS_LIB__EASY_PERCEPTION_DEPLOYMENT_HPP_
