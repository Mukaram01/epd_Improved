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

#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/common/centroid.h"
#include "pcl/common/eigen.h"

#include "opencv2/opencv.hpp"

#include "tf2/LinearMath/Quaternion.h"

#include "p3_ort_base.hpp"
#include "epd_utils_lib/usecase_config.hpp"


namespace Ort
{
// Minimum valid z-depth (in metres) below which a point is considered invalid.
static constexpr float MIN_DEPTH_THRESHOLD_M = 0.0001f;

// Constructor
P3OrtBase::P3OrtBase(
  float ratio,
  int newW,
  int newH,
  int paddedW,
  int paddedH,
  const uint16_t numClasses,
  const std::string & modelPath,
  const boost::optional<size_t> & gpuIdx,
  const boost::optional<int> & intraOpNumThreads,
  const boost::optional<std::vector<std::vector<int64_t>>> & inputShapes)
: OrtBase(modelPath, gpuIdx, intraOpNumThreads, inputShapes),
  m_numClasses(numClasses),
  m_ratio(ratio),
  m_newW(newW),
  m_newH(newH),
  m_paddedW(paddedW),
  m_paddedH(paddedH),
  preprocess_buffer_(m_paddedH * m_paddedW * 3)
{}

// Destructor
P3OrtBase::~P3OrtBase()
{}

// Mutator: Classification
EPD::EPDObjectDetection P3OrtBase::infer(const cv::Mat & inputImg)
{
  std::lock_guard<std::mutex> preprocessBufferLock(preprocess_buffer_mutex_);

  // Pass confThresh=0.0 so all detections pass through the ORT-level filter;
  // the caller applies the user-configured confidence_threshold via
  // applyDetectionFilters() (defined in easy_perception_deployment.hpp).
  return this->infer(
    inputImg, m_newW, m_newH,
    m_paddedW, m_paddedH, m_ratio, preprocess_buffer_.data(), 0.0f,
    cv::Scalar(102.9801, 115.9465, 122.7717));
}

// Mutator: Localization
EPD::EPDObjectLocalization P3OrtBase::infer(
  const cv::Mat & inputImg,
  const cv::Mat & depthImg,
  sensor_msgs::msg::CameraInfo camera_info,
  double camera_to_plane_distance_mm,
  float confThresh)
{
  std::lock_guard<std::mutex> preprocessBufferLock(preprocess_buffer_mutex_);

  const int max_depth_mm = static_cast<int>(camera_to_plane_distance_mm);
  return this->infer(
    inputImg, depthImg, camera_info, camera_to_plane_distance_mm,
    m_newW, m_newH, m_paddedW, m_paddedH, m_ratio, preprocess_buffer_.data(), confThresh,
    cv::Scalar(102.9801, 115.9465, 122.7717), max_depth_mm);
}

// Mutator: Tracking
EPD::EPDObjectTracking P3OrtBase::infer(
  const cv::Mat & inputImg,
  const cv::Mat & depthImg,
  sensor_msgs::msg::CameraInfo camera_info,
  double camera_to_plane_distance_mm,
  const std::string tracker_type,
  std::vector<cv::Ptr<cv::Tracker>> & trackers,
  std::vector<int> & tracker_logs,
  std::vector<EPD::LabelledRect2d> & tracker_results,
  float confThresh)
{
  std::lock_guard<std::mutex> preprocessBufferLock(preprocess_buffer_mutex_);

  const int max_depth_mm = static_cast<int>(camera_to_plane_distance_mm);
  return this->infer(
    inputImg, depthImg, camera_info, camera_to_plane_distance_mm,
    tracker_type, trackers, tracker_logs, tracker_results,
    m_newW, m_newH, m_paddedW, m_paddedH, m_ratio, preprocess_buffer_.data(), confThresh,
    cv::Scalar(102.9801, 115.9465, 122.7717), max_depth_mm);
}


void P3OrtBase::initClassNames(const std::vector<std::string> & classNames)
{
  if (classNames.size() != m_numClasses) {
    throw std::runtime_error("Mismatch number of classes\n");
  }
  m_classNames = classNames;
}

void P3OrtBase::preprocess(
  float * dst,
  const cv::Mat & imgSrc,
  const int64_t targetImgWidth,
  const int64_t targetImgHeight,
  const int numChannels) const
{
  for (int i = 0; i < targetImgHeight; ++i) {
    for (int j = 0; j < targetImgWidth; ++j) {
      for (int c = 0; c < numChannels; ++c) {
        dst[c * targetImgHeight * targetImgWidth +
          i * targetImgWidth + j] =
          imgSrc.ptr<float>(i, j)[c];
      }
    }
  }
}

// Mutator 4
EPD::EPDObjectDetection P3OrtBase::infer(
  const cv::Mat & inputImg,
  int newW,
  int newH,
  int paddedW,
  int paddedH,
  float ratio,
  float * dst,
  float confThresh,
  const cv::Scalar & meanVal)
{
  cv::Mat tmpImg;
  cv::resize(inputImg, tmpImg, cv::Size(newW, newH));

  tmpImg.convertTo(tmpImg, CV_32FC3);
  tmpImg -= meanVal;

  cv::Mat paddedImg(paddedH, paddedW, CV_32FC3, cv::Scalar(0, 0, 0));
  tmpImg.copyTo(paddedImg(cv::Rect(0, 0, newW, newH)));

  this->preprocess(dst, paddedImg, paddedW, paddedH, 3);

  // boxes, labels, scores, masks
  auto inferenceOutput = (*this)({dst});

  if (inferenceOutput[1].second.size() != 1) {
    throw std::runtime_error(
      "Unexpected inference output shape: expected 1 dimension for box count, got " +
      std::to_string(inferenceOutput[1].second.size()));
  }
  size_t nBoxes = inferenceOutput[1].second[0];

  // Determine mask dimensions from the inference output shape.
  // MaskRCNN outputs masks with shape [N, 1, H, W]; fall back to 28x28 if unreadable.
  int64_t mask_H = 28;
  int64_t mask_W = 28;
  {
    const auto & mask_shape = inferenceOutput[3].second;
    if (mask_shape.size() >= 4 && mask_shape[2] > 0 && mask_shape[3] > 0) {
      mask_H = mask_shape[2];
      mask_W = mask_shape[3];
    } else if (mask_shape.size() >= 3 && mask_shape[1] > 0 && mask_shape[2] > 0) {
      mask_H = mask_shape[1];
      mask_W = mask_shape[2];
    }
  }

  const float scale_x = inputImg.cols > 0 ? static_cast<float>(newW) / inputImg.cols : ratio;
  const float scale_y = inputImg.rows > 0 ? static_cast<float>(newH) / inputImg.rows : ratio;

  std::vector<std::array<int, 4>> bboxes;
  std::vector<uint64_t> classIndices;
  std::vector<float> scores;
  std::vector<cv::Mat> masks;

  bboxes.reserve(nBoxes);
  classIndices.reserve(nBoxes);
  scores.reserve(nBoxes);
  masks.reserve(nBoxes);

  for (size_t i = 0; i < nBoxes; ++i) {
    if (inferenceOutput[2].first[i] > confThresh) {
      int xmin = static_cast<int>(inferenceOutput[0].first[i * 4 + 0] / scale_x);
      int ymin = static_cast<int>(inferenceOutput[0].first[i * 4 + 1] / scale_y);
      int xmax = static_cast<int>(inferenceOutput[0].first[i * 4 + 2] / scale_x);
      int ymax = static_cast<int>(inferenceOutput[0].first[i * 4 + 3] / scale_y);

      xmin = std::max<int>(xmin, 0);
      ymin = std::max<int>(ymin, 0);
      xmax = std::min<int>(xmax, inputImg.cols);
      ymax = std::min<int>(ymax, inputImg.rows);

      bboxes.emplace_back(std::array<int, 4>{xmin, ymin, xmax, ymax});
      classIndices.emplace_back(reinterpret_cast<int64_t *>(inferenceOutput[1].first)[i]);
      scores.emplace_back(inferenceOutput[2].first[i]);

      cv::Mat curMask(mask_H, mask_W, CV_32FC1);
      memcpy(
        curMask.data,
        inferenceOutput[3].first + i * mask_H * mask_W,
        mask_H * mask_W * sizeof(float));
      masks.emplace_back(curMask);
    }
  }

  if (bboxes.size() == 0) {
    EPD::EPDObjectDetection output_msg(0);
    return output_msg;
  }

  EPD::EPDObjectDetection output_obj(bboxes.size());
  output_obj.bboxes = bboxes;
  output_obj.classIndices = classIndices;
  output_obj.scores = scores;
  output_obj.masks = masks;

  return output_obj;
}

double P3OrtBase::findMedian(cv::Mat depthImg, int max_depth_mm)
{
  double m = (depthImg.rows * depthImg.cols) / 2;
  int bin = 0;
  double median = -1.0;

  const int histSize = std::max(max_depth_mm, 1);
  float range[] = {0, static_cast<float>(max_depth_mm)};
  const float * histRange = {range};
  bool uniform = true;
  bool accumulate = false;
  cv::Mat hist;
  cv::calcHist(&depthImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

  for (int i = 0; i < histSize && median < 0.0; ++i) {
    bin += cvRound(hist.at<float>(i));
    if (bin > m && median < 0.0) {
      median = i;
    }
  }

  return median;
}

double P3OrtBase::findMin(cv::Mat depthImg, int max_depth_mm)
{
  double min = -1.0;

  const int histSize = std::max(max_depth_mm, 1);
  float range[] = {0, static_cast<float>(max_depth_mm)};
  const float * histRange = {range};
  bool uniform = true;
  bool accumulate = false;
  cv::Mat hist;
  cv::calcHist(&depthImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

  // Skip bin 0, which represents zero-depth (invalid) pixels.
  // Return the first bin index > 0 that has at least one point — this is
  // the nearest valid surface depth.
  for (int i = 1; i < histSize; ++i) {
    if (cvRound(hist.at<float>(i)) > 0) {
      min = i;
      break;
    }
  }

  return min;
}

// A mutator function that will output an EPD::EPDObjectLocalization object that
// contains all information required for Localization.

// Shared per-object geometry helper used by both localization and tracking infer.
// Fills roi, mask, centroid, length, breadth, height, segmented_pcl, and axis
// fields on the provided LocalizedObject.
// Returns true on success, false when the mask is degenerate (no valid contour)
// and the caller should skip this detection.
bool P3OrtBase::populateObjectGeometry(
  EPD::LocalizedObject & obj,
  const std::array<float, 4> & curBbox,
  const cv::Mat & rawMask,
  const cv::Mat & depthImg,
  float ppx, float fx, float ppy, float fy,
  float table_depth,
  bool depth_is_float,
  double camera_to_plane_distance_mm,
  int max_depth_mm,
  const std::string & pcl_frame_id,
  float maskThreshold)
{
  // ROI
  obj.roi.x_offset = curBbox[0];
  obj.roi.y_offset = curBbox[1];
  obj.roi.height   = curBbox[3] - curBbox[1];
  obj.roi.width    = curBbox[2] - curBbox[0];

  // Store original (un-resized) mask
  obj.mask = rawMask.clone();

  const cv::Rect curBoxRect(
    cv::Point(static_cast<int>(curBbox[0]), static_cast<int>(curBbox[1])),
    cv::Point(static_cast<int>(curBbox[2]), static_cast<int>(curBbox[3])));

  // Resize raw mask to bbox dimensions and binarize
  cv::Mat resizedMask;
  cv::resize(rawMask, resizedMask, curBoxRect.size());
  cv::Mat finalMask = (resizedMask > maskThreshold);

  cv::Mat tempFinalMask;
  finalMask.convertTo(tempFinalMask, CV_8U);

  std::vector<cv::Mat> contours;
  cv::Mat hierarchy;
  cv::findContours(
    tempFinalMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  // If there are no contours (e.g. zero-area mask), set default values and signal skip.
  if (contours.empty()) {
    obj.centroid.x = 0.0;
    obj.centroid.y = 0.0;
    obj.centroid.z = 0.0;
    obj.length  = 0.0f;
    obj.breadth = 0.0f;
    obj.height  = 0.0f;
    obj.axis.x  = 0.0f;
    obj.axis.y  = 0.0f;
    obj.axis.z  = 1.0f;
    return false;
  }

  // Find the largest contour
  double maxArea = 0;
  int maxAreaContourId = -1;
  for (unsigned int j = 0; j < contours.size(); j++) {
    double newArea = cv::contourArea(contours[j]);
    if (newArea > maxArea) {
      maxArea = newArea;
      maxAreaContourId = j;
    }
  }

  if (maxAreaContourId < 0) {
    return false;
  }
  const unsigned int maxID = static_cast<unsigned int>(maxAreaContourId);

  // Compute oriented bounding rect from the largest contour
  cv::RotatedRect minRect = cv::minAreaRect(cv::Mat(contours[maxID]));
  cv::Point2f rect_points[4];
  minRect.points(rect_points);

  // Mid-points of each side
  const cv::Point pt_a = (rect_points[0] + rect_points[3]) / 2;
  const cv::Point pt_b = (rect_points[1] + rect_points[2]) / 2;
  const cv::Point pt_c = (rect_points[0] + rect_points[1]) / 2;
  const cv::Point pt_d = (rect_points[3] + rect_points[2]) / 2;

  // Bbox centre in image coordinates
  const cv::Point rotated_mid(
    static_cast<int>((curBbox[0] + curBbox[2]) / 2.0f),
    static_cast<int>((curBbox[1] + curBbox[3]) / 2.0f));

  const float obj_surface_depth = this->findMin(depthImg(curBoxRect), max_depth_mm) * 0.001f;
  const float cx = (rotated_mid.x - ppx) / fx * obj_surface_depth;
  const float cy = (rotated_mid.y - ppy) / fy * obj_surface_depth;

  obj.centroid.x = cx;
  obj.centroid.y = cy;
  obj.centroid.z = obj_surface_depth + (table_depth - obj_surface_depth) / 2.0f;

  // Object real-world size: longer side → length, shorter side → breadth
  if (cv::norm(rect_points[0] - rect_points[1]) > cv::norm(rect_points[1] - rect_points[2])) {
    obj.length  = obj_surface_depth * std::sqrt(
      std::pow((pt_a.x - pt_b.x) / fx, 2) + std::pow((pt_a.y - pt_b.y) / fy, 2));
    obj.breadth = obj_surface_depth * std::sqrt(
      std::pow((pt_c.x - pt_d.x) / fx, 2) + std::pow((pt_c.y - pt_d.y) / fy, 2));
  } else {
    obj.breadth = obj_surface_depth * std::sqrt(
      std::pow((pt_a.x - pt_b.x) / fx, 2) + std::pow((pt_a.y - pt_b.y) / fy, 2));
    obj.length  = obj_surface_depth * std::sqrt(
      std::pow((pt_c.x - pt_d.x) / fx, 2) + std::pow((pt_c.y - pt_d.y) / fy, 2));
  }
  obj.height = table_depth - obj_surface_depth;

  // Build segmented point cloud from masked depth pixels
  pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  segmented_cloud->header.frame_id = pcl_frame_id;
  segmented_cloud->is_dense = true;

  for (int j = 0; j < tempFinalMask.rows; j++) {
    for (int k = 0; k < tempFinalMask.cols; k++) {
      if (tempFinalMask.at<uchar>(j, k) == 0) {
        continue;
      }
      float z = 0.0f;
      if (depth_is_float) {
        z = depthImg.at<float>(curBoxRect.y + j, curBoxRect.x + k);
      } else {
        z = static_cast<float>(
          depthImg.at<uint16_t>(curBoxRect.y + j, curBoxRect.x + k)) * 0.001f;
      }
      if (std::abs(z) < MIN_DEPTH_THRESHOLD_M ||
        std::abs(z) > camera_to_plane_distance_mm * 0.001)
      {
        continue;
      }
      const float px = static_cast<float>((curBoxRect.x + k - ppx) / fx) * z;
      const float py = static_cast<float>((curBoxRect.y + j - ppy) / fy) * z;
      segmented_cloud->points.emplace_back(px, py, z);
    }
  }
  obj.segmented_pcl = *segmented_cloud;

  // Determine principal axis via PCA on the segmented cloud
  if (obj.segmented_pcl.empty()) {
    obj.axis.x = 0.0f;
    obj.axis.y = 0.0f;
    obj.axis.z = 1.0f;
  } else {
    Eigen::Vector4f centerpoint;
    Eigen::Vector3f eigenvalues;
    Eigen::Matrix3f eigenvectors;
    Eigen::Matrix3f covariance_matrix;

    pcl::compute3DCentroid(obj.segmented_pcl, centerpoint);
    pcl::computeCovarianceMatrix(obj.segmented_pcl, centerpoint, covariance_matrix);
    pcl::eigen33(covariance_matrix, eigenvectors, eigenvalues);

    Eigen::Vector3f axis(
      eigenvectors.col(2)(0),
      eigenvectors.col(2)(1),
      eigenvectors.col(2)(2));
    axis = axis.normalized();

    obj.axis.x = axis(0);
    obj.axis.y = axis(1);
    obj.axis.z = axis(2);
  }

  return true;
}

EPD::EPDObjectLocalization P3OrtBase::infer(
  const cv::Mat & inputImg,
  const cv::Mat & depthImg,
  sensor_msgs::msg::CameraInfo camera_info,
  double camera_to_plane_distance_mm,
  int newW,
  int newH,
  int paddedW,
  int paddedH,
  float ratio,
  float * dst,
  float confThresh,
  const cv::Scalar & meanVal,
  int max_depth_mm)
{
  cv::Mat tmpImg;
  cv::resize(inputImg, tmpImg, cv::Size(newW, newH));

  tmpImg.convertTo(tmpImg, CV_32FC3);
  tmpImg -= meanVal;

  cv::Mat paddedImg(paddedH, paddedW, CV_32FC3, cv::Scalar(0, 0, 0));
  tmpImg.copyTo(paddedImg(cv::Rect(0, 0, newW, newH)));

  this->preprocess(dst, paddedImg, paddedW, paddedH, 3);

  // boxes, labels, scores, masks
  auto inferenceOutput = (*this)({dst});

  if (inferenceOutput[1].second.size() != 1) {
    throw std::runtime_error(
      "Unexpected inference output shape: expected 1 dimension for box count, got " +
      std::to_string(inferenceOutput[1].second.size()));
  }
  size_t nBoxes = inferenceOutput[1].second[0];

  // Determine mask dimensions from the inference output shape.
  // MaskRCNN outputs masks with shape [N, 1, H, W]; fall back to 28x28 if unreadable.
  int64_t mask_H = 28;
  int64_t mask_W = 28;
  {
    const auto & mask_shape = inferenceOutput[3].second;
    if (mask_shape.size() >= 4 && mask_shape[2] > 0 && mask_shape[3] > 0) {
      mask_H = mask_shape[2];
      mask_W = mask_shape[3];
    } else if (mask_shape.size() >= 3 && mask_shape[1] > 0 && mask_shape[2] > 0) {
      mask_H = mask_shape[1];
      mask_W = mask_shape[2];
    }
  }

  const float scale_x = inputImg.cols > 0 ? static_cast<float>(newW) / inputImg.cols : ratio;
  const float scale_y = inputImg.rows > 0 ? static_cast<float>(newH) / inputImg.rows : ratio;

  std::vector<std::array<float, 4>> bboxes;
  std::vector<uint64_t> classIndices;
  std::vector<float> scores;
  std::vector<cv::Mat> masks;

  bboxes.reserve(nBoxes);
  classIndices.reserve(nBoxes);
  scores.reserve(nBoxes);
  masks.reserve(nBoxes);

  for (size_t i = 0; i < nBoxes; ++i) {
    if (inferenceOutput[2].first[i] > confThresh) {
      float xmin = inferenceOutput[0].first[i * 4 + 0] / scale_x;
      float ymin = inferenceOutput[0].first[i * 4 + 1] / scale_y;
      float xmax = inferenceOutput[0].first[i * 4 + 2] / scale_x;
      float ymax = inferenceOutput[0].first[i * 4 + 3] / scale_y;

      xmin = std::max<float>(xmin, 0);
      ymin = std::max<float>(ymin, 0);
      xmax = std::min<float>(xmax, inputImg.cols);
      ymax = std::min<float>(ymax, inputImg.rows);

      bboxes.emplace_back(std::array<float, 4>{xmin, ymin, xmax, ymax});
      classIndices.emplace_back(reinterpret_cast<int64_t *>(inferenceOutput[1].first)[i]);
      scores.emplace_back(inferenceOutput[2].first[i]);

      cv::Mat curMask(mask_H, mask_W, CV_32FC1);
      memcpy(
        curMask.data,
        inferenceOutput[3].first + i * mask_H * mask_W,
        mask_H * mask_W * sizeof(float));
      masks.emplace_back(curMask);
    }
  }

  std::vector<std::string> allClassNames = this->getClassNames();
  float maskThreshold = 0.5;

  if (bboxes.size() != classIndices.size()) {
    throw std::runtime_error(
      "Mismatch between bboxes and classIndices sizes in inference output");
  }
  // if (!allClassNames.empty()) {
  //   assert(
  //     allClassNames.size() >
  //     *std::max_element(classIndices.begin(), classIndices.end()));
  // }

  // If there is zero bounding boxes generated, return empty EPDObjectLocalization object.
  if (bboxes.size() == 0) {
    EPD::EPDObjectLocalization output_msg(0);
    return output_msg;
  }

  EPD::EPDObjectLocalization output_obj(bboxes.size());

  // Determine whether the depth image carries float (metres) or uint16 (mm) data.
  const bool depth_is_float = (depthImg.type() == CV_32FC1);
  // Use camera_info frame_id for the segmented point clouds.
  const std::string pcl_frame_id = camera_info.header.frame_id.empty() ?
    "camera_color_optical_frame" : camera_info.header.frame_id;

  // Compute table/plane depth using the configurable max range.
  float table_depth = this->findMedian(depthImg, max_depth_mm) * 0.001;

  const float ppx = camera_info.k.at(2);
  const float fx  = camera_info.k.at(0);
  const float ppy = camera_info.k.at(5);
  const float fy  = camera_info.k.at(4);

  /* START of Populating EPDObjectLocalization object */
  for (size_t i = 0; i < bboxes.size(); ++i) {
    const auto & curBbox = bboxes[i];
    const uint64_t classIdx = classIndices[i];
    output_obj.objects[i].name = allClassNames.empty() ?
      std::to_string(classIdx) : allClassNames[classIdx];

    populateObjectGeometry(
      output_obj.objects[i],
      curBbox,
      masks[i],
      depthImg,
      ppx, fx, ppy, fy,
      table_depth,
      depth_is_float,
      camera_to_plane_distance_mm,
      max_depth_mm,
      pcl_frame_id,
      maskThreshold);
  }
  // END of Populating EPDObjectLocalization object
  return output_obj;
}

// A mutator function that will output an EPD::EPDObjectTracking object that
// contains all information required for Tracking.
EPD::EPDObjectTracking P3OrtBase::infer(
  const cv::Mat & inputImg,
  const cv::Mat & depthImg,
  sensor_msgs::msg::CameraInfo camera_info,
  double camera_to_plane_distance_mm,
  const std::string tracker_type,
  std::vector<cv::Ptr<cv::Tracker>> & trackers,
  std::vector<int> & tracker_logs,
  std::vector<EPD::LabelledRect2d> & tracker_results,
  int newW,
  int newH,
  int paddedW,
  int paddedH,
  float ratio,
  float * dst,
  float confThresh,
  const cv::Scalar & meanVal,
  int max_depth_mm)
{
  cv::Mat tmpImg;
  cv::resize(inputImg, tmpImg, cv::Size(newW, newH));

  tmpImg.convertTo(tmpImg, CV_32FC3);
  tmpImg -= meanVal;

  cv::Mat paddedImg(paddedH, paddedW, CV_32FC3, cv::Scalar(0, 0, 0));
  tmpImg.copyTo(paddedImg(cv::Rect(0, 0, newW, newH)));

  this->preprocess(dst, paddedImg, paddedW, paddedH, 3);

  // boxes, labels, scores, masks
  auto inferenceOutput = (*this)({dst});

  if (inferenceOutput[1].second.size() != 1) {
    throw std::runtime_error(
      "Unexpected inference output shape: expected 1 dimension for box count, got " +
      std::to_string(inferenceOutput[1].second.size()));
  }
  size_t nBoxes = inferenceOutput[1].second[0];

  // Determine mask dimensions from the inference output shape.
  // MaskRCNN outputs masks with shape [N, 1, H, W]; fall back to 28x28 if unreadable.
  int64_t mask_H = 28;
  int64_t mask_W = 28;
  {
    const auto & mask_shape = inferenceOutput[3].second;
    if (mask_shape.size() >= 4 && mask_shape[2] > 0 && mask_shape[3] > 0) {
      mask_H = mask_shape[2];
      mask_W = mask_shape[3];
    } else if (mask_shape.size() >= 3 && mask_shape[1] > 0 && mask_shape[2] > 0) {
      mask_H = mask_shape[1];
      mask_W = mask_shape[2];
    }
  }

  const float scale_x = inputImg.cols > 0 ? static_cast<float>(newW) / inputImg.cols : ratio;
  const float scale_y = inputImg.rows > 0 ? static_cast<float>(newH) / inputImg.rows : ratio;

  std::vector<std::array<float, 4>> bboxes;
  std::vector<uint64_t> classIndices;
  std::vector<float> scores;
  std::vector<cv::Mat> masks;

  bboxes.reserve(nBoxes);
  classIndices.reserve(nBoxes);
  scores.reserve(nBoxes);
  masks.reserve(nBoxes);

  for (size_t i = 0; i < nBoxes; ++i) {
    if (inferenceOutput[2].first[i] > confThresh) {
      float xmin = inferenceOutput[0].first[i * 4 + 0] / scale_x;
      float ymin = inferenceOutput[0].first[i * 4 + 1] / scale_y;
      float xmax = inferenceOutput[0].first[i * 4 + 2] / scale_x;
      float ymax = inferenceOutput[0].first[i * 4 + 3] / scale_y;

      xmin = std::max<float>(xmin, 0);
      ymin = std::max<float>(ymin, 0);
      xmax = std::min<float>(xmax, inputImg.cols);
      ymax = std::min<float>(ymax, inputImg.rows);

      bboxes.emplace_back(std::array<float, 4>{xmin, ymin, xmax, ymax});
      classIndices.emplace_back(reinterpret_cast<int64_t *>(inferenceOutput[1].first)[i]);
      scores.emplace_back(inferenceOutput[2].first[i]);

      cv::Mat curMask(mask_H, mask_W, CV_32FC1);
      memcpy(
        curMask.data,
        inferenceOutput[3].first + i * mask_H * mask_W,
        mask_H * mask_W * sizeof(float));
      masks.emplace_back(curMask);
    }
  }

  std::vector<std::optional<size_t>> detection_to_tracker;
  tracking_evaluate(
    bboxes, inputImg, tracker_type, trackers, tracker_logs, tracker_results, detection_to_tracker);

  std::vector<std::string> allClassNames = this->getClassNames();
  float maskThreshold = 0.5;

  // assert(bboxes.size() == classIndices.size());
  // if (!allClassNames.empty()) {
  //   assert(
  //     allClassNames.size() >
  //     *std::max_element(classIndices.begin(), classIndices.end()));
  // }

  // If there is zero bounding boxes generated, return empty EPDObjectTracking object.
  if (bboxes.size() == 0) {
    EPD::EPDObjectTracking output_msg(0);
    return output_msg;
  }

  EPD::EPDObjectTracking output_obj(bboxes.size());

  // Determine whether the depth image carries float (metres) or uint16 (mm) data.
  const bool depth_is_float = (depthImg.type() == CV_32FC1);
  // Use camera_info frame_id for the segmented point clouds.
  const std::string pcl_frame_id = camera_info.header.frame_id.empty() ?
    "camera_color_optical_frame" : camera_info.header.frame_id;

  float table_depth = this->findMedian(depthImg, max_depth_mm) * 0.001;

  const float ppx = camera_info.k.at(2);
  const float fx  = camera_info.k.at(0);
  const float ppy = camera_info.k.at(5);
  const float fy  = camera_info.k.at(4);

  // No. of objects will be equal to number of bboxes
  /* START of Populating EPDObjectTracking object */
  for (size_t i = 0; i < bboxes.size(); ++i) {
    const auto & curBbox = bboxes[i];
    const uint64_t classIdx = classIndices[i];
    output_obj.objects[i].name = allClassNames.empty() ?
      std::to_string(classIdx) : allClassNames[classIdx];

    // Map each detection to its matched tracker and preserve persistent object ids.
    if (i < detection_to_tracker.size() && detection_to_tracker[i].has_value()) {
      output_obj.object_ids[i] = tracker_results[detection_to_tracker[i].value()].obj_tag;
    } else {
      output_obj.object_ids[i] = "untracked";
    }

    populateObjectGeometry(
      output_obj.objects[i],
      curBbox,
      masks[i],
      depthImg,
      ppx, fx, ppy, fy,
      table_depth,
      depth_is_float,
      camera_to_plane_distance_mm,
      max_depth_mm,
      pcl_frame_id,
      maskThreshold);
  }
  // END of Populating EPDObjectTracking object
  return output_obj;
}

// Filter out accurate tracked objects using both new detection results and predicted
// tracking results while preserving object identity across frames.
void P3OrtBase::tracking_evaluate(
  const std::vector<std::array<float, 4>> & bboxes,
  const cv::Mat & img,
  const std::string tracker_type,
  std::vector<cv::Ptr<cv::Tracker>> & trackers,
  std::vector<int> & tracker_logs,
  std::vector<EPD::LabelledRect2d> & tracker_results,
  std::vector<std::optional<size_t>> & detection_to_tracker)
{
  constexpr float kIouMatchThreshold = 0.3F;
  constexpr float kCentroidDistanceThreshold = 80.0F;
  constexpr int kMaxMissedFrames = 3;

  detection_to_tracker.assign(bboxes.size(), std::nullopt);

  if (trackers.size() != tracker_results.size()) {
    trackers.clear();
    tracker_results.clear();
  }

  if (bboxes.empty()) {
    for (size_t i = tracker_results.size(); i-- > 0;) {
      tracker_results[i].missed_frames += 1;
      if (tracker_results[i].missed_frames > kMaxMissedFrames) {
        trackers.erase(trackers.begin() + i);
        tracker_results.erase(tracker_results.begin() + i);
      }
    }
    return;
  }

  for (size_t i = 0; i < trackers.size(); ++i) {
    cv::Rect tracker_bbox = tracker_results[i].obj_bounding_box;
    const bool ok = trackers[i]->update(img, tracker_bbox);
    if (ok) {
      tracker_results[i].obj_bounding_box = cv::Rect2d(tracker_bbox);
    }
  }

  std::vector<cv::Rect2d> detected_boxes;
  detected_boxes.reserve(bboxes.size());
  for (const auto & cur_bbox : bboxes) {
    detected_boxes.emplace_back(
      cur_bbox[0],
      cur_bbox[1],
      cur_bbox[2] - cur_bbox[0],
      cur_bbox[3] - cur_bbox[1]);
  }

  const size_t n_detections = detected_boxes.size();
  const size_t n_trackers = tracker_results.size();

  struct CandidateMatch
  {
    size_t detection_idx;
    size_t tracker_idx;
    float score;
  };

  std::vector<CandidateMatch> candidates;
  candidates.reserve(n_detections * std::max<size_t>(n_trackers, 1));

  for (size_t det_idx = 0; det_idx < n_detections; ++det_idx) {
    const auto & detected_box = detected_boxes[det_idx];
    const cv::Point2d det_center(
      detected_box.x + (detected_box.width / 2.0),
      detected_box.y + (detected_box.height / 2.0));

    for (size_t tracker_idx = 0; tracker_idx < n_trackers; ++tracker_idx) {
      const auto & tracked_box = tracker_results[tracker_idx].obj_bounding_box;
      const cv::Point2d tracked_center(
        tracked_box.x + (tracked_box.width / 2.0),
        tracked_box.y + (tracked_box.height / 2.0));

      const float iou_score = static_cast<float>(getIOU(detected_box, tracked_box));
      const float centroid_distance = static_cast<float>(cv::norm(det_center - tracked_center));

      if (iou_score < kIouMatchThreshold && centroid_distance > kCentroidDistanceThreshold) {
        continue;
      }

      const float normalized_distance =
        std::min(centroid_distance / kCentroidDistanceThreshold, 1.0F);
      const float score = iou_score - (0.2F * normalized_distance);
      candidates.push_back({det_idx, tracker_idx, score});
    }
  }

  std::sort(
    candidates.begin(),
    candidates.end(),
    [](const CandidateMatch & lhs, const CandidateMatch & rhs) {
      return lhs.score > rhs.score;
    });

  std::vector<bool> detection_matched(n_detections, false);
  std::vector<bool> tracker_matched(n_trackers, false);

  for (const auto & candidate : candidates) {
    if (detection_matched[candidate.detection_idx] || tracker_matched[candidate.tracker_idx]) {
      continue;
    }
    detection_matched[candidate.detection_idx] = true;
    tracker_matched[candidate.tracker_idx] = true;
    tracker_results[candidate.tracker_idx].obj_bounding_box = detected_boxes[candidate.detection_idx];
    tracker_results[candidate.tracker_idx].missed_frames = 0;
    detection_to_tracker[candidate.detection_idx] = candidate.tracker_idx;
  }

  for (size_t tracker_idx = n_trackers; tracker_idx-- > 0;) {
    if (tracker_matched[tracker_idx]) {
      continue;
    }
    tracker_results[tracker_idx].missed_frames += 1;
    if (tracker_results[tracker_idx].missed_frames > kMaxMissedFrames) {
      trackers.erase(trackers.begin() + tracker_idx);
      tracker_results.erase(tracker_results.begin() + tracker_idx);
    }
  }

  for (size_t det_idx = 0; det_idx < n_detections; ++det_idx) {
    if (detection_matched[det_idx]) {
      continue;
    }

    create_tracker_tag(tracker_logs);

    cv::Ptr<cv::Tracker> temp_tracker = create_tracker(tracker_type);
    temp_tracker->init(img, detected_boxes[det_idx]);
    trackers.push_back(temp_tracker);

    EPD::LabelledRect2d tracker_output;
    tracker_output.obj_tag = std::to_string(tracker_logs.back());
    tracker_output.obj_bounding_box = detected_boxes[det_idx];
    tracker_output.missed_frames = 0;
    tracker_results.push_back(tracker_output);

    detection_to_tracker[det_idx] = tracker_results.size() - 1;
  }
}

double P3OrtBase::getIOU(cv::Rect2d detected_box, cv::Rect2d tracked_box) const
{
  cv::Rect2d intersection = detected_box & tracked_box;
  const double union_area = detected_box.area() + tracked_box.area() - intersection.area();
  if (union_area <= 0.0) {
    return 0.0;
  }
  return intersection.area() / union_area;
}

void P3OrtBase::create_tracker_tag(std::vector<int> & tracker_logs)
{
  if (tracker_logs.size() == 0) {
    tracker_logs.push_back(0);
  } else {
    tracker_logs.push_back(tracker_logs.back() + 1);
  }
}

// Create tracker by name
cv::Ptr<cv::Tracker> P3OrtBase::create_tracker(std::string tracker_type)
{
  if (tracker_type == "KCF") {
    return cv::TrackerKCF::create();
  } else if (tracker_type == "MEDIANFLOW") {
    // OpenCV 4.5+ removed TrackerMedianFlow; fallback to CSRT for compatibility.
    return cv::TrackerCSRT::create();
  } else if (tracker_type == "CSRT") {
    return cv::TrackerCSRT::create();
  } else {
    throw std::runtime_error(
            "Invalid OpenCV Tracker name given in usecase_config.json. "
            "Please use [KCF, MEDIANFLOW (mapped to CSRT), CSRT] only.");
  }
}

}  // namespace Ort
