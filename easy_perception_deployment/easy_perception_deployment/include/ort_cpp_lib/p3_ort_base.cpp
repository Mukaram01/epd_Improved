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

  return this->infer(
    inputImg, m_newW, m_newH,
    m_paddedW, m_paddedH, m_ratio, preprocess_buffer_.data(), 0.5,
    cv::Scalar(102.9801, 115.9465, 122.7717));
}

// Mutator: Localization
EPD::EPDObjectLocalization P3OrtBase::infer(
  const cv::Mat & inputImg,
  const cv::Mat & depthImg,
  sensor_msgs::msg::CameraInfo camera_info,
  double camera_to_plane_distance_mm)
{
  std::lock_guard<std::mutex> preprocessBufferLock(preprocess_buffer_mutex_);

  return this->infer(
    inputImg, depthImg, camera_info, camera_to_plane_distance_mm,
    m_newW, m_newH, m_paddedW, m_paddedH, m_ratio, preprocess_buffer_.data(), 0.5,
    cv::Scalar(102.9801, 115.9465, 122.7717));
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
  std::vector<EPD::LabelledRect2d> & tracker_results)
{
  std::lock_guard<std::mutex> preprocessBufferLock(preprocess_buffer_mutex_);

  return this->infer(
    inputImg, depthImg, camera_info, camera_to_plane_distance_mm,
    tracker_type, trackers, tracker_logs, tracker_results,
    m_newW, m_newH, m_paddedW, m_paddedH, m_ratio, preprocess_buffer_.data(), 0.5,
    cv::Scalar(102.9801, 115.9465, 122.7717));
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

      cv::Mat curMask(28, 28, CV_32FC1);
      memcpy(
        curMask.data,
        inferenceOutput[3].first + i * 28 * 28,
        28 * 28 * sizeof(float));
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

double P3OrtBase::findMedian(cv::Mat depthImg)
{
  double m = (depthImg.rows * depthImg.cols) / 2;
  int bin = 0;
  double median = -1.0;

  // Setting to hardcoded 2000 millimeters
  // This is the limit of intel realsense D415.
  int histSize = 2000;
  float range[] = {0, 2000};
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

double P3OrtBase::findMin(cv::Mat depthImg)
{
  int bin = 0;
  double min = -1.0;

  // Setting to hardcoded 2000 millimeters
  // This is the limit of intel realsense D415.
  int histSize = 2000;
  float range[] = {0, 2000};
  const float * histRange = {range};
  bool uniform = true;
  bool accumulate = false;
  cv::Mat hist;
  cv::calcHist(&depthImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

  for (int i = 0; i < histSize; ++i) {
    bin += cvRound(hist.at<float>(i));
    // Store the first depth value that is shared among more than 1 point.
    // Break and escape for loop.
    if (i != 0 && cvRound(hist.at<float>(i)) > 0) {
      // std::cout << "Depth Value = " << i << " has " << cvRound(hist.at<float>(i)) << std::endl;
      min = i;
      break;
    }
  }

  return min;
}

// DEBUG
// A mutator function that will output an EPD::EPDObjectLocalization object that
// contains all information required for Localization.
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

      cv::Mat curMask(28, 28, CV_32FC1);
      memcpy(
        curMask.data,
        inferenceOutput[3].first + i * 28 * 28,
        28 * 28 * sizeof(float));
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

  float table_depth = this->findMedian(depthImg) * 0.001;
  // No. of objects will be equal to number of bboxes
  /* START of Populating EPDObjectLocalization object */
  for (size_t i = 0; i < bboxes.size(); ++i) {
    const auto & curBbox = bboxes[i];
    const uint64_t classIdx = classIndices[i];
    cv::Mat curMask = masks[i].clone();
    const std::string curLabel = allClassNames.empty() ?
      std::to_string(classIdx) :
      allClassNames[classIdx];

    output_obj.objects[i].name = curLabel;
    // Top x of ROI
    output_obj.objects[i].roi.x_offset = curBbox[0];
    // Top y of ROI
    output_obj.objects[i].roi.y_offset = curBbox[1];
    // Bounding Box height as ROI
    output_obj.objects[i].roi.height = curBbox[3] - curBbox[1];
    // Bounding Box width as ROI
    output_obj.objects[i].roi.width = curBbox[2] - curBbox[0];

    output_obj.objects[i].mask = curMask;

    // Visualizing masks
    const cv::Rect curBoxRect(cv::Point(curBbox[0], curBbox[1]),
      cv::Point(curBbox[2], curBbox[3]));

    cv::resize(curMask, curMask, curBoxRect.size());

    // Assigning masks that exceed the maskThreshold.
    cv::Mat finalMask = (curMask > maskThreshold);

    std::vector<cv::Mat> contours;
    cv::Mat hierarchy;
    cv::Mat tempFinalMask;
    finalMask.convertTo(tempFinalMask, CV_8U);
    // Generate contours.
    cv::findContours(
      tempFinalMask, contours, hierarchy, cv::RETR_TREE,
      cv::CHAIN_APPROX_SIMPLE);

    // For more details, refer to link below:
    // https://tinyurl.com/y5qnnxud
    float ppx = camera_info.k.at(2);
    float fx = camera_info.k.at(0);
    float ppy = camera_info.k.at(5);
    float fy = camera_info.k.at(4);

    // Getting rotated rectangle and draw the major axis
    std::vector<cv::RotatedRect> minRect(contours.size());
    float obj_surface_depth;
    cv::Point pt_a, pt_b, pt_c, pt_d;
    cv::Point rotated_mid;

    // Getting only the largest contour
    // The largest contour is the one which has the largest area.
    double maxArea = 0;
    int maxAreaContourId = 999;
    for (unsigned int j = 0; j < contours.size(); j++) {
      double newArea = cv::contourArea(contours[j]);
      if (newArea > maxArea) {
        maxArea = newArea;
        maxAreaContourId = j;
      }  // End if
    }  // End for
    unsigned int maxID = maxAreaContourId;

    for (unsigned int index = 0; index < contours.size(); index++) {
      if (index != maxID) {
        continue;
      }
      // Function that compute rotated rectangle based on contours
      minRect[index] = cv::minAreaRect(cv::Mat(contours[index]));
      cv::Point2f rect_points[4];
      // 4 points of the rotated rectangle
      minRect[index].points(rect_points);

      // Mid points of the each side of the rotated rectangle
      pt_a = (rect_points[0] + rect_points[3]) / 2;
      pt_b = (rect_points[1] + rect_points[2]) / 2;
      pt_c = (rect_points[0] + rect_points[1]) / 2;
      pt_d = (rect_points[3] + rect_points[2]) / 2;

      // For temporary, bboxes center
      rotated_mid = (cv::Point(curBbox[0], curBbox[1]) +
        cv::Point(curBbox[2], curBbox[3])) / 2;

      obj_surface_depth = this->findMin(depthImg(curBoxRect)) * 0.001;
      float x = (rotated_mid.x - ppx) / fx * obj_surface_depth;
      float y = (rotated_mid.y - ppy) / fy * obj_surface_depth;

      output_obj.objects[i].centroid.x = x;
      output_obj.objects[i].centroid.y = y;
      output_obj.objects[i].centroid.z = obj_surface_depth +
        (table_depth - obj_surface_depth) / 2;

      // Get Real Size and angle of object
      // Compare the length of 2 side of the rectangle,
      // the longer side will be the major axis
      if (cv::norm(rect_points[0] - rect_points[1]) >
        cv::norm(rect_points[1] - rect_points[2]))
      {
        // Calculates the length of the object
        output_obj.objects[i].length = obj_surface_depth * sqrt(
          pow((pt_a.x - pt_b.x) / fx, 2) +
          pow((pt_a.y - pt_b.y) / fy, 2));
        // Calculates the breadth of the object
        output_obj.objects[i].breadth = obj_surface_depth * sqrt(
          pow((pt_c.x - pt_d.x) / fx, 2) +
          pow((pt_c.y - pt_d.y) / fy, 2));
      } else {
        // Gets object breadth and length
        output_obj.objects[i].breadth = obj_surface_depth * sqrt(
          pow((pt_a.x - pt_b.x) / fx, 2) +
          pow((pt_a.y - pt_b.y) / fy, 2));
        output_obj.objects[i].length = obj_surface_depth * sqrt(
          pow((pt_c.x - pt_d.x) / fx, 2) +
          pow((pt_c.y - pt_d.y) / fy, 2));
      }
      // Setting height of object
      output_obj.objects[i].height = table_depth - obj_surface_depth;

      pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      segmented_cloud->header.frame_id = "camera_color_optical_frame";
      segmented_cloud->is_dense = true;


      // Converting Depth Image to PointCloud
      for (int j = 0; j < tempFinalMask.rows; j++) {
        for (int k = 0; k < tempFinalMask.cols; k++) {
          // TODO(cardboardcode) convert segmented mask into segmented pointcloud
          int pixelValue = static_cast<int>(tempFinalMask.at<uchar>(j, k));

          if (pixelValue != 0) {
            float z = static_cast<float>(depthImg.at<uint16_t>(
                curBoxRect.y + j, curBoxRect.x + k) * 0.001);
            float x = static_cast<float>((curBoxRect.x + k - ppx) / fx) * z;
            float y = static_cast<float>((curBoxRect.y + j - ppy) / fy) * z;

            // Ignore all points that has a value of less than 0.1mm in z.
            if (std::abs(z) < 0.0001 || std::abs(z) > camera_to_plane_distance_mm * 0.001) {
              continue;
            } else {
              pcl::PointXYZ curPoint(x, y, z);
              segmented_cloud->points.push_back(curPoint);
            }
          }
        }
      }

      output_obj.objects[i].segmented_pcl = *segmented_cloud;

      // Determine object axis of segmented_pcl
      Eigen::Vector3f axis;
      Eigen::Vector4f centerpoint;
      Eigen::Vector3f eigenvalues;
      Eigen::Matrix3f eigenvectors;
      Eigen::Matrix3f covariance_matrix;

      if (output_obj.objects[i].segmented_pcl.empty()) {
        output_obj.objects[i].axis.x = 0.0f;
        output_obj.objects[i].axis.y = 0.0f;
        output_obj.objects[i].axis.z = 1.0f;
      } else {
        pcl::compute3DCentroid(output_obj.objects[i].segmented_pcl, centerpoint);

        pcl::computeCovarianceMatrix(
          output_obj.objects[i].segmented_pcl,
          centerpoint,
          covariance_matrix);
        pcl::eigen33(covariance_matrix, eigenvectors, eigenvalues);

        axis = Eigen::Vector3f(
          eigenvectors.col(2)(0),
          eigenvectors.col(2)(1),
          eigenvectors.col(2)(2));

        axis = axis.normalized();

        output_obj.objects[i].axis.x = axis(0);
        output_obj.objects[i].axis.y = axis(1);
        output_obj.objects[i].axis.z = axis(2);
      }
    }
  }
  // END of Populating EPDObjectLocalization object
  return output_obj;
}

// DEBUG
// A mutator function that will output an EPD::EPDObjectTracking object that
// contains all information required for Localization.
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

      cv::Mat curMask(28, 28, CV_32FC1);
      memcpy(
        curMask.data,
        inferenceOutput[3].first + i * 28 * 28,
        28 * 28 * sizeof(float));
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

  float table_depth = this->findMedian(depthImg) * 0.001;

  // No. of objects will be equal to number of bboxes
  /* START of Populating EPDObjectTracking object */
  for (size_t i = 0; i < bboxes.size(); ++i) {
    const auto & curBbox = bboxes[i];
    const uint64_t classIdx = classIndices[i];
    cv::Mat curMask = masks[i].clone();
    const std::string curLabel = allClassNames.empty() ?
      std::to_string(classIdx) :
      allClassNames[classIdx];

    output_obj.objects[i].name = curLabel;
    // Top x of ROI
    output_obj.objects[i].roi.x_offset = curBbox[0];
    // Top y of ROI
    output_obj.objects[i].roi.y_offset = curBbox[1];
    // Bounding Box height as ROI
    output_obj.objects[i].roi.height = curBbox[3] - curBbox[1];
    // Bounding Box width as ROI
    output_obj.objects[i].roi.width = curBbox[2] - curBbox[0];

    output_obj.objects[i].mask = curMask;

    // Map each detection to its matched tracker and preserve persistent object ids.
    if (i < detection_to_tracker.size() && detection_to_tracker[i].has_value()) {
      output_obj.object_ids[i] = tracker_results[detection_to_tracker[i].value()].obj_tag;
    } else {
      output_obj.object_ids[i] = "untracked";
    }

    // Visualizing masks
    const cv::Rect curBoxRect(cv::Point(curBbox[0], curBbox[1]),
      cv::Point(curBbox[2], curBbox[3]));

    cv::resize(curMask, curMask, curBoxRect.size());

    // Assigning masks that exceed the maskThreshold.
    cv::Mat finalMask = (curMask > maskThreshold);

    std::vector<cv::Mat> contours;
    cv::Mat hierarchy;
    cv::Mat tempFinalMask;
    finalMask.convertTo(tempFinalMask, CV_8U);
    // Generate contours.
    cv::findContours(
      tempFinalMask, contours, hierarchy, cv::RETR_TREE,
      cv::CHAIN_APPROX_SIMPLE);

    // For more details, refer to link below:
    // https://tinyurl.com/y5qnnxud
    float ppx = camera_info.k.at(2);
    float fx = camera_info.k.at(0);
    float ppy = camera_info.k.at(5);
    float fy = camera_info.k.at(4);

    // Getting rotated rectangle and draw the major axis
    std::vector<cv::RotatedRect> minRect(contours.size());
    float obj_surface_depth;
    cv::Point pt_a, pt_b, pt_c, pt_d;
    cv::Point rotated_mid;

    // Getting only the largest contour
    // The largest contour is the one which has the largest area.
    double maxArea = 0;
    int maxAreaContourId = 999;
    for (unsigned int j = 0; j < contours.size(); j++) {
      double newArea = cv::contourArea(contours[j]);
      if (newArea > maxArea) {
        maxArea = newArea;
        maxAreaContourId = j;
      }
    }
    unsigned int maxID = maxAreaContourId;

    for (unsigned int index = 0; index < contours.size(); index++) {
      if (index != maxID) {
        continue;
      }
      // Function that compute rotated rectangle based on contours
      minRect[index] = cv::minAreaRect(cv::Mat(contours[index]));
      cv::Point2f rect_points[4];
      // 4 points of the rotated rectangle
      minRect[index].points(rect_points);

      // Mid points of the each side of the rotated rectangle
      pt_a = (rect_points[0] + rect_points[3]) / 2;
      pt_b = (rect_points[1] + rect_points[2]) / 2;
      pt_c = (rect_points[0] + rect_points[1]) / 2;
      pt_d = (rect_points[3] + rect_points[2]) / 2;

      // For temporary, bboxes center
      rotated_mid = (cv::Point(curBbox[0], curBbox[1]) +
        cv::Point(curBbox[2], curBbox[3])) / 2;

      obj_surface_depth = this->findMin(depthImg(curBoxRect)) * 0.001;
      float x = (rotated_mid.x - ppx) / fx * obj_surface_depth;
      float y = (rotated_mid.y - ppy) / fy * obj_surface_depth;

      output_obj.objects[i].centroid.x = x;
      output_obj.objects[i].centroid.y = y;
      output_obj.objects[i].centroid.z = obj_surface_depth +
        (table_depth - obj_surface_depth) / 2;

      // Get Real Size and angle of object
      // Compare the length of 2 side of the rectangle,
      // the longer side will be the major axis
      if (cv::norm(rect_points[0] - rect_points[1]) >
        cv::norm(rect_points[1] - rect_points[2]))
      {
        // Calculates the length of the object
        output_obj.objects[i].length = obj_surface_depth * sqrt(
          pow((pt_a.x - pt_b.x) / fx, 2) +
          pow((pt_a.y - pt_b.y) / fy, 2));
        // Calculates the breadth of the object
        output_obj.objects[i].breadth = obj_surface_depth * sqrt(
          pow((pt_c.x - pt_d.x) / fx, 2) +
          pow((pt_c.y - pt_d.y) / fy, 2));
      } else {
        // Gets object breadth and length
        output_obj.objects[i].breadth = obj_surface_depth * sqrt(
          pow((pt_a.x - pt_b.x) / fx, 2) +
          pow((pt_a.y - pt_b.y) / fy, 2));
        output_obj.objects[i].length = obj_surface_depth * sqrt(
          pow((pt_c.x - pt_d.x) / fx, 2) +
          pow((pt_c.y - pt_d.y) / fy, 2));
      }
      // Setting height of object
      output_obj.objects[i].height = table_depth - obj_surface_depth;

      pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      segmented_cloud->header.frame_id = "camera_color_optical_frame";
      segmented_cloud->is_dense = true;


      // Converting Depth Image to PointCloud
      for (int j = 0; j < tempFinalMask.rows; j++) {
        for (int k = 0; k < tempFinalMask.cols; k++) {
          // TODO(cardboardcode) convert segmented mask into segmented pointcloud
          int pixelValue = static_cast<int>(tempFinalMask.at<uchar>(j, k));

          if (pixelValue != 0) {
            float z = static_cast<float>(depthImg.at<uint16_t>(
                curBoxRect.y + j, curBoxRect.x + k) * 0.001);
            float x = static_cast<float>((curBoxRect.x + k - ppx) / fx) * z;
            float y = static_cast<float>((curBoxRect.y + j - ppy) / fy) * z;

            // Ignore all points that has a value of less than 0.1mm in z.
            if (std::abs(z) < 0.0001 || std::abs(z) > camera_to_plane_distance_mm * 0.001) {
              continue;
            } else {
              pcl::PointXYZ curPoint(x, y, z);
              segmented_cloud->points.push_back(curPoint);
            }
          }
        }
      }

      output_obj.objects[i].segmented_pcl = *segmented_cloud;

      // Determine object axis of segmented_pcl
      Eigen::Vector3f axis;
      Eigen::Vector4f centerpoint;
      Eigen::Vector3f eigenvalues;
      Eigen::Matrix3f eigenvectors;
      Eigen::Matrix3f covariance_matrix;

      if (output_obj.objects[i].segmented_pcl.empty()) {
        output_obj.objects[i].axis.x = 0.0f;
        output_obj.objects[i].axis.y = 0.0f;
        output_obj.objects[i].axis.z = 1.0f;
      } else {
        pcl::compute3DCentroid(output_obj.objects[i].segmented_pcl, centerpoint);

        pcl::computeCovarianceMatrix(
          output_obj.objects[i].segmented_pcl,
          centerpoint,
          covariance_matrix);
        pcl::eigen33(covariance_matrix, eigenvectors, eigenvalues);

        axis = Eigen::Vector3f(
          eigenvectors.col(2)(0),
          eigenvectors.col(2)(1),
          eigenvectors.col(2)(2));

        axis = axis.normalized();

        output_obj.objects[i].axis.x = axis(0);
        output_obj.objects[i].axis.y = axis(1);
        output_obj.objects[i].axis.z = axis(2);
      }
    }
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
