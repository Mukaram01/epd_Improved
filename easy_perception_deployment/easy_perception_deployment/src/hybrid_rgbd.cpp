// Copyright 2026 Advanced Remanufacturing and Technology Centre
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

#include "epd_utils_lib/hybrid_rgbd.hpp"

#include <Eigen/Eigenvalues>
#include <pcl/common/centroid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <limits>

namespace EPD
{
EPDObjectLocalization refineRgbd(
  const EPDObjectDetection & detections, const cv::Mat & depth,
  const Observation & observation, const std::vector<std::string> & class_names,
  const HybridRgbdOptions & options, const GeometryThresholds & thresholds)
{
  EPDObjectLocalization result(detections.size());
  const auto & rgb = observation.rgb();
  const auto & dep = observation.aligned_depth();
  const auto & info = observation.camera_info();
  uint32_t input_failures = 0;
  const auto positive = [](double v) {return std::isfinite(v) && v > 0;};
  if (!positive(options.maximum_neighbor_distance_m) ||
    !positive(options.minimum_depth_m) || !positive(options.maximum_depth_m) ||
    options.maximum_depth_m <= options.minimum_depth_m ||
    !std::isfinite(options.minimum_component_fraction) ||
    options.minimum_component_fraction <= 0.5 || options.minimum_component_fraction > 1 ||
    !std::isfinite(options.maximum_overlap_fraction) || options.maximum_overlap_fraction < 0 ||
    options.maximum_overlap_fraction >= 1 ||
    !std::isfinite(options.maximum_visible_extent_m) || options.maximum_visible_extent_m < 0)
  {
    input_failures |= reason(GeometryFailure::INVALID_CONFIGURATION);
  }
  if (!rgb || !dep || !info || depth.empty() ||
    (depth.type() != CV_16UC1 && depth.type() != CV_32FC1))
  {
    input_failures |= reason(GeometryFailure::INSUFFICIENT_DEPTH);
  }
  if (rgb && dep && info) {
    if (rgb->header.frame_id.empty() || rgb->header.frame_id != dep->header.frame_id ||
      rgb->header.frame_id != info->header.frame_id)
    {
      input_failures |= reason(GeometryFailure::FRAME_MISMATCH);
    }
    if (!observation.synchronization().synchronized ||
      !observation.synchronization().exact_sensor_stamp ||
      !observation.synchronization().source_healthy ||
      rgb->header.stamp != dep->header.stamp || rgb->header.stamp != info->header.stamp ||
      (rgb->header.stamp.sec == 0 && rgb->header.stamp.nanosec == 0))
    {
      input_failures |= reason(GeometryFailure::UNSYNCHRONIZED_INPUT);
    }
    if (rgb->width != dep->width || rgb->height != dep->height ||
      info->width != rgb->width || info->height != rgb->height ||
      depth.cols != static_cast<int>(rgb->width) || depth.rows != static_cast<int>(rgb->height) ||
      !validIntrinsics(info->k[0], info->k[4], info->k[2], info->k[5]) ||
      info->binning_x > 1 || info->binning_y > 1 || info->roi.x_offset || info->roi.y_offset ||
      std::any_of(
        info->d.begin(), info->d.end(), [](double d) {
          return !std::isfinite(
            d) || d != 0;
        }))
    {
      input_failures |= reason(GeometryFailure::INVALID_INTRINSICS);
    }
  }
  if (detections.masks.size() != detections.size() ||
    detections.scores.size() != detections.size() ||
    detections.classIndices.size() != detections.size())
  {
    input_failures |= reason(GeometryFailure::INVALID_MASK);
  }

  // Count ownership BEFORE any geometry filtering, including later rejected proposals.
  cv::Mat ownership = cv::Mat::zeros(depth.size(), CV_32SC1);
  std::vector<cv::Rect> rois(detections.size());
  for (size_t i = 0; i < detections.size(); ++i) {
    auto & object = result.objects[i];
    object.source_observation_id = observation.observation_id();
    object.source_sensor_stamp = observation.sensor_stamp();
    object.source_frame = observation.frame_id();
    object.failure_reasons = input_failures;
    // Identity is independent of geometry: keep class and confidence even when
    // the RGB-D input cannot support refinement.
    if (i < detections.scores.size() && i < detections.classIndices.size()) {
      object.confidence = detections.scores[i];
      const auto label = detections.classIndices[i];
      object.name = label < class_names.size() ? class_names[label] : std::to_string(label);
    }
    if (input_failures) {continue;}
    const auto & b = detections.bboxes[i];
    if (b[0] < 0 || b[1] < 0 || b[2] <= b[0] || b[3] <= b[1] ||
      b[2] > depth.cols || b[3] > depth.rows)
    {
      object.failure_reasons |= reason(GeometryFailure::INVALID_ROI);
      continue;
    }
    rois[i] = cv::Rect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    object.roi.x_offset = b[0];
    object.roi.y_offset = b[1];
    object.roi.width = rois[i].width;
    object.roi.height = rois[i].height;
    if (detections.masks[i].empty() || detections.masks[i].type() != CV_32FC1 ||
      !cv::checkRange(detections.masks[i]))
    {
      object.failure_reasons |= reason(GeometryFailure::INVALID_MASK);
      continue;
    }
    cv::Mat resized;
    cv::resize(detections.masks[i], resized, rois[i].size());
    object.mask = resized > 0.5F;
    object.mask_pixel_count = cv::countNonZero(object.mask);
    cv::Mat votes;
    object.mask.convertTo(votes, CV_32SC1, 1.0 / 255.0);
    ownership(rois[i]) += votes;
  }

  // Exclusive support is measured before resolving any conflicts. A claimant
  // cannot improve its rank by receiving pixels earlier in the iteration.
  std::vector<double> exclusive_fraction(result.size(), 0.0);
  for (size_t i = 0; i < result.size(); ++i) {
    const auto & object = result.objects[i];
    if (object.failure_reasons || !object.mask_pixel_count) {continue;}
    const cv::Mat exclusive = (ownership(rois[i]) == 1) & object.mask;
    exclusive_fraction[i] = static_cast<double>(cv::countNonZero(exclusive)) /
      object.mask_pixel_count;
  }
  cv::Mat claimant(depth.size(), CV_32SC1, cv::Scalar(-1));
  for (size_t i = 0; i < result.size(); ++i) {
    const auto & object = result.objects[i];
    if (object.failure_reasons || exclusive_fraction[i] < options.minimum_component_fraction) {
      continue;
    }
    const auto roi = rois[i];
    for (int y = 0; y < roi.height; ++y) {
      for (int x = 0; x < roi.width; ++x) {
        if (!object.mask.at<uint8_t>(y, x) || ownership.at<int>(roi.y + y, roi.x + x) <= 1) {
          continue;
        }
        auto & claim = claimant.at<int>(roi.y + y, roi.x + x);
        claim = claim == -1 ? static_cast<int>(i) : -2;  // ties remain unowned
      }
    }
  }

  for (size_t i = 0; i < result.size(); ++i) {
    auto & object = result.objects[i];
    if (object.failure_reasons) {continue;}
    const auto roi = rois[i];
    const cv::Mat proposed = object.mask.clone();
    const cv::Mat disputed = (ownership(roi) > 1) & proposed;
    // A mostly contested proposal has no independent basis for an object pose.
    // Keep its source detection intact; its usable refined region is empty.
    if (exclusive_fraction[i] < options.minimum_component_fraction &&
      1.0 - exclusive_fraction[i] > options.maximum_overlap_fraction)
    {
      object.failure_reasons |= reason(GeometryFailure::OVERLAPPING_OWNERSHIP);
      object.quality = GeometryQuality::DEGRADED;
      object.mask.setTo(0);
      continue;
    }
    object.mask.setTo(0, disputed);
    const cv::Mat claim = claimant(roi) == static_cast<int>(i);
    object.mask.setTo(255, claim & proposed);
    cv::Mat points(roi.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat available = cv::Mat::zeros(roi.size(), CV_8UC1);
    size_t valid_count = 0;
    for (int y = 0; y < roi.height; ++y) {
      for (int x = 0; x < roi.width; ++x) {
        if (!object.mask.at<uint8_t>(y, x)) {continue;}
        const float z = depth.type() == CV_16UC1 ?
          depth.at<uint16_t>(roi.y + y, roi.x + x) * 0.001F : depth.at<float>(roi.y + y, roi.x + x);
        if (!std::isfinite(z) || z < options.minimum_depth_m || z > options.maximum_depth_m) {
          continue;
        }
        const float px = (roi.x + x - info->k[2]) / info->k[0] * z;
        const float py = (roi.y + y - info->k[5]) / info->k[4] * z;
        if (!std::isfinite(px) || !std::isfinite(py)) {continue;}
        points.at<cv::Vec3f>(y, x) = cv::Vec3f(px, py, z);
        available.at<uint8_t>(y, x) = 255;
      }
    }
    // Reclaimed pixels need a measured, depth-continuous path to exclusive
    // support. A strong 2D claim alone cannot cross a depth discontinuity.
    cv::Mat reached = cv::Mat::zeros(roi.size(), CV_8UC1);
    std::vector<cv::Point> frontier;
    const cv::Point offsets[] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (int y = 0; y < roi.height; ++y) {
      for (int x = 0; x < roi.width; ++x) {
        if (available.at<uint8_t>(y, x) && !disputed.at<uint8_t>(y, x)) {
          reached.at<uint8_t>(y, x) = 255;
          frontier.emplace_back(x, y);
        }
      }
    }
    for (size_t j = 0; j < frontier.size(); ++j) {
      const auto p = frontier[j];
      for (const auto & offset : offsets) {
        const auto q = p + offset;
        if (q.x < 0 || q.y < 0 || q.x >= roi.width || q.y >= roi.height ||
          reached.at<uint8_t>(q) || !available.at<uint8_t>(q)) {continue;}
        if (cv::norm(points.at<cv::Vec3f>(p) - points.at<cv::Vec3f>(q)) >
          options.maximum_neighbor_distance_m) {continue;}
        reached.at<uint8_t>(q) = 255;
        frontier.push_back(q);
      }
    }
    available &= reached;
    valid_count = cv::countNonZero(available);
    const cv::Mat recovered = disputed & available;
    const size_t unresolved = cv::countNonZero(disputed) - cv::countNonZero(recovered);
    if (object.mask_pixel_count &&
      unresolved > options.maximum_overlap_fraction * object.mask_pixel_count)
    {
      object.failure_reasons |= reason(GeometryFailure::OVERLAPPING_OWNERSHIP);
      object.quality = GeometryQuality::DEGRADED;
      object.mask.setTo(0);
      continue;
    }
    // Four-neighbour Euclidean connectivity retains sloped/curved surfaces, while
    // refusing depth discontinuities. Never grow outside the detector's own mask.
    std::vector<cv::Point> largest;
    std::vector<std::vector<cv::Point>> components;
    cv::Mat visited = cv::Mat::zeros(roi.size(), CV_8UC1);
    for (int y = 0; y < roi.height; ++y) {
      for (int x = 0; x < roi.width; ++x) {
        if (!available.at<uint8_t>(y, x) || visited.at<uint8_t>(y, x)) {continue;}
        std::vector<cv::Point> component{{x, y}};
        visited.at<uint8_t>(y, x) = 1;
        for (size_t j = 0; j < component.size(); ++j) {
          const cv::Point p = component[j];
          for (const auto & offset : offsets) {
            const cv::Point q = p + offset;
            if (q.x < 0 || q.y < 0 || q.x >= roi.width || q.y >= roi.height ||
              !available.at<uint8_t>(q) || visited.at<uint8_t>(q)) {continue;}
            if (cv::norm(points.at<cv::Vec3f>(p) - points.at<cv::Vec3f>(q)) >
              options.maximum_neighbor_distance_m) {continue;}
            visited.at<uint8_t>(q) = 1;
            component.push_back(q);
          }
        }
        if (component.size() > largest.size()) {largest = component;}
        components.push_back(std::move(component));
      }
    }
    // Preserve small, disconnected visible fragments supported by the dominant
    // surface in metric space. Query the original core only: no chain of weak
    // fragments may expand the recovery radius. Never fill pixels absent from
    // this owner's proposal, or acquire another owner's exclusive pixels.
    if (!largest.empty() && largest.size() >= valid_count * options.minimum_component_fraction) {
      auto core = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      float low_z = std::numeric_limits<float>::infinity();
      float high_z = -low_z;
      for (const auto & p : largest) {
        const auto xyz = points.at<cv::Vec3f>(p);
        core->emplace_back(xyz[0], xyz[1], xyz[2]);
        low_z = std::min(low_z, xyz[2]);
        high_z = std::max(high_z, xyz[2]);
      }
      pcl::KdTreeFLANN<pcl::PointXYZ> tree;
      tree.setInputCloud(core);
      const auto core_size = largest.size();
      for (const auto & component : components) {
        if (component.size() >= core_size) {continue;}
        bool supported = false, compatible_depth = true;
        for (const auto & p : component) {
          const auto xyz = points.at<cv::Vec3f>(p);
          if (xyz[2] < low_z - options.maximum_neighbor_distance_m ||
            xyz[2] > high_z + options.maximum_neighbor_distance_m)
          {
            compatible_depth = false;
            break;
          }
          std::vector<int> index(1);
          std::vector<float> distance(1);
          if (tree.nearestKSearch(pcl::PointXYZ(xyz[0], xyz[1], xyz[2]), 1, index, distance) &&
            distance[0] <=
            options.maximum_neighbor_distance_m * options.maximum_neighbor_distance_m)
          {
            supported = true;
          }
        }
        if (supported && compatible_depth) {
          largest.insert(largest.end(), component.begin(), component.end());
        }
      }
    }
    object.mask.setTo(0);
    for (const auto & p : largest) {
      object.mask.at<uint8_t>(p) = 255;
    }
    if (valid_count && largest.size() < valid_count * options.minimum_component_fraction) {
      object.failure_reasons |= reason(GeometryFailure::AMBIGUOUS_SURFACE);
    }
    object.valid_depth_pixel_count = largest.size();
    // Denominator stays the ORIGINAL proposal support: filtering cannot inflate quality.
    object.valid_depth_ratio = object.mask_pixel_count ?
      static_cast<double>(largest.size()) / object.mask_pixel_count : 0;
    object.segmented_pcl.header.frame_id = observation.frame_id();
    object.segmented_pcl.header.stamp =
      static_cast<uint64_t>(observation.sensor_stamp().sec) * 1000000ULL +
      observation.sensor_stamp().nanosec / 1000;
    object.segmented_pcl.is_dense = true;
    for (const auto & p : largest) {
      const auto xyz = points.at<cv::Vec3f>(p);
      object.segmented_pcl.emplace_back(xyz[0], xyz[1], xyz[2]);
    }
    if (!largest.empty()) {
      Eigen::Vector4f center;
      pcl::compute3DCentroid(object.segmented_pcl, center);
      object.centroid.x = center.x();
      object.centroid.y = center.y();
      object.centroid.z = center.z();
      Eigen::Matrix3f covariance;
      pcl::computeCovarianceMatrixNormalized(object.segmented_pcl, center, covariance);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
      if (solver.info() == Eigen::Success && solver.eigenvalues().allFinite() &&
        solver.eigenvalues()[1] > 1e-12F &&
        solver.eigenvalues()[2] > 1.1F * solver.eigenvalues()[1])
      {
        Eigen::Vector3f axis = solver.eigenvectors().col(2).normalized();
        Eigen::Index dominant;
        axis.cwiseAbs().maxCoeff(&dominant);
        if (axis[dominant] < 0) {
          axis = -axis;                         // canonical, repeatable sign
        }
        object.axis.x = axis.x();
        object.axis.y = axis.y();
        object.axis.z = axis.z();
      } else {
        object.failure_reasons |= reason(GeometryFailure::INVALID_ORIENTATION);
        if (solver.info() == Eigen::Success && solver.eigenvalues().allFinite() &&
          solver.eigenvalues()[1] > 1e-12F)
        {
          object.failure_reasons |= reason(GeometryFailure::AMBIGUOUS_ORIENTATION);
        }
      }
      Eigen::Vector3f low = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
      Eigen::Vector3f high = -low;
      for (const auto & p : object.segmented_pcl) {
        low = low.cwiseMin(p.getVector3fMap());
        high = high.cwiseMax(p.getVector3fMap());
      }
      const Eigen::Vector3f extent = high - low;
      object.length = std::max(extent.x(), extent.y());
      object.breadth = std::min(extent.x(), extent.y());
      object.height = extent.z();
      if (options.maximum_visible_extent_m > 0 &&
        extent.norm() > options.maximum_visible_extent_m)
      {
        object.failure_reasons |= reason(GeometryFailure::GEOMETRY_MISMATCH);
      }
    }
    const auto refinement_failures = object.failure_reasons;
    validateLocalizedObject(
      object, depth.cols, depth.rows,
      info->k[0], info->k[4], info->k[2], info->k[5], thresholds);
    object.failure_reasons |= refinement_failures;
    if (refinement_failures) {object.quality = GeometryQuality::INVALID;}
    const uint32_t axis_ambiguity = reason(GeometryFailure::INVALID_ORIENTATION) |
      reason(GeometryFailure::AMBIGUOUS_ORIENTATION);
    if (object.failure_reasons == axis_ambiguity) {object.quality = GeometryQuality::DEGRADED;}
  }
  return result;
}
}  // namespace EPD
