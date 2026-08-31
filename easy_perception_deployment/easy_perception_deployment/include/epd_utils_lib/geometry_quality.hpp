// Copyright 2026 Advanced Remanufacturing and Technology Centre
// Licensed under the Apache License, Version 2.0

#ifndef EPD_UTILS_LIB__GEOMETRY_QUALITY_HPP_
#define EPD_UTILS_LIB__GEOMETRY_QUALITY_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "epd_utils_lib/message_utils.hpp"

namespace EPD
{
struct GeometryThresholds
{
  size_t minimum_mask_pixels{16};
  size_t minimum_depth_pixels{12};
  double minimum_valid_depth_ratio{0.20};
  size_t minimum_cloud_points{12};
};

inline uint32_t reason(GeometryFailure value)
{
  return static_cast<uint32_t>(value);
}

inline bool finitePoint(const geometry_msgs::msg::Point & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

inline bool finiteAxis(const geometry_msgs::msg::Vector3 & axis)
{
  return std::isfinite(axis.x) && std::isfinite(axis.y) && std::isfinite(axis.z);
}

inline bool validIntrinsics(double fx, double fy, double ppx, double ppy)
{
  return std::isfinite(fx) && std::isfinite(fy) && std::isfinite(ppx) &&
         std::isfinite(ppy) && fx > 0.0 && fy > 0.0;
}

inline bool roiInsideImage(
  const sensor_msgs::msg::RegionOfInterest & roi, uint32_t width, uint32_t height)
{
  return roi.width > 0 && roi.height > 0 && roi.x_offset < width && roi.y_offset < height &&
         static_cast<uint64_t>(roi.x_offset) + roi.width <= width &&
         static_cast<uint64_t>(roi.y_offset) + roi.height <= height;
}

inline bool populateMaskedDepthCentroid(
  LocalizedObject & object, const cv::Mat & binary_mask, const cv::Mat & depth_m,
  double fx, double fy, double ppx, double ppy)
{
  if (!validIntrinsics(fx, fy, ppx, ppy) || binary_mask.empty() || depth_m.empty() ||
    binary_mask.size() != depth_m.size() || depth_m.type() != CV_32FC1)
  {
    return false;
  }
  object.mask_pixel_count = 0;
  object.valid_depth_pixel_count = 0;
  object.segmented_pcl.clear();
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  for (int row = 0; row < binary_mask.rows; ++row) {
    for (int col = 0; col < binary_mask.cols; ++col) {
      if (binary_mask.at<uint8_t>(row, col) == 0) {
        continue;
      }
      ++object.mask_pixel_count;
      const float z = depth_m.at<float>(row, col);
      if (!std::isfinite(z) || z <= 0.0F) {
        continue;
      }
      const float x = static_cast<float>((col - ppx) / fx * z);
      const float y = static_cast<float>((row - ppy) / fy * z);
      if (!std::isfinite(x) || !std::isfinite(y)) {
        continue;
      }
      object.segmented_pcl.emplace_back(x, y, z);
      sx += x;
      sy += y;
      sz += z;
      ++object.valid_depth_pixel_count;
    }
  }
  object.valid_depth_ratio = object.mask_pixel_count == 0 ? 0.0 :
    static_cast<double>(object.valid_depth_pixel_count) / object.mask_pixel_count;
  if (object.valid_depth_pixel_count == 0) {
    return false;
  }
  const double count = static_cast<double>(object.valid_depth_pixel_count);
  object.centroid.x = sx / count;
  object.centroid.y = sy / count;
  object.centroid.z = sz / count;
  return finitePoint(object.centroid);
}

inline GeometryQuality validateLocalizedObject(
  LocalizedObject & object, uint32_t image_width, uint32_t image_height,
  double fx, double fy, double ppx, double ppy,
  const GeometryThresholds & thresholds = GeometryThresholds())
{
  uint32_t failures = 0;
  if (!validIntrinsics(fx, fy, ppx, ppy)) {
    failures |= reason(GeometryFailure::INVALID_INTRINSICS);
  }
  if (!roiInsideImage(object.roi, image_width, image_height)) {
    failures |= reason(GeometryFailure::INVALID_ROI);
  }
  if (object.mask.empty() || object.mask.cols != static_cast<int>(object.roi.width) ||
    object.mask.rows != static_cast<int>(object.roi.height) ||
    object.mask_pixel_count < thresholds.minimum_mask_pixels)
  {
    failures |= reason(GeometryFailure::INVALID_MASK);
  }
  if (object.valid_depth_pixel_count < thresholds.minimum_depth_pixels ||
    !std::isfinite(object.valid_depth_ratio) ||
    object.valid_depth_ratio < thresholds.minimum_valid_depth_ratio)
  {
    failures |= reason(GeometryFailure::INSUFFICIENT_DEPTH);
  }
  if (object.segmented_pcl.size() < thresholds.minimum_cloud_points) {
    failures |= reason(GeometryFailure::EMPTY_CLOUD);
  }
  for (const auto & point : object.segmented_pcl.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      failures |= reason(GeometryFailure::NONFINITE_GEOMETRY);
      break;
    }
  }
  if (!finitePoint(object.centroid)) {
    failures |= reason(GeometryFailure::NONFINITE_GEOMETRY);
  }
  if (!std::isfinite(object.length) || !std::isfinite(object.breadth) ||
    !std::isfinite(object.height) || object.length <= 0.0F || object.breadth <= 0.0F ||
    object.height <= 0.0F)
  {
    failures |= reason(GeometryFailure::INVALID_DIMENSIONS);
  }
  const double axis_norm = std::sqrt(
    object.axis.x * object.axis.x + object.axis.y * object.axis.y + object.axis.z * object.axis.z);
  if (!finiteAxis(object.axis) || !std::isfinite(axis_norm) || axis_norm < 0.999 ||
    axis_norm > 1.001)
  {
    failures |= reason(GeometryFailure::INVALID_ORIENTATION);
  }
  object.failure_reasons = failures;
  object.quality = failures == 0 ? GeometryQuality::VALID :
    (failures == reason(GeometryFailure::INSUFFICIENT_DEPTH) ?
    GeometryQuality::DEGRADED : GeometryQuality::INVALID);
  return object.quality;
}
}  // namespace EPD

#endif  // EPD_UTILS_LIB__GEOMETRY_QUALITY_HPP_
