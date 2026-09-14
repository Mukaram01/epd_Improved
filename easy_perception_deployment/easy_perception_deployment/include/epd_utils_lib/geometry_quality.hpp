// Copyright 2026 Advanced Remanufacturing and Technology Centre
// Licensed under the Apache License, Version 2.0

#ifndef EPD_UTILS_LIB__GEOMETRY_QUALITY_HPP_
#define EPD_UTILS_LIB__GEOMETRY_QUALITY_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>

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

inline std::string failureNames(uint32_t failures)
{
  struct Entry { GeometryFailure flag; const char * name; };
  static constexpr Entry entries[] = {
    {GeometryFailure::INVALID_INTRINSICS, "invalid_intrinsics"},
    {GeometryFailure::INVALID_ROI, "roi_out_of_bounds"},
    {GeometryFailure::INVALID_MASK, "empty_or_mismatched_mask"},
    {GeometryFailure::INSUFFICIENT_DEPTH, "insufficient_depth"},
    {GeometryFailure::EMPTY_CLOUD, "too_few_segmented_points"},
    {GeometryFailure::NONFINITE_GEOMETRY, "nonfinite_geometry"},
    {GeometryFailure::INVALID_DIMENSIONS, "invalid_dimensions"},
    {GeometryFailure::INVALID_ORIENTATION, "degenerate_axis"},
    {GeometryFailure::FRAME_MISMATCH, "frame_mismatch"},
    {GeometryFailure::GEOMETRY_EXCEPTION, "geometry_exception"},
  };
  std::ostringstream out;
  bool first = true;
  for (const auto & entry : entries) {
    if ((failures & reason(entry.flag)) == 0) continue;
    if (!first) out << ',';
    out << entry.name;
    first = false;
  }
  return first ? "none" : out.str();
}

// Shared by localization and tracking. Depth statistics are measured in the
// detection mask, not inferred from a box centre or a configured plane.
inline std::string geometryDiagnostic(const LocalizedObject & object, const cv::Mat & depth)
{
  std::ostringstream out;
  out << "class=" << object.name << " roi=<" << object.roi.x_offset << ','
      << object.roi.y_offset << ',' << object.roi.width << ',' << object.roi.height
      << "> mask_size=<" << object.mask.cols << ',' << object.mask.rows
      << "> mask_pixels=" << object.mask_pixel_count
      << " valid_depth_pixels=" << object.valid_depth_pixel_count
      << " valid_depth_ratio=" << object.valid_depth_ratio;
  const bool inside = object.roi.width > 0 && object.roi.height > 0 &&
    static_cast<uint64_t>(object.roi.x_offset) + object.roi.width <= static_cast<uint64_t>(depth.cols) &&
    static_cast<uint64_t>(object.roi.y_offset) + object.roi.height <= static_cast<uint64_t>(depth.rows);
  if (!depth.empty() && inside && object.mask.type() == CV_8UC1 &&
    object.mask.cols == static_cast<int>(object.roi.width) &&
    object.mask.rows == static_cast<int>(object.roi.height) &&
    (depth.type() == CV_16UC1 || depth.type() == CV_32FC1))
  {
    std::vector<float> values;
    size_t zero = 0, nonfinite = 0, negative = 0;
    for (int y = 0; y < object.mask.rows; ++y) {
      for (int x = 0; x < object.mask.cols; ++x) {
        if (!object.mask.at<uint8_t>(y, x)) continue;
        const int u = static_cast<int>(object.roi.x_offset) + x;
        const int v = static_cast<int>(object.roi.y_offset) + y;
        const float z = depth.type() == CV_16UC1 ? depth.at<uint16_t>(v,u)*0.001F : depth.at<float>(v,u);
        if (!std::isfinite(z)) ++nonfinite;
        else if (z == 0) ++zero;
        else if (z < 0) ++negative;
        else values.push_back(z);
      }
    }
    out << " depth_encoding=" << (depth.type() == CV_16UC1 ? "16UC1_mm" : "32FC1_m")
        << " zero_depth_pixels=" << zero << " nonfinite_depth_pixels=" << nonfinite
        << " negative_depth_pixels=" << negative;
    if (!values.empty()) {
      const auto bounds = std::minmax_element(values.begin(), values.end());
      const float low = *bounds.first, high = *bounds.second;
      std::nth_element(values.begin(), values.begin()+values.size()/2, values.end());
      out << " min/median/max_depth_m=<" << low << ',' << values[values.size()/2] << ',' << high << '>';
    } else {
      out << " depth_statistics=unavailable_no_positive_depth";
    }
  } else {
    out << " depth_statistics=unavailable_mask_roi_or_encoding_mismatch";
  }
  out << " generated_points=" << object.segmented_pcl.size()
      << " centroid=<" << object.centroid.x << ',' << object.centroid.y << ',' << object.centroid.z
      << "> dimensions=<" << object.length << ',' << object.breadth << ',' << object.height
      << "> axis=<" << object.axis.x << ',' << object.axis.y << ',' << object.axis.z
      << "> geometry_status=" << (object.quality == GeometryQuality::VALID ? "PASS" : "FAIL")
      << " reject_reason=" << failureNames(object.failure_reasons)
      << " source_frame=" << object.source_frame << " observation_id=" << object.source_observation_id;
  return out.str();
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
