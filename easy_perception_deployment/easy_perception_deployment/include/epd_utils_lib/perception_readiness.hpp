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

#ifndef EPD_UTILS_LIB__PERCEPTION_READINESS_HPP_
#define EPD_UTILS_LIB__PERCEPTION_READINESS_HPP_

#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "epd_utils_lib/geometry_quality.hpp"

namespace EPD
{
// Detection-level readiness. Identity (class, confidence, 2D mask) is reported
// independently of 3D validity; only GRASP_CANDIDATE objects are published on
// the valid-pose outputs. GRASP_CANDIDATE never implies collision-free motion.
enum class PerceptionState : uint8_t
{
  REJECTED,         // 2D evidence unusable or contested; identity still reported
  DETECTED,         // class + confidence + owned 2D mask, no validated 3D position
  LOCALISED,        // validated position and visible surface, orientation not valid
  POSE_VALID,       // validated orientation, but not published (e.g. output cap, no track)
  GRASP_CANDIDATE   // published on the valid-pose output for downstream planning
};

inline const char * stateName(PerceptionState state)
{
  switch (state) {
    case PerceptionState::REJECTED: return "REJECTED";
    case PerceptionState::DETECTED: return "DETECTED";
    case PerceptionState::LOCALISED: return "LOCALISED";
    case PerceptionState::POSE_VALID: return "POSE_VALID";
    case PerceptionState::GRASP_CANDIDATE: return "GRASP_CANDIDATE";
  }
  return "REJECTED";
}

inline uint32_t orientationFailures()
{
  return reason(GeometryFailure::INVALID_ORIENTATION) |
         reason(GeometryFailure::AMBIGUOUS_ORIENTATION);
}

// Failures that make the 2D detection itself unusable or contested.
inline uint32_t detectionFailures()
{
  return reason(GeometryFailure::INVALID_CONFIGURATION) |
         reason(GeometryFailure::UNSYNCHRONIZED_INPUT) |
         reason(GeometryFailure::FRAME_MISMATCH) |
         reason(GeometryFailure::INVALID_INTRINSICS) |
         reason(GeometryFailure::INVALID_ROI) |
         reason(GeometryFailure::INVALID_MASK) |
         reason(GeometryFailure::OVERLAPPING_OWNERSHIP);
}

// `published` is true only when the object was emitted on the valid-pose output.
inline PerceptionState classifyPerceptionState(const LocalizedObject & object, bool published)
{
  const auto failures = object.failure_reasons;
  if (object.name.empty() || (failures & detectionFailures()) || object.mask_pixel_count == 0) {
    return PerceptionState::REJECTED;
  }
  if (object.quality == GeometryQuality::VALID && failures == 0) {
    return published ? PerceptionState::GRASP_CANDIDATE : PerceptionState::POSE_VALID;
  }
  // Every non-orientation geometry check passed: the measured position stands.
  if ((failures & orientationFailures()) && (failures & ~orientationFailures()) == 0 &&
    finitePoint(object.centroid) && !object.segmented_pcl.empty())
  {
    return PerceptionState::LOCALISED;
  }
  return PerceptionState::DETECTED;
}

struct ReadinessRecord
{
  size_t detection_index{0};
  PerceptionState state{PerceptionState::REJECTED};
  std::string name;
  float confidence{0.0F};
  sensor_msgs::msg::RegionOfInterest roi;
  size_t mask_pixel_count{0};
  size_t valid_depth_pixel_count{0};
  uint32_t failure_reasons{0};
  std::string note;  // non-geometry reason, e.g. output cap or missing track
  uint64_t track_id{0};
  geometry_msgs::msg::Point centroid;
  geometry_msgs::msg::Vector3 axis;
};

inline ReadinessRecord makeReadinessRecord(
  size_t index, const LocalizedObject & object, bool published, const std::string & note = "")
{
  ReadinessRecord record;
  record.detection_index = index;
  record.state = classifyPerceptionState(object, published);
  record.name = object.name;
  record.confidence = object.confidence;
  record.roi = object.roi;
  record.mask_pixel_count = object.mask_pixel_count;
  record.valid_depth_pixel_count = object.valid_depth_pixel_count;
  record.failure_reasons = object.failure_reasons;
  record.note = note;
  // Never report geometry beyond what the state validates.
  if (record.state == PerceptionState::LOCALISED || record.state == PerceptionState::POSE_VALID ||
    record.state == PerceptionState::GRASP_CANDIDATE)
  {
    record.centroid = object.centroid;
  }
  if (record.state == PerceptionState::POSE_VALID ||
    record.state == PerceptionState::GRASP_CANDIDATE)
  {
    record.axis = object.axis;
  }
  return record;
}

// One DiagnosticStatus per detector proposal, stamped with the source sensor stamp.
inline diagnostic_msgs::msg::DiagnosticArray makeReadinessReport(
  const std::vector<ReadinessRecord> & records, const std_msgs::msg::Header & source_header,
  uint64_t observation_id, const std::string & hardware_id)
{
  diagnostic_msgs::msg::DiagnosticArray report;
  report.header = source_header;
  for (const auto & record : records) {
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "easy_perception_deployment/detection/" + std::to_string(record.detection_index);
    status.hardware_id = hardware_id;
    status.message = stateName(record.state);
    switch (record.state) {
      case PerceptionState::GRASP_CANDIDATE:
      case PerceptionState::POSE_VALID:
        status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        break;
      case PerceptionState::REJECTED:
        status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
        break;
      default:
        status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
    }
    const auto add = [&status](const std::string & key, const std::string & value) {
        diagnostic_msgs::msg::KeyValue item;
        item.key = key;
        item.value = value;
        status.values.push_back(std::move(item));
      };
    const bool positioned = record.state != PerceptionState::REJECTED &&
      record.state != PerceptionState::DETECTED;
    const bool oriented = record.state == PerceptionState::POSE_VALID ||
      record.state == PerceptionState::GRASP_CANDIDATE;
    std::ostringstream roi, centroid, axis;
    roi << record.roi.x_offset << ',' << record.roi.y_offset << ',' << record.roi.width << ',' <<
      record.roi.height;
    centroid << record.centroid.x << ',' << record.centroid.y << ',' << record.centroid.z;
    axis << record.axis.x << ',' << record.axis.y << ',' << record.axis.z;
    add("state", stateName(record.state));
    add("class", record.name);
    add("confidence", std::to_string(record.confidence));
    add("roi", roi.str());
    add("mask_pixels", std::to_string(record.mask_pixel_count));
    add("valid_depth_pixels", std::to_string(record.valid_depth_pixel_count));
    add("position_valid", positioned ? "true" : "false");
    add("orientation_valid", oriented ? "true" : "false");
    add("centroid_m", positioned ? centroid.str() : "");
    add("axis", oriented ? axis.str() : "");
    add("reasons", failureNames(record.failure_reasons));
    add("note", record.note);
    add("track_id", record.track_id ? std::to_string(record.track_id) : "");
    add("observation_id", std::to_string(observation_id));
    report.status.push_back(std::move(status));
  }
  return report;
}
}  // namespace EPD
#endif  // EPD_UTILS_LIB__PERCEPTION_READINESS_HPP_
