// Copyright 2026 Advanced Remanufacturing and Technology Centre
// Licensed under the Apache License, Version 2.0

#ifndef EPD_UTILS_LIB__TEMPORAL_TRACKER_HPP_
#define EPD_UTILS_LIB__TEMPORAL_TRACKER_HPP_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "sensor_msgs/msg/region_of_interest.hpp"

namespace EPD
{
enum class TrackLifecycle : uint8_t {TENTATIVE, CONFIRMED, LOST};

struct TrackingDetection
{
  std::string name;
  sensor_msgs::msg::RegionOfInterest roi;
  geometry_msgs::msg::Point centroid;
  bool geometry_valid{false};
  float detector_confidence{0.0F};
};

struct TrackAssignment
{
  uint64_t track_id{0};
  TrackLifecycle lifecycle{TrackLifecycle::TENTATIVE};
  float detector_confidence{0.0F};
  float tracking_confidence{0.0F};
  std::optional<geometry_msgs::msg::Vector3> velocity;
};

struct TrackerThresholds
{
  double minimum_iou{0.20};
  double maximum_roi_centroid_distance_px{100.0};
  double maximum_3d_distance_m{0.25};
  uint32_t confirmation_hits{2};
  uint32_t maximum_missed_observations{3};
  size_t maximum_active_tracks{64};
  double minimum_velocity_dt_s{0.001};
  double maximum_velocity_dt_s{5.0};
};

struct TrackerMetrics
{
  uint64_t tracks_created{0};
  uint64_t tracks_confirmed{0};
  uint64_t tracks_lost{0};
  uint64_t tracks_expired{0};
  uint64_t associations_matched{0};
  uint64_t associations_rejected{0};
  uint64_t duplicate_update_attempts{0};
  uint64_t out_of_order_observations{0};
  uint64_t id_switches{0};
  size_t active_tracks{0};
  size_t max_active_tracks{0};
  size_t geometry_valid_tracks{0};
  size_t two_d_only_tracks{0};
  int64_t last_processing_latency_us{0};
  uint64_t latest_track_observation_id{0};
  builtin_interfaces::msg::Time latest_track_stamp;
};

class TemporalTracker
{
public:
  explicit TemporalTracker(TrackerThresholds thresholds = TrackerThresholds())
  : thresholds_(thresholds) {}

  std::vector<TrackAssignment> update(
    uint64_t observation_id, const builtin_interfaces::msg::Time & stamp,
    const std::vector<TrackingDetection> & detections)
  {
    const auto started = std::chrono::steady_clock::now();
    if (observation_id < last_observation_id_) {
      ++metrics_.out_of_order_observations;
      return {};
    }
    if (observation_id == last_observation_id_ && observation_id != 0) {
      ++metrics_.duplicate_update_attempts;
      return {};
    }
    last_observation_id_ = observation_id;

    std::vector<TrackAssignment> assignments(detections.size());
    std::vector<bool> track_used(tracks_.size(), false);
    std::vector<bool> detection_used(detections.size(), false);
    struct Candidate {size_t detection; size_t track; double score;};
    std::vector<Candidate> candidates;
    for (size_t d = 0; d < detections.size(); ++d) {
      if (!validDetection(detections[d])) {continue;}
      for (size_t t = 0; t < tracks_.size(); ++t) {
        if (detections[d].name != tracks_[t].name) {
          ++metrics_.associations_rejected;
          continue;
        }
        const double overlap = iou(detections[d].roi, tracks_[t].roi);
        const double roi_distance = roiCentroidDistance(detections[d].roi, tracks_[t].roi);
        const bool has_3d = detections[d].geometry_valid && tracks_[t].geometry_valid;
        const double distance_3d = has_3d ? pointDistance(
          detections[d].centroid,
          tracks_[t].centroid) : 0.0;
        if ((overlap < thresholds_.minimum_iou &&
          roi_distance > thresholds_.maximum_roi_centroid_distance_px) ||
          (has_3d && distance_3d > thresholds_.maximum_3d_distance_m))
        {
          ++metrics_.associations_rejected;
          continue;
        }
        candidates.push_back({d, t, overlap - roi_distance / 1000.0 - distance_3d});
      }
    }
    std::sort(
      candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
        if (a.score != b.score) {return a.score > b.score;}
        if (a.track != b.track) {return a.track < b.track;}
        return a.detection < b.detection;
      });
    for (const auto & candidate : candidates) {
      if (detection_used[candidate.detection] || track_used[candidate.track]) {continue;}
      detection_used[candidate.detection] = true;
      track_used[candidate.track] = true;
      assignments[candidate.detection] = updateTrack(
        tracks_[candidate.track], observation_id, stamp, detections[candidate.detection]);
      ++metrics_.associations_matched;
    }
    for (size_t t = tracks_.size(); t-- > 0; ) {
      if (track_used[t]) {continue;}
      ++tracks_[t].missed;
      tracks_[t].tracking_confidence = std::max(
        0.0F, tracks_[t].tracking_confidence - 0.25F);
      if (tracks_[t].lifecycle != TrackLifecycle::LOST) {
        tracks_[t].lifecycle = TrackLifecycle::LOST;
        ++metrics_.tracks_lost;
      }
      if (tracks_[t].missed > thresholds_.maximum_missed_observations) {
        tracks_.erase(tracks_.begin() + t);
        track_used.erase(track_used.begin() + t);
        ++metrics_.tracks_expired;
      }
    }
    for (size_t d = 0; d < detections.size(); ++d) {
      if (detection_used[d] || !validDetection(detections[d]) ||
        tracks_.size() >= thresholds_.maximum_active_tracks) {continue;}
      Track track;
      track.id = next_track_id_++;
      track.name = detections[d].name;
      track.roi = detections[d].roi;
      track.centroid = detections[d].centroid;
      track.geometry_valid = detections[d].geometry_valid;
      track.latest_detector_confidence = detections[d].detector_confidence;
      track.tracking_confidence = 0.25F;
      track.first_observation_id = observation_id;
      track.last_observation_id = observation_id;
      track.first_stamp = stamp;
      track.last_stamp = stamp;
      track.hits = 1;
      tracks_.push_back(track);
      assignments[d].track_id = track.id;
      assignments[d].lifecycle = track.lifecycle;
      assignments[d].detector_confidence = track.latest_detector_confidence;
      assignments[d].tracking_confidence = track.tracking_confidence;
      ++metrics_.tracks_created;
    }
    refreshMetrics(observation_id, stamp, started);
    return assignments;
  }

  const TrackerMetrics & metrics() const {return metrics_;}
  size_t activeTrackCount() const {return tracks_.size();}

private:
  struct Track
  {
    uint64_t id{0};
    std::string name;
    sensor_msgs::msg::RegionOfInterest roi;
    geometry_msgs::msg::Point centroid;
    bool geometry_valid{false};
    float latest_detector_confidence{0.0F};
    float tracking_confidence{0.0F};
    uint64_t first_observation_id{0};
    uint64_t last_observation_id{0};
    builtin_interfaces::msg::Time first_stamp;
    builtin_interfaces::msg::Time last_stamp;
    uint32_t hits{0};
    uint32_t missed{0};
    bool confirmed{false};
    TrackLifecycle lifecycle{TrackLifecycle::TENTATIVE};
    std::optional<geometry_msgs::msg::Vector3> velocity;
  };

  static bool validDetection(const TrackingDetection & detection)
  {
    return !detection.name.empty() && detection.roi.width > 0 && detection.roi.height > 0;
  }

  static double iou(
    const sensor_msgs::msg::RegionOfInterest & a,
    const sensor_msgs::msg::RegionOfInterest & b)
  {
    const double left = std::max(a.x_offset, b.x_offset);
    const double top = std::max(a.y_offset, b.y_offset);
    const double right = std::min(a.x_offset + a.width, b.x_offset + b.width);
    const double bottom = std::min(a.y_offset + a.height, b.y_offset + b.height);
    const double intersection = std::max(0.0, right - left) * std::max(0.0, bottom - top);
    const double total = static_cast<double>(a.width) * a.height +
      static_cast<double>(b.width) * b.height - intersection;
    return total > 0.0 ? intersection / total : 0.0;
  }

  static double roiCentroidDistance(
    const sensor_msgs::msg::RegionOfInterest & a,
    const sensor_msgs::msg::RegionOfInterest & b)
  {
    const double dx = a.x_offset + a.width / 2.0 - b.x_offset - b.width / 2.0;
    const double dy = a.y_offset + a.height / 2.0 - b.y_offset - b.height / 2.0;
    return std::hypot(dx, dy);
  }

  static double pointDistance(
    const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b)
  {
    return std::hypot(std::hypot(a.x - b.x, a.y - b.y), a.z - b.z);
  }

  TrackAssignment updateTrack(
    Track & track, uint64_t observation_id, const builtin_interfaces::msg::Time & stamp,
    const TrackingDetection & detection)
  {
    TrackAssignment assignment;
    assignment.track_id = track.id;
    const int64_t old_ns = static_cast<int64_t>(track.last_stamp.sec) * 1000000000LL +
      track.last_stamp.nanosec;
    const int64_t new_ns = static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
    const double dt = static_cast<double>(new_ns - old_ns) / 1e9;
    if (track.geometry_valid && detection.geometry_valid &&
      dt >= thresholds_.minimum_velocity_dt_s && dt <= thresholds_.maximum_velocity_dt_s)
    {
      geometry_msgs::msg::Vector3 velocity;
      velocity.x = (detection.centroid.x - track.centroid.x) / dt;
      velocity.y = (detection.centroid.y - track.centroid.y) / dt;
      velocity.z = (detection.centroid.z - track.centroid.z) / dt;
      if (std::isfinite(velocity.x) && std::isfinite(velocity.y) && std::isfinite(velocity.z)) {
        track.velocity = velocity;
      } else {
        track.velocity.reset();
      }
    } else {
      track.velocity.reset();
    }
    track.roi = detection.roi;
    track.centroid = detection.centroid;
    track.geometry_valid = detection.geometry_valid;
    track.latest_detector_confidence = detection.detector_confidence;
    track.tracking_confidence = std::min(1.0F, track.tracking_confidence + 0.25F);
    track.last_observation_id = observation_id;
    track.last_stamp = stamp;
    track.missed = 0;
    ++track.hits;
    if (!track.confirmed && track.hits >= thresholds_.confirmation_hits) {
      track.confirmed = true;
      track.lifecycle = TrackLifecycle::CONFIRMED;
      ++metrics_.tracks_confirmed;
    } else if (track.confirmed) {
      track.lifecycle = TrackLifecycle::CONFIRMED;
    }
    assignment.lifecycle = track.lifecycle;
    assignment.detector_confidence = track.latest_detector_confidence;
    assignment.tracking_confidence = track.tracking_confidence;
    assignment.velocity = track.velocity;
    return assignment;
  }

  void refreshMetrics(
    uint64_t observation_id, const builtin_interfaces::msg::Time & stamp,
    const std::chrono::steady_clock::time_point & started)
  {
    metrics_.active_tracks = tracks_.size();
    metrics_.max_active_tracks = std::max(metrics_.max_active_tracks, tracks_.size());
    metrics_.geometry_valid_tracks = std::count_if(
      tracks_.begin(), tracks_.end(), [](const Track & track) {return track.geometry_valid;});
    metrics_.two_d_only_tracks = tracks_.size() - metrics_.geometry_valid_tracks;
    metrics_.latest_track_observation_id = observation_id;
    metrics_.latest_track_stamp = stamp;
    metrics_.last_processing_latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - started).count();
  }

  TrackerThresholds thresholds_;
  std::vector<Track> tracks_;
  TrackerMetrics metrics_;
  uint64_t next_track_id_{1};
  uint64_t last_observation_id_{0};
};
}  // namespace EPD

#endif  // EPD_UTILS_LIB__TEMPORAL_TRACKER_HPP_
