// Copyright 2026 Easy Perception Deployment contributors
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
//
// Registration of an unaligned (depth-optical-frame) depth image into the
// colour optical frame, plus latest-frame-wins pairing of colour, colour
// CameraInfo and depth. Used by epd_sensor_ingress when the camera publishes
// depth/image_rect_raw instead of aligned_depth_to_color/image_raw.

#ifndef EPD_UTILS_LIB__DEPTH_REGISTRATION_HPP_
#define EPD_UTILS_LIB__DEPTH_REGISTRATION_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <utility>

#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace EPD
{

/*! \brief Rigid transform p_color = R * p_depth + t (metres). */
struct RigidTransform
{
  std::array<double, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::array<double, 3> translation{0, 0, 0};
};

inline int64_t stampNs(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
}

/*! \brief Empty when the CameraInfo can be used as a pinhole model for an image of size w x h. */
inline std::string pinholeError(const sensor_msgs::msg::CameraInfo & info, uint32_t w, uint32_t h)
{
  const auto & k = info.k;
  const bool finite_positive = std::isfinite(k[0]) && std::isfinite(k[4]) &&
    std::isfinite(k[2]) && std::isfinite(k[5]) && k[0] > 0 && k[4] > 0;
  if (!finite_positive) {return "invalid_intrinsics";}
  if (info.width != w || info.height != h) {return "camera_info_size_mismatch";}
  if (info.binning_x > 1 || info.binning_y > 1 || info.roi.x_offset || info.roi.y_offset) {
    return "camera_info_binning_or_roi_unsupported";
  }
  // Registration assumes rectified images; a non-zero model would be silently ignored.
  if (std::any_of(
      info.d.begin(), info.d.end(),
      [](double d) {return !std::isfinite(d) || d != 0;}))
  {
    return "distortion_unsupported";
  }
  return {};
}

/*! \brief Reproject depth (depth optical frame) into the colour optical frame.
 *
 * Each valid depth pixel is deprojected at its two opposite corners, transformed
 * into the colour frame and written to the colour pixels whose centres it covers,
 * keeping the nearest surface (z-buffer), like librealsense's align. Output depth is
 * the colour-frame z in the input encoding (16UC1 millimetres or 32FC1 metres),
 * with the colour image's size and frame; 0 marks no measurement. The caller sets
 * the stamp. Returns an empty string on success or a failure code.
 */
inline std::string registerDepthToColor(
  const sensor_msgs::msg::Image & depth, const sensor_msgs::msg::CameraInfo & depth_info,
  const sensor_msgs::msg::CameraInfo & color_info, const RigidTransform & color_from_depth,
  sensor_msgs::msg::Image & out)
{
  const bool mm = depth.encoding == "16UC1" || depth.encoding == "mono16";
  if (!mm && depth.encoding != "32FC1") {return "unsupported_depth_encoding";}
  const size_t bpp = mm ? 2 : 4;
  if (depth.step < depth.width * bpp || depth.data.size() < size_t(depth.step) * depth.height) {
    return "malformed_depth_image";
  }
  if (depth.is_bigendian) {return "big_endian_depth_unsupported";}
  if (depth.header.frame_id.empty() || depth.header.frame_id != depth_info.header.frame_id) {
    return "depth_camera_info_frame_mismatch";
  }
  if (auto e = pinholeError(depth_info, depth.width, depth.height); !e.empty()) {
    return "depth_" + e;
  }
  if (auto e = pinholeError(color_info, color_info.width, color_info.height); !e.empty()) {
    return "color_" + e;
  }
  if (color_info.width == 0 || color_info.height == 0) {return "color_camera_info_size_mismatch";}

  const double fxd = depth_info.k[0], fyd = depth_info.k[4], cxd = depth_info.k[2],
    cyd = depth_info.k[5];
  const double fxc = color_info.k[0], fyc = color_info.k[4], cxc = color_info.k[2],
    cyc = color_info.k[5];
  const auto & R = color_from_depth.rotation;
  const auto & t = color_from_depth.translation;
  const int W = static_cast<int>(color_info.width), H = static_cast<int>(color_info.height);
  const double scale = mm ? 0.001 : 1.0;

  std::vector<float> zbuf(size_t(W) * H, std::numeric_limits<float>::infinity());
  // Project the depth pixel corner (u, v) at range z to colour pixel coordinates; false if behind.
  const auto project = [&](double u, double v, double z, double & uc, double & vc, double & zc) {
      const double x = (u - cxd) / fxd * z, y = (v - cyd) / fyd * z;
      const double px = R[0] * x + R[1] * y + R[2] * z + t[0];
      const double py = R[3] * x + R[4] * y + R[5] * z + t[1];
      zc = R[6] * x + R[7] * y + R[8] * z + t[2];
      if (!(zc > 0)) {return false;}
      uc = fxc * px / zc + cxc;
      vc = fyc * py / zc + cyc;
      return true;
    };

  for (uint32_t v = 0; v < depth.height; ++v) {
    const uint8_t * row = depth.data.data() + size_t(v) * depth.step;
    for (uint32_t u = 0; u < depth.width; ++u) {
      double z;
      if (mm) {
        uint16_t raw;
        std::memcpy(&raw, row + u * 2, 2);
        z = raw * scale;
      } else {
        float raw;
        std::memcpy(&raw, row + u * 4, 4);
        z = raw;
      }
      if (!std::isfinite(z) || z <= 0) {continue;}
      double u0, v0, z0, u1, v1, z1, uc, vc, zc;
      // The pixel centre's colour-frame z is the value written (what colour-K deprojection needs).
      if (!project(u - 0.5, v - 0.5, z, u0, v0, z0) || !project(u + 0.5, v + 0.5, z, u1, v1, z1) ||
        !project(u, v, z, uc, vc, zc))
      {
        continue;
      }
      // Colour pixels whose centres lie inside the footprint; a footprint smaller than one
      // colour pixel still writes the pixel nearest its centre.
      int xa = static_cast<int>(std::ceil(std::min(u0, u1))),
        xb = static_cast<int>(std::floor(std::max(u0, u1)));
      int ya = static_cast<int>(std::ceil(std::min(v0, v1))),
        yb = static_cast<int>(std::floor(std::max(v0, v1)));
      if (xa > xb) {xa = xb = static_cast<int>(std::lround(uc));}
      if (ya > yb) {ya = yb = static_cast<int>(std::lround(vc));}
      xa = std::max(xa, 0); ya = std::max(ya, 0); xb = std::min(xb, W - 1);
      yb = std::min(yb, H - 1);
      for (int y = ya; y <= yb; ++y) {
        for (int x = xa; x <= xb; ++x) {
          float & cell = zbuf[size_t(y) * W + x];
          cell = std::min(cell, static_cast<float>(zc));
        }
      }
    }
  }

  out.header.frame_id = color_info.header.frame_id;
  out.width = color_info.width;
  out.height = color_info.height;
  out.encoding = depth.encoding == "mono16" ? "16UC1" : depth.encoding;
  out.is_bigendian = 0;
  out.step = static_cast<uint32_t>(W * bpp);
  out.data.assign(size_t(out.step) * H, 0);
  for (size_t i = 0; i < zbuf.size(); ++i) {
    if (!std::isfinite(zbuf[i])) {continue;}
    if (mm) {
      const double value = std::round(zbuf[i] / scale);
      const uint16_t q = value >= 65535.0 ? 0 : static_cast<uint16_t>(value);
      std::memcpy(out.data.data() + i * 2, &q, 2);
    } else {
      std::memcpy(out.data.data() + i * 4, &zbuf[i], 4);
    }
  }
  return {};
}

/*! \brief Latest-frame-wins pairing of colour image, its CameraInfo and the nearest depth frame.
 *
 * Only the newest colour frame is ever pending: an older pending frame is superseded
 * (counted, never processed later), so a slow consumer cannot build a backlog.
 * A depth frame pairs only when |depth stamp - colour stamp| <= max_skew_ns.
 */
class RgbdPairer
{
public:
  using Image = sensor_msgs::msg::Image;
  using Info = sensor_msgs::msg::CameraInfo;

  struct Triplet
  {
    Image::ConstSharedPtr color;
    Info::ConstSharedPtr color_info;
    Image::ConstSharedPtr depth;
    int64_t skew_ns{0};
  };

  explicit RgbdPairer(int64_t max_skew_ns, size_t cache = 8)
  : max_skew_ns_(max_skew_ns), cache_(std::max<size_t>(cache, 2)) {}

  std::optional<Triplet> addColor(Image::ConstSharedPtr msg)
  {
    if (pending_) {++superseded_;}
    pending_ = std::move(msg);
    return tryPair();
  }
  std::optional<Triplet> addColorInfo(Info::ConstSharedPtr msg)
  {
    infos_[stampNs(msg->header.stamp)] = std::move(msg);
    while (infos_.size() > cache_) {infos_.erase(infos_.begin());}
    return tryPair();
  }
  std::optional<Triplet> addDepth(Image::ConstSharedPtr msg)
  {
    depths_.push_back(std::move(msg));
    while (depths_.size() > cache_) {depths_.pop_front();}
    return tryPair();
  }

  uint64_t superseded() const {return superseded_;}
  uint64_t unpaired() const {return unpaired_;}

private:
  std::optional<Triplet> tryPair()
  {
    if (!pending_) {return std::nullopt;}
    const int64_t c = stampNs(pending_->header.stamp);
    const auto info = infos_.find(c);
    Image::ConstSharedPtr best;
    int64_t best_skew = std::numeric_limits<int64_t>::max();
    int64_t newest_depth = std::numeric_limits<int64_t>::min();
    for (const auto & d : depths_) {
      const int64_t s = stampNs(d->header.stamp);
      newest_depth = std::max(newest_depth, s);
      if (std::llabs(s - c) < std::llabs(best_skew)) {best = d; best_skew = s - c;}
    }
    if (best && std::llabs(best_skew) <= max_skew_ns_ && info != infos_.end()) {
      Triplet out{pending_, info->second, best, best_skew};
      pending_.reset();
      return out;
    }
    // A depth frame newer than the tolerance window means no closer one can still arrive.
    const bool depth_passed = !depths_.empty() && newest_depth > c + max_skew_ns_;
    const bool info_passed = !infos_.empty() && infos_.rbegin()->first > c;
    if ((depth_passed && !(best && std::llabs(best_skew) <= max_skew_ns_)) ||
      (info_passed && info == infos_.end()))
    {
      ++unpaired_;
      pending_.reset();
    }
    return std::nullopt;
  }

  int64_t max_skew_ns_;
  size_t cache_;
  Image::ConstSharedPtr pending_;
  std::map<int64_t, Info::ConstSharedPtr> infos_;
  std::deque<Image::ConstSharedPtr> depths_;
  uint64_t superseded_{0};
  uint64_t unpaired_{0};
};

}  // namespace EPD

#endif  // EPD_UTILS_LIB__DEPTH_REGISTRATION_HPP_
