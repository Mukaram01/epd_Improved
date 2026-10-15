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

#ifndef EPD_UTILS_LIB__HYBRID_RGBD_HPP_
#define EPD_UTILS_LIB__HYBRID_RGBD_HPP_

#include <string>
#include <vector>

#include "epd_utils_lib/geometry_quality.hpp"
#include "epd_utils_lib/observation.hpp"

namespace EPD
{
// Startup-only options. Size checks constrain visible geometry, never identity.
struct HybridRgbdOptions
{
  double maximum_neighbor_distance_m{0.015};
  double minimum_component_fraction{0.60};
  double maximum_overlap_fraction{0.25};
  double minimum_depth_m{0.05};
  double maximum_depth_m{2.0};
  double maximum_visible_extent_m{0.0};  // zero disables the optional size check
};

// Input masks are bounding-box-local probabilities, as returned by P3 infer(RGB).
// One output per input preserves detector indexing, including rejected candidates.
EPDObjectLocalization refineRgbd(
  const EPDObjectDetection & detections, const cv::Mat & depth,
  const Observation & observation, const std::vector<std::string> & class_names,
  const HybridRgbdOptions & options = HybridRgbdOptions(),
  const GeometryThresholds & thresholds = GeometryThresholds());
}  // namespace EPD
#endif  // EPD_UTILS_LIB__HYBRID_RGBD_HPP_
