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

// Bounded, offline adapter for sealed 512x512 development masks/depth. No labels.
#include <jsoncpp/json/json.h>
#include <fstream>
#include <iostream>
#include "epd_utils_lib/hybrid_rgbd.hpp"

static void readExact(const char * path, void * data, size_t bytes)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream.read(static_cast<char *>(data), bytes) || stream.peek() != EOF) {
    throw std::runtime_error("wrong input length");
  }
}

int main(int argc, char ** argv)
{
  try {
    if (argc != 10) {throw std::runtime_error("depth masks output count fx fy ppx ppy max_depth");}
    const int count = std::stoi(argv[4]);
    if (count < 0 || count > 12) {throw std::runtime_error("instance cap");}
    cv::Mat depth(512, 512, CV_32FC1);
    readExact(argv[1], depth.data, depth.total() * sizeof(float));
    std::vector<uint8_t> masks(static_cast<size_t>(count) * 512 * 512);
    readExact(argv[2], masks.data(), masks.size());
    auto rgb = std::make_shared<sensor_msgs::msg::Image>();
    auto dep = std::make_shared<sensor_msgs::msg::Image>();
    auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
    rgb->width = dep->width = info->width = 512;
    rgb->height = dep->height = info->height = 512;
    rgb->header.frame_id = dep->header.frame_id = info->header.frame_id = "offline_camera_optical";
    rgb->header.stamp.sec = dep->header.stamp.sec = info->header.stamp.sec = 1;
    info->k = {std::stod(argv[5]), 0, std::stod(argv[7]),
      0, std::stod(argv[6]), std::stod(argv[8]), 0, 0, 1};
    EPD::EPDObjectDetection detections(0);
    for (int i = 0; i < count; ++i) {
      cv::Mat binary(512, 512, CV_8UC1, masks.data() + i * 512 * 512);
      cv::Mat probability;
      binary.convertTo(probability, CV_32FC1);  // exported boolean bytes are 0/1
      detections.bboxes.push_back({0, 0, 512, 512});
      detections.masks.push_back(probability);
      detections.classIndices.push_back(0);
      detections.scores.push_back(1);  // not used for refinement, prefiltered export
    }
    EPD::SynchronizationMetadata sync;
    sync.synchronized = sync.exact_sensor_stamp = true;
    EPD::Observation observation(1, "offline", rgb, dep, info, sync);
    EPD::HybridRgbdOptions options;
    options.maximum_depth_m = std::stod(argv[9]);
    const auto result = EPD::refineRgbd(detections, depth, observation, {"part"}, options);
    Json::Value summary(Json::arrayValue);
    std::ofstream output(std::string(argv[3]) + ".masks.u8", std::ios::binary);
    for (const auto & object : result.objects) {
      cv::Mat mask = object.mask.empty() ? cv::Mat::zeros(512, 512, CV_8UC1) : object.mask;
      output.write(reinterpret_cast<const char *>(mask.data), mask.total());
      Json::Value row;
      row["valid"] = object.quality == EPD::GeometryQuality::VALID;
      row["quality"] = object.quality == EPD::GeometryQuality::VALID ? "valid" :
        (object.quality == EPD::GeometryQuality::DEGRADED ? "ambiguous" : "invalid");
      row["reasons"] = EPD::failureNames(object.failure_reasons);
      row["retained_fraction"] = object.valid_depth_ratio;
      row["points"] = Json::UInt64(object.segmented_pcl.size());
      row["centroid_m"].append(object.centroid.x);
      row["centroid_m"].append(object.centroid.y);
      row["centroid_m"].append(object.centroid.z);
      row["visible_extents_m"].append(object.length);
      row["visible_extents_m"].append(object.breadth);
      row["visible_extents_m"].append(object.height);
      summary.append(row);
    }
    if (!output) {throw std::runtime_error("mask write failed");}
    std::ofstream json(std::string(argv[3]) + ".json");
    json << summary;
    if (!json) {throw std::runtime_error("summary write failed");}
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
