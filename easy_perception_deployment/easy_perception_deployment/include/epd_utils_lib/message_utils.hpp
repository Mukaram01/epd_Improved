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

#ifndef EPD_UTILS_LIB__MESSAGE_UTILS_HPP_
#define EPD_UTILS_LIB__MESSAGE_UTILS_HPP_

#include <string>
#include <vector>
#include <cstdint>
#include "opencv2/opencv.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/region_of_interest.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace EPD
{
enum class GeometryQuality : uint8_t {VALID, DEGRADED, INVALID};

enum class GeometryFailure : uint32_t
{
  NONE = 0,
  INVALID_INTRINSICS = 1U << 0,
  INVALID_ROI = 1U << 1,
  INVALID_MASK = 1U << 2,
  INSUFFICIENT_DEPTH = 1U << 3,
  EMPTY_CLOUD = 1U << 4,
  NONFINITE_GEOMETRY = 1U << 5,
  INVALID_DIMENSIONS = 1U << 6,
  INVALID_ORIENTATION = 1U << 7,
  FRAME_MISMATCH = 1U << 8,
  GEOMETRY_EXCEPTION = 1U << 9
};

/*! \class EPDObjectDetection
    \brief An Easy Perception Deployment (EPD) ObjectDetection class object.
    This object functions as a transient container of inference results to
    transport them for processing in by EasyPerceptionDeployment class object.
*/
class EPDObjectDetection
{
public:
  /*! \brief A vector of bounding boxes with xmin, ymin, xmax, ymax.*/
  std::vector<std::array<int, 4>> bboxes;
  /*! \brief A vector of indices that indicate the numerical identities of
  corresponding bounding boxes of the same index.
  */
  std::vector<uint64_t> classIndices;
  /*! \brief A vector of indices that indicate the float confidence scores of
  corresponding bounding boxes of the same index.
  */
  std::vector<float> scores;
  /*! \brief A vector of image-encoded 32FC1 greyscale masks for P3 results only.
  */
  std::vector<cv::Mat> masks;

  /*! \brief Returns the number of detections. Always equal to bboxes.size().*/
  size_t size() const {return bboxes.size();}

  /*! \brief A Constructor function. This object can only be called a known
  size to minimize memory use for storage.*/
  explicit EPDObjectDetection(size_t input_size)
  {
    bboxes.reserve(input_size);
    classIndices.reserve(input_size);
    scores.reserve(input_size);
    masks.reserve(input_size);
  }
};

struct LocalizedObject
{
  std::string name;
  float confidence{0.0F};
  sensor_msgs::msg::RegionOfInterest roi;
  cv::Mat mask;
  geometry_msgs::msg::Point centroid;
  float length{0.0F};
  float breadth{0.0F};
  float height{0.0F};
  pcl::PointCloud<pcl::PointXYZ> segmented_pcl;
  geometry_msgs::msg::Vector3 axis;
  GeometryQuality quality{GeometryQuality::INVALID};
  uint32_t failure_reasons{static_cast<uint32_t>(GeometryFailure::NONE)};
  uint64_t source_observation_id{0};
  builtin_interfaces::msg::Time source_sensor_stamp;
  std::string source_frame;
  size_t mask_pixel_count{0};
  size_t valid_depth_pixel_count{0};
  double valid_depth_ratio{0.0};
};

class EPDObjectLocalization
{
public:
  std::vector<LocalizedObject> objects;

  /*! \brief Returns the number of localized objects. Equivalent to objects.size().*/
  size_t size() const {return objects.size();}

  explicit EPDObjectLocalization(size_t input_size)
  {
    objects.resize(input_size);
  }
};

class EPDObjectTracking
{
public:
  std::vector<std::string> object_ids;
  std::vector<LocalizedObject> objects;

  /*! \brief Returns the number of tracked objects. Equivalent to objects.size().*/
  size_t size() const {return objects.size();}

  explicit EPDObjectTracking(size_t input_size)
  {
    objects.resize(input_size);
    object_ids.resize(input_size, "untracked");
  }
};

struct LabelledRect2d
{
  std::string obj_tag;
  cv::Rect2d obj_bounding_box;
  int missed_frames = 0;
};

}  // namespace EPD

#endif  // EPD_UTILS_LIB__MESSAGE_UTILS_HPP_
