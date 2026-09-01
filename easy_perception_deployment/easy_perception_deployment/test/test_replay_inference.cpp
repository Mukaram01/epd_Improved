// Copyright 2026 Advanced Remanufacturing and Technology Centre
// Licensed under the Apache License, Version 2.0

#include <filesystem>
#include <memory>
#include <string>

#include "cv_bridge/cv_bridge.h"
#include "epd_utils_lib/epd_container.hpp"
#include "epd_utils_lib/observation.hpp"
#include "epd_utils_lib/usecase_config.hpp"
#include "gtest/gtest.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "sensor_msgs/image_encodings.hpp"

TEST(P8ReplayInference, ValidFixtureObservationReachesProductionMaskRcnn)
{
  const std::filesystem::path package(PATH_TO_PACKAGE);
  const auto previous_cwd = std::filesystem::current_path();
  std::filesystem::current_path(package);

  cv::Mat source_rgb = cv::imread((package / "test/colored_img.png").string(), cv::IMREAD_COLOR);
  cv::Mat source_depth = cv::imread(
    (package / "test/depth_img.png").string(), cv::IMREAD_UNCHANGED);
  ASSERT_FALSE(source_rgb.empty());
  ASSERT_FALSE(source_depth.empty());

  cv::Mat rgb;
  cv::Mat depth;
  cv::resize(source_rgb, rgb, cv::Size(320, 240));
  cv::resize(source_depth, depth, cv::Size(320, 240), 0.0, 0.0, cv::INTER_NEAREST);
  depth.convertTo(depth, CV_16UC1);
  ASSERT_TRUE(rgb.isContinuous());
  ASSERT_TRUE(depth.isContinuous());

  std_msgs::msg::Header header;
  header.stamp.sec = 1;
  header.frame_id = "fixture_color_optical_frame";
  auto rgb_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, rgb).toImageMsg();
  auto depth_msg = cv_bridge::CvImage(
    header, sensor_msgs::image_encodings::TYPE_16UC1, depth).toImageMsg();
  auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  info->header = header;
  info->width = 320;
  info->height = 240;
  info->k = {305.1870422363281, 0.0, 161.6538848876953,
    0.0, 304.9342956542969, 117.71758270263672, 0.0, 0.0, 1.0};

  EXPECT_EQ(rgb_msg->width, 320U);
  EXPECT_EQ(rgb_msg->height, 240U);
  EXPECT_EQ(rgb_msg->encoding, sensor_msgs::image_encodings::BGR8);
  EXPECT_EQ(rgb_msg->step, 320U * 3U);
  EXPECT_EQ(rgb_msg->data.size(), 320U * 240U * 3U);
  EXPECT_EQ(depth_msg->step, 320U * 2U);
  EXPECT_EQ(depth_msg->data.size(), 320U * 240U * 2U);

  EPD::LatestObservationStore store;
  EPD::SynchronizationMetadata sync;
  sync.synchronized = true;
  sync.exact_sensor_stamp = true;
  const auto observation = store.publish("fixture_camera", rgb_msg, depth_msg, info, sync);
  ASSERT_NE(observation, nullptr);
  EXPECT_EQ(observation->observation_id(), 1U);

  const auto decoded_rgb = cv_bridge::toCvShare(
    observation->rgb(), sensor_msgs::image_encodings::BGR8)->image;
  const auto decoded_depth = cv_bridge::toCvShare(
    observation->aligned_depth(), sensor_msgs::image_encodings::TYPE_16UC1)->image;
  EXPECT_EQ(decoded_rgb.type(), CV_8UC3);
  EXPECT_EQ(decoded_rgb.rows, 240);
  EXPECT_EQ(decoded_rgb.cols, 320);
  EXPECT_EQ(decoded_rgb.step, 320U * 3U);
  EXPECT_TRUE(decoded_rgb.isContinuous());
  EXPECT_EQ(decoded_depth.type(), CV_16UC1);
  EXPECT_EQ(decoded_depth.step, 320U * 2U);

  EPD::EPDContainer agent;
  agent.useCaseMode = EPD::TRACKING_MODE;
  agent.setFrameDimension(decoded_rgb.cols, decoded_rgb.rows);
  ASSERT_NO_THROW(agent.initORTSessionHandler());
  ASSERT_NE(agent.p3_ort_session, nullptr);
  EXPECT_NO_THROW({
    const auto result = agent.p3_ort_session->infer(
      decoded_rgb, decoded_depth, *info, 1000.0, agent.tracker_type,
      agent.trackers, agent.tracker_logs, agent.tracker_results,
      agent.confidence_threshold);
    (void)result;
  });

  std::filesystem::current_path(previous_cwd);
}
