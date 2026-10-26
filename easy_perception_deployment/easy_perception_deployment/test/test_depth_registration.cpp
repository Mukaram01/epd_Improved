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

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>

#include "epd_utils_lib/depth_registration.hpp"

namespace
{
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;

CameraInfo info(const std::string & frame, uint32_t w, uint32_t h, double f = 100.0)
{
  CameraInfo out;
  out.header.frame_id = frame;
  out.width = w;
  out.height = h;
  out.k = {f, 0, (w - 1) / 2.0, 0, f, (h - 1) / 2.0, 0, 0, 1};
  out.d = {0, 0, 0, 0, 0};
  out.distortion_model = "plumb_bob";
  return out;
}

Image depthMm(const std::string & frame, uint32_t w, uint32_t h, uint16_t fill)
{
  Image out;
  out.header.frame_id = frame;
  out.width = w;
  out.height = h;
  out.encoding = "16UC1";
  out.step = w * 2;
  out.data.resize(size_t(out.step) * h);
  for (size_t i = 0; i < size_t(w) * h; ++i) {
    std::memcpy(out.data.data() + i * 2, &fill, 2);
  }
  return out;
}

uint16_t at(const Image & image, uint32_t x, uint32_t y)
{
  uint16_t v;
  std::memcpy(&v, image.data.data() + y * image.step + x * 2, 2);
  return v;
}

void set(Image & image, uint32_t x, uint32_t y, uint16_t v)
{
  std::memcpy(image.data.data() + y * image.step + x * 2, &v, 2);
}

Image::ConstSharedPtr stamped(int32_t sec, uint32_t ns)
{
  auto image = std::make_shared<Image>();
  image->header.stamp.sec = sec;
  image->header.stamp.nanosec = ns;
  return image;
}

CameraInfo::ConstSharedPtr stampedInfo(int32_t sec, uint32_t ns)
{
  auto out = std::make_shared<CameraInfo>();
  out->header.stamp.sec = sec;
  out->header.stamp.nanosec = ns;
  return out;
}
}  // namespace

TEST(DepthRegistration, IdentityExtrinsicsAndIntrinsicsReproduceTheInputExactly)
{
  auto depth = depthMm("depth", 8, 6, 0);
  for (uint32_t y = 0; y < 6; ++y) {
    for (uint32_t x = 0; x < 8; ++x) {
      set(depth, x, y, static_cast<uint16_t>(500 + 10 * x + y));
    }
  }
  set(depth, 3, 2, 0);
  Image out;
  ASSERT_EQ(
    EPD::registerDepthToColor(depth, info("depth", 8, 6), info("color", 8, 6), {}, out),
    "");
  EXPECT_EQ(out.header.frame_id, "color");
  EXPECT_EQ(out.encoding, "16UC1");
  for (uint32_t y = 0; y < 6; ++y) {
    for (uint32_t x = 0; x < 8; ++x) {
      EXPECT_EQ(at(out, x, y), at(depth, x, y)) << x << "," << y;
    }
  }
}

TEST(DepthRegistration, BaselineShiftsPixelsByTheExpectedDisparityAndKeepsColourGeometry)
{
  // 20 mm baseline at 1 m with f=100 px is a 2 px disparity; output takes the colour size.
  auto depth = depthMm("depth", 20, 10, 0);
  set(depth, 9, 4, 1000);
  EPD::RigidTransform t;
  t.translation = {0.02, 0.0, 0.0};
  Image out;
  ASSERT_EQ(
    EPD::registerDepthToColor(
      depth, info("depth", 20, 10), info(
        "color", 30,
        12), t, out), "");
  EXPECT_EQ(out.width, 30U);
  EXPECT_EQ(out.height, 12U);
  // x = (9-9.5)/100 = -0.005 m; colour u = 100*(-0.005+0.02)/1 + 14.5 = 16;
  // v = 100*(-0.005) + 5.5 = 5.
  int written = 0;
  for (uint32_t y = 0; y < 12; ++y) {
    for (uint32_t x = 0; x < 30; ++x) {
      if (at(out, x, y)) {
        ++written;
        EXPECT_EQ(at(out, x, y), 1000);
        EXPECT_EQ(x, 16U);
        EXPECT_EQ(y, 5U);
      }
    }
  }
  EXPECT_EQ(written, 1);
}

TEST(DepthRegistration, NearestSurfaceWinsWhenTwoDepthPixelsLandOnOneColourPixel)
{
  // Pure 20 mm shift: depth pixels 5 (1 m) and 6 (2 m) both land on colour pixel 7.
  auto depth = depthMm("depth", 8, 1, 0);
  set(depth, 2, 0, 500);   // x=(2-3.5)/100*0.5=-0.0075; +0.02 -> u=100*0.0125/0.5+3.5=6
  set(depth, 6, 0, 2000);  // x=(6-3.5)/100*2=0.05;      +0.02 -> u=100*0.07/2+3.5=7
  set(depth, 5, 0, 1000);  // x=0.015;  +0.02 -> u=100*0.035/1+3.5=7
  EPD::RigidTransform t;
  t.translation = {0.02, 0.0, 0.0};
  Image out;
  ASSERT_EQ(EPD::registerDepthToColor(depth, info("depth", 8, 1), info("color", 8, 1), t, out), "");
  EXPECT_EQ(at(out, 6, 0), 500);
  EXPECT_EQ(at(out, 7, 0), 1000);
}

TEST(DepthRegistration, WritesColourFrameDepthRatherThanTheRawDepthValue)
{
  // A 0.1 m translation along the optical axis must change the written range.
  auto depth = depthMm("depth", 4, 4, 800);
  EPD::RigidTransform t;
  t.translation = {0.0, 0.0, 0.1};
  Image out;
  ASSERT_EQ(EPD::registerDepthToColor(depth, info("depth", 4, 4), info("color", 4, 4), t, out), "");
  EXPECT_EQ(at(out, 1, 1), 900);
}

TEST(DepthRegistration, RefusesInputsItCannotRegisterHonestly)
{
  auto depth = depthMm("depth", 4, 4, 800);
  Image out;
  auto distorted = info("color", 4, 4);
  distorted.d[0] = 0.1;
  EXPECT_EQ(
    EPD::registerDepthToColor(depth, info("depth", 4, 4), distorted, {}, out),
    "color_distortion_unsupported");
  EXPECT_EQ(
    EPD::registerDepthToColor(depth, info("other", 4, 4), info("color", 4, 4), {}, out),
    "depth_camera_info_frame_mismatch");
  EXPECT_EQ(
    EPD::registerDepthToColor(depth, info("depth", 5, 4), info("color", 4, 4), {}, out),
    "depth_camera_info_size_mismatch");
  auto bad_k = info("color", 4, 4);
  bad_k.k[0] = 0;
  EXPECT_EQ(
    EPD::registerDepthToColor(
      depth, info(
        "depth", 4,
        4), bad_k, {}, out), "color_invalid_intrinsics");
  depth.encoding = "rgb8";
  EXPECT_EQ(
    EPD::registerDepthToColor(depth, info("depth", 4, 4), info("color", 4, 4), {}, out),
    "unsupported_depth_encoding");
}

TEST(RgbdPairer, PairsTheNearestDepthWithinToleranceAndTheExactColourInfo)
{
  EPD::RgbdPairer pairer(20000000);  // 20 ms
  EXPECT_FALSE(pairer.addDepth(stamped(10, 0)));
  EXPECT_FALSE(pairer.addColorInfo(stampedInfo(10, 5000000)));
  const auto triplet = pairer.addColor(stamped(10, 5000000));
  ASSERT_TRUE(triplet);
  EXPECT_EQ(triplet->skew_ns, -5000000);
  EXPECT_EQ(triplet->depth->header.stamp.nanosec, 0U);
  EXPECT_EQ(triplet->color_info->header.stamp.nanosec, 5000000U);
}

TEST(RgbdPairer, WaitsForLateDepthAndInfoThenPairs)
{
  EPD::RgbdPairer pairer(20000000);
  EXPECT_FALSE(pairer.addColor(stamped(10, 0)));
  EXPECT_FALSE(pairer.addDepth(stamped(10, 3000000)));
  ASSERT_TRUE(pairer.addColorInfo(stampedInfo(10, 0)));
  EXPECT_EQ(pairer.unpaired(), 0U);
}

TEST(RgbdPairer, LatestColourFrameWinsAndOlderPendingFramesAreDropped)
{
  EPD::RgbdPairer pairer(20000000);
  EXPECT_FALSE(pairer.addColor(stamped(10, 0)));
  EXPECT_FALSE(pairer.addColor(stamped(10, 50000000)));
  EXPECT_FALSE(pairer.addColor(stamped(10, 100000000)));
  EXPECT_EQ(pairer.superseded(), 2U);
  EXPECT_FALSE(pairer.addColorInfo(stampedInfo(10, 100000000)));
  const auto triplet = pairer.addDepth(stamped(10, 101000000));
  ASSERT_TRUE(triplet);
  EXPECT_EQ(triplet->color->header.stamp.nanosec, 100000000U);
}

TEST(RgbdPairer, DropsAColourFrameWhoseDepthCanNoLongerArriveWithinTolerance)
{
  EPD::RgbdPairer pairer(20000000);
  EXPECT_FALSE(pairer.addColorInfo(stampedInfo(10, 0)));
  EXPECT_FALSE(pairer.addColor(stamped(10, 0)));
  // Only a depth frame 60 ms later exists: out of tolerance, and nothing closer can still come.
  EXPECT_FALSE(pairer.addDepth(stamped(10, 60000000)));
  EXPECT_EQ(pairer.unpaired(), 1U);
  // Nothing remains pending.
  EXPECT_FALSE(pairer.addDepth(stamped(10, 1000000)));
}
