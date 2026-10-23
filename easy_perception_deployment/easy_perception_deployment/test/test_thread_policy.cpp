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


#include <gtest/gtest.h>
#include "epd_utils_lib/thread_policy.hpp"

TEST(ThreadPolicy, AutoLeavesTwoHardwareThreadsAndCapsAtSix)
{
  EXPECT_EQ(EPD::autoIntraOpThreads(8), 6);   // reference workstation: measured optimum
  EXPECT_EQ(EPD::autoIntraOpThreads(6), 4);
  EXPECT_EQ(EPD::autoIntraOpThreads(32), 6);  // more threads stopped helping at 8 vCPUs
  EXPECT_EQ(EPD::autoIntraOpThreads(2), 1);
  EXPECT_EQ(EPD::autoIntraOpThreads(1), 1);
  EXPECT_EQ(EPD::autoIntraOpThreads(0), 2);   // unknown hardware keeps the historical default
}
