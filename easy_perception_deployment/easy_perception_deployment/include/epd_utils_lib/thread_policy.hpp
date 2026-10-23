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

#ifndef EPD_UTILS_LIB__THREAD_POLICY_HPP_
#define EPD_UTILS_LIB__THREAD_POLICY_HPP_

#include <algorithm>

namespace EPD
{
// "auto" intra-op threads: leave two hardware threads for the camera driver, ROS
// executors and the RGB-D/tracking stages. Measured on the reference 8-vCPU workstation,
// MaskRCNN-10 at 1280x736 stops improving beyond 6 threads, so cap there.
inline int autoIntraOpThreads(unsigned int hardware_threads)
{
  if (hardware_threads == 0) {return 2;}  // unknown: keep the historical default
  return std::clamp(static_cast<int>(hardware_threads) - 2, 1, 6);
}
}  // namespace EPD
#endif  // EPD_UTILS_LIB__THREAD_POLICY_HPP_
