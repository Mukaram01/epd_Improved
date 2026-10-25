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

#ifndef EPD_UTILS_LIB__LOG_FORMAT_HPP_
#define EPD_UTILS_LIB__LOG_FORMAT_HPP_

namespace EPD
{
// printf-style RCLCPP_* logging: %llu and %lld require exactly these C types.
// Fixed-width aliases (uint64_t is unsigned long on LP64 Linux) would not
// match the format specifiers, so the C types are intentional here.
using LogULongLong = unsigned long long;  // NOLINT(runtime/int)
using LogLongLong = long long;  // NOLINT(runtime/int)
}  // namespace EPD

#endif  // EPD_UTILS_LIB__LOG_FORMAT_HPP_
