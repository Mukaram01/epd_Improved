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

#ifndef EPD_UTILS_LIB__CLASS_SELECTION_HPP_
#define EPD_UTILS_LIB__CLASS_SELECTION_HPP_

#include <algorithm>
#include <string>
#include <vector>

namespace EPD
{
// Target-object selection for Localization and Tracking (usecase_config
// "class_list"). An empty selection keeps the original behaviour: every class
// may be published. Unselected detections keep their identity in the
// readiness report but are never published on the valid-pose outputs, so an
// unrelated object can never become a grasp candidate.
constexpr const char * kNotSelectedNote = "not_selected";

inline bool isClassSelected(
  const std::vector<std::string> & selected_class_names, const std::string & name)
{
  return selected_class_names.empty() ||
         std::find(selected_class_names.begin(), selected_class_names.end(), name) !=
         selected_class_names.end();
}

// Selected names that do not appear in the model's label list. A typo must be
// a configuration error, not a silent "nothing is ever published".
inline std::vector<std::string> unknownSelectedClasses(
  const std::vector<std::string> & selected_class_names,
  const std::vector<std::string> & label_names)
{
  std::vector<std::string> unknown;
  for (const auto & name : selected_class_names) {
    if (std::find(label_names.begin(), label_names.end(), name) == label_names.end()) {
      unknown.push_back(name);
    }
  }
  return unknown;
}
}  // namespace EPD

#endif  // EPD_UTILS_LIB__CLASS_SELECTION_HPP_
