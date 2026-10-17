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

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "epd_utils_lib/class_selection.hpp"

TEST(ClassSelection, EmptySelectionKeepsEveryClass)
{
  const std::vector<std::string> none;
  EXPECT_TRUE(EPD::isClassSelected(none, "banana"));
  EXPECT_TRUE(EPD::isClassSelected(none, "mouse"));
}

TEST(ClassSelection, OnlySelectedClassesPass)
{
  // Scene A of the 2026-10-16 D435i evidence: apple, two bananas and a mouse.
  const std::vector<std::string> fruit{"banana", "apple"};
  EXPECT_TRUE(EPD::isClassSelected(fruit, "banana"));
  EXPECT_TRUE(EPD::isClassSelected(fruit, "apple"));
  EXPECT_FALSE(EPD::isClassSelected(fruit, "mouse"));
  EXPECT_FALSE(EPD::isClassSelected(fruit, ""));
  EXPECT_FALSE(EPD::isClassSelected(fruit, "Banana"));
}

TEST(ClassSelection, UnknownNamesAreReported)
{
  const std::vector<std::string> labels{"person", "banana", "apple", "mouse"};
  EXPECT_TRUE(EPD::unknownSelectedClasses({"banana", "apple"}, labels).empty());
  EXPECT_EQ(
    EPD::unknownSelectedClasses({"bannana", "apple", "pear"}, labels),
    (std::vector<std::string>{"bannana", "pear"}));
}
