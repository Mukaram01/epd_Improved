#!/usr/bin/env python3

# Copyright 2022 Advanced Remanufacturing and Technology Centre
# Copyright 2022 ROS-Industrial Consortium Asia Pacific Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import argparse
import getopt
import json


class EPDConfigurator():

    def __init__(self, start_dirpath, args):

        self.isInEPDPackageRoot(start_dirpath)

        self._path_to_model = ''
        self._path_to_label_list = ''
        self._input_image_topic = ''
        self.visualizeFlag = True
        self.useCPU = True
        self.intra_op_num_threads = 0
        self.publish_detection_segmentation = True

        self.usecase_mode = 0

        self.count_class_list = []
        self.path_to_color_template = ''
        self.color_match_histogram_metric = 'Correlation'
        self.track_type = ''
        self.input_image_topic = ''

        # Check if session_config.json exits.
        self.session_config_filepath = start_dirpath \
            + "/config/session_config.json"
        if os.path.isfile(self.session_config_filepath):
            print("[ config_epd ] - session_config.json FOUND.")
            self.parse_session_config(self.session_config_filepath)
        else:
            print("[ config_epd ] - ERROR. session_config.json MISSING.")
            sys.exit(1)
        # Check if input_image_topic.json exists.
        self.inputimagetopic_config_filepath = start_dirpath \
            + "/config/input_image_topic.json"
        if os.path.isfile(self.inputimagetopic_config_filepath):
            print("[ config_epd ] - input_image_topic.json FOUND.")
            self.parse_inputimagetopic_config(
                self.inputimagetopic_config_filepath)
        else:
            print("[ config_epd ] - ERROR. input_image_topic.json MISSING.")
            sys.exit(1)
        # Check if usecase_config.json exists.
        self.usecase_config_filepath = start_dirpath \
            + "/config/usecase_config.json"
        if os.path.isfile(self.usecase_config_filepath):
            print("[ config_epd ] - usecase_config.json FOUND.")
            self.parse_usecase_config(self.usecase_config_filepath)
        else:
            print("[ config_epd ] - ERROR. usecase_config.json MISSING.")
            sys.exit(1)

        if len(args) < 2:
            print('Please specify a configuration.')
            self.print_help()
            sys.exit(1)

        self.parse_args(args[1:])

        self.write_out(
            self.session_config_filepath,
            self.usecase_config_filepath)

    def print_help(self):
        print('config_epd.py [--visualize ] [--action ] [--cpu ] [--gpu ] ' +
              '[--model ] [--label ] [-m ] [-l ]')
        print()
        print('-v --visualize   Sets EPD to Visualize Mode.')
        print('-a --action   Sets EPD to Action Mode.')
        print('-g --gpu   Sets EPD to GPU Mode.')
        print('-c --cpu   Sets EPD to CPU Mode.')
        print('-m --model   Sets new onnx model to be deployed via EPD.')
        print('-l --label   Sets new label list to be deployed via EPD.')
        print('--use   Sets usecase mode to be deployed via EPD. ' +
              'Eg. [0,1,2,3,4].')
        print('--class-list   Sets class names for COUNTING use case ' +
              '(comma-separated).')
        print('--color-template   Sets color template file path for ' +
              'COLOR-MATCHING use case.')
        print('--color-hist-metric   Sets histogram comparison metric for ' +
              'COLOR-MATCHING use case. Acceptable values: ' +
              'Correlation, Chi-square, Intersection, Bhattacharyya.')
        print('--track-type   Sets tracker type for TRACKING use case.')
        print('--topic   Sets the subscriber topic name EPD uses ' +
              'to get input images.')
        print('--intra-op-threads   Sets intra-op thread count for ORT.')
        print('--publish-segmentation   Enables detection mask/point-cloud output.')
        print('--no-publish-segmentation   Disables detection mask/point-cloud output.')

    def isInEPDPackageRoot(self, start_dirpath):
        if (os.path.isdir(start_dirpath + "/scripts") and
           os.path.isdir(start_dirpath + "/launch") and
           os.path.isdir(start_dirpath + "/data")):
            print("[ config_epd ] - Executing in root of EPD package.")
        else:
            print("[ config_epd ] - ERROR. Not in root of EPD package")
            print("[ config_epd ] - Please run in root of EPD package.")
            print("[ config_epd ] - Exiting...")
            sys.exit(1)

    def parse_args(self, args):
        # TODO(cardboardcode): Add options for usecase_config.json
        # CLI configuration.
        opts, _ = getopt.getopt(args, 'hvagcm:l:',
                                        ['visualize',
                                         'action',
                                         'gpu',
                                         'cpu',
                                         'model=',
                                         'label=',
                                         'use=',
                                         'class-list=',
                                         'color-template=',
                                         'color-hist-metric=',
                                         'track-type=',
                                         'topic=',
                                         'intra-op-threads=',
                                         'publish-segmentation',
                                         'no-publish-segmentation'])

        for opt, arg in opts:
            if opt == '-h':
                self.print_help()
                sys.exit(0)
            elif opt in ('-v', '--visualize'):
                print("[ session_config.json ] - Setting to Visualize Mode.")
                self.visualizeFlag = True
            elif opt in ('-a', '--action'):
                print("[ session_config.json ] - Setting to Action Mode.")
                self.visualizeFlag = False
            elif opt in ('-g', '--gpu'):
                print("[ session_config.json ] - Setting to GPU Mode.")
                self.useCPU = False
            elif opt in ('-c', '--cpu'):
                print("[ session_config.json ] - Setting to CPU Mode.")
                self.useCPU = True
            elif opt in ('-m', '--model'):
                resolved_path = os.path.abspath(os.path.expanduser(arg))
                if not os.path.isfile(resolved_path):
                    print("[ config_epd ] - ERROR." +
                          " input model file does not exist at " +
                          resolved_path + ".")
                    print("[ config_epd ] - Exiting.")
                    sys.exit(2)
                self._path_to_model = resolved_path
            elif opt in ('-l', '--label'):
                resolved_path = os.path.abspath(os.path.expanduser(arg))
                if not os.path.isfile(resolved_path):
                    print("[ config_epd ] - ERROR." +
                          " input label list does not exist at " +
                          resolved_path + ".")
                    print("[ config_epd ] - Exiting.")
                    sys.exit(2)
                self._path_to_label_list = resolved_path
            elif opt in ('--use'):
                try:
                    usecase_mode = int(arg)
                except ValueError:
                    print("[ session_config.json ] - Invalid Use Case Mode " +
                          "provided. Expected integer 0-4. Exiting...")
                    sys.exit(1)
                if usecase_mode < 0 or usecase_mode > 4:
                    self.set_use_case_from_cli(usecase_mode)
                    continue
                self.set_use_case_from_cli(usecase_mode)
            elif opt in ('--class-list'):
                class_list = [item.strip() for item in arg.split(',')
                              if item.strip()]
                if not class_list:
                    print("[ config_epd ] - ERROR." +
                          " class list cannot be empty.")
                    print("[ config_epd ] - Exiting.")
                    sys.exit(2)
                self.count_class_list = class_list
            elif opt in ('--color-template'):
                self.path_to_color_template = arg
            elif opt in ('--color-hist-metric'):
                self.color_match_histogram_metric = \
                    self.normalize_color_histogram_metric(arg)
            elif opt in ('--track-type'):
                self.track_type = arg
            elif opt in ('--topic'):
                print("[ session_config.json ] - Setting new input " +
                      "image topic to", arg)
                self.input_image_topic = arg
            elif opt in ('--intra-op-threads'):
                try:
                    thread_count = int(arg)
                except ValueError:
                    print("[ session_config.json ] - ERROR." +
                          " intra-op-threads must be an integer.")
                    print("[ config_epd ] - Exiting.")
                    sys.exit(2)
                if thread_count < 0:
                    print("[ session_config.json ] - ERROR." +
                          " intra-op-threads must be >= 0.")
                    print("[ config_epd ] - Exiting.")
                    sys.exit(2)
                self.intra_op_num_threads = thread_count
            elif opt in ('--publish-segmentation'):
                print("[ session_config.json ] - Enabling detection segmentation output.")
                self.publish_detection_segmentation = True
            elif opt in ('--no-publish-segmentation'):
                print("[ session_config.json ] - Disabling detection segmentation output.")
                self.publish_detection_segmentation = False
        self.validate_usecase_inputs()

    def parse_session_config(self, session_config_filepath):

        f = open(session_config_filepath)
        data = json.load(f)
        self._path_to_model = data["path_to_model"]
        self._path_to_label_list = data["path_to_label_list"]
        self.intra_op_num_threads = data.get("intra_op_num_threads", 0)
        self.publish_detection_segmentation = data.get(
            "publish_detection_segmentation",
            True)
        if data["useCPU"] == "CPU":
            self.useCPU = True
        else:
            self.useCPU = False
        if data["visualizeFlag"] == "visualize":
            self.visualizeFlag = True
        else:
            self.visualizeFlag = False
        f.close()

    def parse_usecase_config(self, usecase_config_filepath):

        f = open(usecase_config_filepath)
        data = json.load(f)
        self.usecase_mode = data["usecase_mode"]
        if self.usecase_mode == 0:
            print("[ Use Case ] - CLASSIFICATION")
        elif self.usecase_mode == 1:
            print("[ Use Case ] - COUNTING")
            self.count_class_list = data["class_list"]
        elif self.usecase_mode == 2:
            print("[ Use Case ] - COLOR-MATCHING")
            self.path_to_color_template = data["path_to_color_template"]
            if "color_match_histogram_metric" in data:
                self.color_match_histogram_metric = \
                    self.normalize_color_histogram_metric(
                        data["color_match_histogram_metric"])
        elif self.usecase_mode == 3:
            print("[ Use Case ] - LOCALIZATION")
        elif self.usecase_mode == 4:
            print("[ Use Case ] - TRACKING")
            self.track_type = data["track_type"]
        else:
            print("[ Use Case ] - INVALID. Please rectify" +
                  " usecase_config.json. Exiting...")
            f.close()
            sys.exit(1)
        f.close()

    def parse_inputimagetopic_config(self, inputimagetopic_config_filepath):
        f = open(inputimagetopic_config_filepath)
        data = json.load(f)
        self.input_image_topic = data["input_image_topic"]
        f.close()

    def normalize_color_histogram_metric(self, metric):
        if isinstance(metric, int):
            metric_value = metric
        else:
            metric_str = str(metric).strip()
            if metric_str.isdigit():
                metric_value = int(metric_str)
            else:
                metric_lower = metric_str.lower()
                if metric_lower == "correlation":
                    return "Correlation"
                if metric_lower in ("chi-square", "chisquare", "chi_square"):
                    return "Chi-square"
                if metric_lower == "intersection":
                    return "Intersection"
                if metric_lower == "bhattacharyya":
                    return "Bhattacharyya"
                print("[ config_epd ] - ERROR." +
                      " Invalid color-hist-metric provided. " +
                      "Expected Correlation, Chi-square, Intersection, " +
                      "or Bhattacharyya.")
                sys.exit(2)

        if metric_value == 0:
            return "Correlation"
        if metric_value == 1:
            return "Chi-square"
        if metric_value == 2:
            return "Intersection"
        if metric_value == 3:
            return "Bhattacharyya"
        print("[ config_epd ] - ERROR." +
              " Invalid color-hist-metric provided. " +
              "Expected 0-3 or Correlation, Chi-square, Intersection, " +
              "or Bhattacharyya.")
        sys.exit(2)

    def set_use_case_from_cli(self, usecase_mode):
        self.usecase_mode = usecase_mode

        if usecase_mode == 0:
            print("[ session_config.json ] - " +
                  "Setting Use Case Mode to CLASSIFICATION.")
        elif usecase_mode == 1:
            print("[ session_config.json ] - " +
                  "Setting Use Case Mode to COUNTING.")
        elif usecase_mode == 2:
            print("[ session_config.json ] - " +
                  "Setting Use Case Mode to COLOR-MATCHING.")
        elif usecase_mode == 3:
            print("[ session_config.json ] - " +
                  "Setting Use Case Mode to LOCALIZATION.")
        elif usecase_mode == 4:
            print("[ session_config.json ] - " +
                  "Setting Use Case Mode to TRACKING.")
        else:
            print("[ session_config.json ] - " +
                  "Invalid Use Case Mode provided. Exiting...")
            sys.exit(1)

    def validate_usecase_inputs(self):
        is_interactive = sys.stdin.isatty()
        if self.usecase_mode == 1:
            if not self.count_class_list:
                if not is_interactive:
                    print("[ config_epd ] - ERROR." +
                          " --class-list is required for COUNTING " +
                          "use case in non-interactive mode.")
                    print("[ config_epd ] - Exiting.")
                    sys.exit(2)
                n = int(input("Please enter number of object class names : "))
                self.count_class_list.clear()
                for _ in range(0, n):
                    ele = input("Please enter class name: ")
                    self.count_class_list.append(ele)
        elif self.usecase_mode == 2:
            if not self.path_to_color_template:
                if not is_interactive:
                    print("[ config_epd ] - ERROR." +
                          " --color-template is required for " +
                          "COLOR-MATCHING use case in non-interactive mode.")
                    print("[ config_epd ] - Exiting.")
                    sys.exit(2)
                self.path_to_color_template = input(
                    "Please enter Color Image File Path: ")
            self.color_match_histogram_metric = \
                self.normalize_color_histogram_metric(
                    self.color_match_histogram_metric)
        elif self.usecase_mode == 4:
            if not self.track_type:
                if not is_interactive:
                    print("[ config_epd ] - ERROR." +
                          " --track-type is required for TRACKING " +
                          "use case in non-interactive mode.")
                    print("[ config_epd ] - Exiting.")
                    sys.exit(2)
                self.track_type = input(
                    "Please enter Tracker Type [KCF, MEDIANFLOW, CSRT]: ")

    def write_out(self, session_config_filepath, usecase_config_filepath):

        if self.visualizeFlag:
            visualizeFlag_string = "visualize"
        else:
            visualizeFlag_string = "robot"

        if self.useCPU:
            useCPU_string = "CPU"
        else:
            useCPU_string = "GPU"

        dict = {
            "path_to_model": self._path_to_model,
            "path_to_label_list": self._path_to_label_list,
            "visualizeFlag": visualizeFlag_string,
            "useCPU": useCPU_string,
            "intra_op_num_threads": self.intra_op_num_threads,
            "publish_detection_segmentation": self.publish_detection_segmentation
            }
        json_object_1 = json.dumps(dict, indent=4)

        with open(session_config_filepath, 'w') as outfile_1:
            outfile_1.write(json_object_1)

        if self.usecase_mode == 0:
            dict = {
                "usecase_mode": self.usecase_mode
            }
            json_object_2 = json.dumps(dict, indent=4)
            with open(usecase_config_filepath, 'w') as outfile_2:
                outfile_2.write(json_object_2)
        elif self.usecase_mode == 1:
            dict = {
                "usecase_mode": self.usecase_mode,
                "class_list": self.count_class_list
            }
            json_object_2 = json.dumps(dict, indent=4)
            with open(usecase_config_filepath, 'w') as outfile_2:
                outfile_2.write(json_object_2)
        elif self.usecase_mode == 2:
            dict = {
                "usecase_mode": self.usecase_mode,
                "path_to_color_template": self.path_to_color_template,
                "color_match_histogram_metric":
                    self.color_match_histogram_metric
            }
            json_object_2 = json.dumps(dict, indent=4)
            with open(usecase_config_filepath, 'w') as outfile_2:
                outfile_2.write(json_object_2)
        elif self.usecase_mode == 3:
            dict = {
                "usecase_mode": self.usecase_mode
            }
            json_object_2 = json.dumps(dict, indent=4)
            with open(usecase_config_filepath, 'w') as outfile_2:
                outfile_2.write(json_object_2)
        elif self.usecase_mode == 4:
            dict = {
                "usecase_mode": self.usecase_mode,
                "track_type": self.track_type
            }
            json_object_2 = json.dumps(dict, indent=4)
            with open(usecase_config_filepath, 'w') as outfile_2:
                outfile_2.write(json_object_2)

        dict = {
            "input_image_topic": self.input_image_topic
            }
        json_object_2 = json.dumps(dict, indent=4)

        with open(self.inputimagetopic_config_filepath, 'w') as outfile_1:
            outfile_1.write(json_object_2)


def main(args=None):
    # Checks if this script is run in the root of the
    # easy_perception_deployment ROS2 package.
    start_dirpath = os.getcwd()
    # Check if the following folders/files are in the directory
    # when script is being executed:
    # /data, /scripts, /launch, CMakeLists.txt and package.xml
    configurator = EPDConfigurator(start_dirpath, args)


if __name__ == "__main__":
    main(sys.argv)
