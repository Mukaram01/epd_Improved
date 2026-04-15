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


import argparse
import os
import sys
import json


class EPDConfigError(ValueError):
    """Raised for invalid EPD configuration input; caught in main() to set exit code."""


class EPDConfigurator:
    COLOR_HISTOGRAM_METRIC_CHOICES = (
        "Correlation",
        "Chi-square",
        "Intersection",
        "Bhattacharyya")
    TRACK_TYPE_CHOICES = ("KCF", "MEDIANFLOW", "CSRT")

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
            raise EPDConfigError("session_config.json missing")
        # Check if input_image_topic.json exists.
        self.inputimagetopic_config_filepath = start_dirpath \
            + "/config/input_image_topic.json"
        if os.path.isfile(self.inputimagetopic_config_filepath):
            print("[ config_epd ] - input_image_topic.json FOUND.")
            self.parse_inputimagetopic_config(
                self.inputimagetopic_config_filepath)
        else:
            print("[ config_epd ] - ERROR. input_image_topic.json MISSING.")
            raise EPDConfigError("input_image_topic.json missing")
        # Check if usecase_config.json exists.
        self.usecase_config_filepath = start_dirpath \
            + "/config/usecase_config.json"
        if os.path.isfile(self.usecase_config_filepath):
            print("[ config_epd ] - usecase_config.json FOUND.")
            self.parse_usecase_config(self.usecase_config_filepath)
        else:
            print("[ config_epd ] - ERROR. usecase_config.json MISSING.")
            raise EPDConfigError("usecase_config.json missing")

        arg_list = self._validate_and_get_arg_list(args)
        if not arg_list:
            print('Please specify a configuration.')
            self._build_arg_parser().print_help()
            raise EPDConfigError("No configuration arguments provided")

        self.parse_args(arg_list)

        self.write_out(
            self.session_config_filepath,
            self.usecase_config_filepath)

    def _validate_and_get_arg_list(self, args):
        if args is None:
            raise EPDConfigError(
                "Malformed CLI args: expected list/tuple like sys.argv, got None")
        if not isinstance(args, (list, tuple)):
            raise EPDConfigError(
                "Malformed CLI args: expected list/tuple like sys.argv")
        if not args:
            raise EPDConfigError(
                "Malformed CLI args: expected argv[0] script name")
        if not all(isinstance(arg, str) for arg in args):
            raise EPDConfigError(
                "Malformed CLI args: all argv entries must be strings")
        return list(args[1:])

    def _build_arg_parser(self):
        parser = argparse.ArgumentParser(
            prog='config_epd.py',
            description='Configure EasyPerceptionDeployment settings',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=(
                'Use-case modes:\n'
                '  0  CLASSIFICATION\n'
                '  1  COUNTING       (requires --class-list)\n'
                '  2  COLOR-MATCHING (requires --color-template)\n'
                '  3  LOCALIZATION\n'
                '  4  TRACKING       (requires --track-type)\n'
            ))

        vis_group = parser.add_mutually_exclusive_group()
        vis_group.add_argument(
            '-v', '--visualize', action='store_true',
            help='Sets EPD to Visualize Mode.')
        vis_group.add_argument(
            '-a', '--action', action='store_true',
            help='Sets EPD to Action Mode.')

        hw_group = parser.add_mutually_exclusive_group()
        hw_group.add_argument(
            '-g', '--gpu', action='store_true',
            help='Sets EPD to GPU Mode.')
        hw_group.add_argument(
            '-c', '--cpu', action='store_true',
            help='Sets EPD to CPU Mode.')

        parser.add_argument(
            '-m', '--model', metavar='PATH',
            help='Sets new onnx model to be deployed via EPD.')
        parser.add_argument(
            '-l', '--label', metavar='PATH',
            help='Sets new label list to be deployed via EPD.')
        parser.add_argument(
            '--use', type=int, metavar='MODE',
            help='Sets usecase mode [0,1,2,3,4].')
        parser.add_argument(
            '--class-list', metavar='CLASS1,CLASS2,...',
            help='Sets class names for COUNTING use case (required when --use 1).')
        parser.add_argument(
            '--color-template', metavar='PATH',
            help='Sets color template file path for COLOR-MATCHING use case '
                 '(required when --use 2).')
        parser.add_argument(
            '--color-hist-metric', metavar='METRIC',
            help='Histogram comparison metric for COLOR-MATCHING. '
                 'Values: 0-3 or Correlation, Chi-square, Intersection, Bhattacharyya.')
        parser.add_argument(
            '--track-type', metavar='TYPE',
            help='Sets tracker type for TRACKING use case '
                 '(required when --use 4). Values: KCF, MEDIANFLOW, CSRT.')
        parser.add_argument(
            '--topic', metavar='TOPIC',
            help='Sets the subscriber topic name EPD uses to get input images.')
        parser.add_argument(
            '--intra-op-threads', type=int, metavar='N', dest='intra_op_threads',
            help='Sets intra-op thread count for ORT.')

        seg_group = parser.add_mutually_exclusive_group()
        seg_group.add_argument(
            '--publish-segmentation', dest='publish_segmentation',
            action='store_true',
            help='Enables detection mask/point-cloud output.')
        seg_group.add_argument(
            '--no-publish-segmentation', dest='publish_segmentation',
            action='store_false',
            help='Disables detection mask/point-cloud output.')
        parser.set_defaults(publish_segmentation=None)

        return parser

    def isInEPDPackageRoot(self, start_dirpath):
        if (os.path.isdir(start_dirpath + "/scripts") and
           os.path.isdir(start_dirpath + "/launch") and
           os.path.isdir(start_dirpath + "/data")):
            print("[ config_epd ] - Executing in root of EPD package.")
        else:
            print("[ config_epd ] - ERROR. Not in root of EPD package")
            print("[ config_epd ] - Please run in root of EPD package.")
            print("[ config_epd ] - Exiting...")
            raise EPDConfigError("Not in root of EPD package")

    def parse_args(self, args):
        parser = self._build_arg_parser()
        namespace = parser.parse_args(args)

        if namespace.visualize:
            print("[ session_config.json ] - Setting to Visualize Mode.")
            self.visualizeFlag = True
        elif namespace.action:
            print("[ session_config.json ] - Setting to Action Mode.")
            self.visualizeFlag = False

        if namespace.gpu:
            print("[ session_config.json ] - Setting to GPU Mode.")
            self.useCPU = False
        elif namespace.cpu:
            print("[ session_config.json ] - Setting to CPU Mode.")
            self.useCPU = True

        if namespace.model is not None:
            resolved_path = os.path.abspath(os.path.expanduser(namespace.model))
            if not os.path.isfile(resolved_path):
                print("[ config_epd ] - ERROR."
                      " input model file does not exist at "
                      + resolved_path + ".")
                print("[ config_epd ] - Exiting.")
                raise EPDConfigError(
                    "Model file does not exist: " + resolved_path)
            self._path_to_model = namespace.model

        if namespace.label is not None:
            resolved_path = os.path.abspath(os.path.expanduser(namespace.label))
            if not os.path.isfile(resolved_path):
                print("[ config_epd ] - ERROR."
                      " input label list does not exist at "
                      + resolved_path + ".")
                print("[ config_epd ] - Exiting.")
                raise EPDConfigError(
                    "Label list file does not exist: " + resolved_path)
            self._path_to_label_list = namespace.label

        if namespace.use is not None:
            self.set_use_case_from_cli(namespace.use)

        if namespace.class_list is not None:
            class_list = [item.strip() for item in namespace.class_list.split(',')
                          if item.strip()]
            if not class_list:
                print("[ config_epd ] - ERROR."
                      " class list cannot be empty.")
                print("[ config_epd ] - Exiting.")
                raise EPDConfigError("class list cannot be empty")
            self.count_class_list = class_list

        if namespace.color_template is not None:
            self.path_to_color_template = \
                self.validate_existing_file_path(
                    namespace.color_template,
                    "color template")

        if namespace.color_hist_metric is not None:
            self.color_match_histogram_metric = \
                self.normalize_color_histogram_metric(namespace.color_hist_metric)

        if namespace.track_type is not None:
            self.track_type = self.normalize_track_type(namespace.track_type)

        if namespace.topic is not None:
            print("[ session_config.json ] - Setting new input "
                  "image topic to", namespace.topic)
            self.input_image_topic = namespace.topic

        if namespace.intra_op_threads is not None:
            if namespace.intra_op_threads < 0:
                print("[ session_config.json ] - ERROR."
                      " intra-op-threads must be >= 0.")
                print("[ config_epd ] - Exiting.")
                raise EPDConfigError("intra-op-threads must be >= 0")
            self.intra_op_num_threads = namespace.intra_op_threads

        if namespace.publish_segmentation is not None:
            if namespace.publish_segmentation:
                print("[ session_config.json ] - Enabling detection "
                      "segmentation output.")
            else:
                print("[ session_config.json ] - Disabling detection "
                      "segmentation output.")
            self.publish_detection_segmentation = namespace.publish_segmentation

        self.validate_usecase_inputs()

    def parse_session_config(self, session_config_filepath):
        data = self._load_json_config(
            session_config_filepath,
            "session_config.json",
            required_keys=[
                "path_to_model",
                "path_to_label_list",
                "useCPU",
                "visualizeFlag"])
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

    def parse_usecase_config(self, usecase_config_filepath):
        data = self._load_json_config(
            usecase_config_filepath,
            "usecase_config.json",
            required_keys=["usecase_mode"])
        self.usecase_mode = data["usecase_mode"]
        if self.usecase_mode == 0:
            print("[ Use Case ] - CLASSIFICATION")
        elif self.usecase_mode == 1:
            print("[ Use Case ] - COUNTING")
            self.count_class_list = self.normalize_class_list(
                data.get("class_list"))
        elif self.usecase_mode == 2:
            print("[ Use Case ] - COLOR-MATCHING")
            self.path_to_color_template = self.validate_existing_file_path(
                data.get("path_to_color_template"),
                "color template")
            if "color_match_histogram_metric" in data:
                self.color_match_histogram_metric = \
                    self.normalize_color_histogram_metric(
                        data["color_match_histogram_metric"])
        elif self.usecase_mode == 3:
            print("[ Use Case ] - LOCALIZATION")
        elif self.usecase_mode == 4:
            print("[ Use Case ] - TRACKING")
            self.track_type = self.normalize_track_type(data.get("track_type"))
        else:
            print("[ Use Case ] - INVALID. Please rectify" +
                  " usecase_config.json. Exiting...")
            raise EPDConfigError(
                "Invalid usecase_mode in usecase_config.json: "
                + str(self.usecase_mode))

    def parse_inputimagetopic_config(self, inputimagetopic_config_filepath):
        data = self._load_json_config(
            inputimagetopic_config_filepath,
            "input_image_topic.json",
            required_keys=["input_image_topic"])
        self.input_image_topic = data["input_image_topic"]

    def _load_json_config(
            self,
            filepath,
            config_name,
            required_keys=None,
            defaults=None):
        required_keys = required_keys or []
        defaults = defaults or {}
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print("[ config_epd ] - ERROR. " +
                  f"{config_name} is malformed JSON: {e}.")
            print("[ config_epd ] - Exiting.")
            raise EPDConfigError(f"{config_name} is malformed JSON: {e}") from e
        except OSError as e:
            print("[ config_epd ] - ERROR. " +
                  f"Unable to read {config_name}: {e}.")
            print("[ config_epd ] - Exiting.")
            raise EPDConfigError(f"Unable to read {config_name}: {e}") from e

        if not isinstance(data, dict):
            print("[ config_epd ] - ERROR. " +
                  f"{config_name} must contain a JSON object.")
            print("[ config_epd ] - Exiting.")
            raise EPDConfigError(f"{config_name} must contain a JSON object")

        merged = dict(defaults)
        merged.update(data)
        missing = [key for key in required_keys if key not in merged]
        if missing:
            print("[ config_epd ] - ERROR. " +
                  f"{config_name} missing required keys: {missing}.")
            print("[ config_epd ] - Exiting.")
            raise EPDConfigError(
                f"{config_name} missing required keys: {missing}")
        return merged

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
                print("[ config_epd ] - ERROR."
                      " Invalid color-hist-metric provided. "
                      "Expected " +
                      ", ".join(self.COLOR_HISTOGRAM_METRIC_CHOICES) + ".")
                raise EPDConfigError(
                    "Invalid color-hist-metric: " + metric_str)

        if metric_value == 0:
            return "Correlation"
        if metric_value == 1:
            return "Chi-square"
        if metric_value == 2:
            return "Intersection"
        if metric_value == 3:
            return "Bhattacharyya"
        print("[ config_epd ] - ERROR."
              " Invalid color-hist-metric provided. "
              "Expected 0-3 or " +
              ", ".join(self.COLOR_HISTOGRAM_METRIC_CHOICES) + ".")
        raise EPDConfigError(
            "Invalid color-hist-metric value: " + str(metric_value))

    def normalize_track_type(self, track_type):
        normalized = str(track_type or "").strip().upper()
        if normalized in self.TRACK_TYPE_CHOICES:
            return normalized
        print("[ config_epd ] - ERROR."
              " Invalid track-type provided. "
              "Expected one of: " +
              ", ".join(self.TRACK_TYPE_CHOICES) + ".")
        raise EPDConfigError("Invalid track-type: " + str(track_type))

    def normalize_class_list(self, class_list):
        if not isinstance(class_list, list):
            print("[ config_epd ] - ERROR."
                  " class_list must be a list of non-empty class names.")
            raise EPDConfigError("class_list must be a list")
        normalized_class_list = []
        for class_name in class_list:
            class_name_str = str(class_name).strip()
            if class_name_str:
                normalized_class_list.append(class_name_str)
        if not normalized_class_list:
            print("[ config_epd ] - ERROR."
                  " class_list must contain at least one class name.")
            raise EPDConfigError(
                "class_list must contain at least one class name")
        return normalized_class_list

    def validate_existing_file_path(self, filepath, option_name):
        candidate_path = str(filepath or "").strip()
        if not candidate_path:
            print("[ config_epd ] - ERROR."
                  f" {option_name} path cannot be empty.")
            raise EPDConfigError(f"{option_name} path cannot be empty")
        resolved_path = os.path.abspath(os.path.expanduser(candidate_path))
        if not os.path.isfile(resolved_path):
            print("[ config_epd ] - ERROR."
                  f" {option_name} file does not exist at {resolved_path}.")
            print("[ config_epd ] - Exiting.")
            raise EPDConfigError(
                f"{option_name} file does not exist: {resolved_path}")
        return candidate_path

    def set_use_case_from_cli(self, usecase_mode):
        self.usecase_mode = usecase_mode

        if usecase_mode == 0:
            print("[ session_config.json ] - "
                  "Setting Use Case Mode to CLASSIFICATION.")
        elif usecase_mode == 1:
            print("[ session_config.json ] - "
                  "Setting Use Case Mode to COUNTING.")
        elif usecase_mode == 2:
            print("[ session_config.json ] - "
                  "Setting Use Case Mode to COLOR-MATCHING.")
        elif usecase_mode == 3:
            print("[ session_config.json ] - "
                  "Setting Use Case Mode to LOCALIZATION.")
        elif usecase_mode == 4:
            print("[ session_config.json ] - "
                  "Setting Use Case Mode to TRACKING.")
        else:
            print("[ session_config.json ] - "
                  "Invalid Use Case Mode provided. Exiting...")
            raise EPDConfigError(
                "Invalid use case mode: " + str(usecase_mode))

    def validate_usecase_inputs(self):
        is_interactive = sys.stdin.isatty()
        if self.usecase_mode == 1:
            if not self.count_class_list:
                if not is_interactive:
                    print("[ config_epd ] - ERROR."
                          " --class-list is required for COUNTING "
                          "use case in non-interactive mode.")
                    print("[ config_epd ] - Exiting.")
                    raise EPDConfigError(
                        "--class-list required for COUNTING use case")
                n = int(input("Please enter number of object class names : "))
                self.count_class_list.clear()
                for _ in range(0, n):
                    ele = input("Please enter class name: ")
                    self.count_class_list.append(ele)
            self.count_class_list = self.normalize_class_list(
                self.count_class_list)
        elif self.usecase_mode == 2:
            if not self.path_to_color_template:
                if not is_interactive:
                    print("[ config_epd ] - ERROR."
                          " --color-template is required for "
                          "COLOR-MATCHING use case in non-interactive mode.")
                    print("[ config_epd ] - Exiting.")
                    raise EPDConfigError(
                        "--color-template required for COLOR-MATCHING use case")
                self.path_to_color_template = input(
                    "Please enter Color Image File Path: ")
            self.path_to_color_template = self.validate_existing_file_path(
                self.path_to_color_template,
                "color template")
            self.color_match_histogram_metric = \
                self.normalize_color_histogram_metric(
                    self.color_match_histogram_metric)
        elif self.usecase_mode == 4:
            if not self.track_type:
                if not is_interactive:
                    print("[ config_epd ] - ERROR."
                          " --track-type is required for TRACKING "
                          "use case in non-interactive mode.")
                    print("[ config_epd ] - Exiting.")
                    raise EPDConfigError(
                        "--track-type required for TRACKING use case")
                self.track_type = input(
                    "Please enter Tracker Type [KCF, MEDIANFLOW, CSRT]: ")
            self.track_type = self.normalize_track_type(self.track_type)

    def write_out(self, session_config_filepath, usecase_config_filepath):

        if self.visualizeFlag:
            visualizeFlag_string = "visualize"
        else:
            visualizeFlag_string = "robot"

        if self.useCPU:
            useCPU_string = "CPU"
        else:
            useCPU_string = "GPU"

        session_config = {
            "path_to_model": self._path_to_model,
            "path_to_label_list": self._path_to_label_list,
            "visualizeFlag": visualizeFlag_string,
            "useCPU": useCPU_string,
            "intra_op_num_threads": self.intra_op_num_threads,
            "publish_detection_segmentation":
                self.publish_detection_segmentation
            }
        json_object_1 = json.dumps(session_config, indent=4)

        with open(session_config_filepath, 'w') as outfile_1:
            outfile_1.write(json_object_1)

        if self.usecase_mode == 0:
            usecase_config = {
                "usecase_mode": self.usecase_mode
            }
        elif self.usecase_mode == 1:
            usecase_config = {
                "usecase_mode": self.usecase_mode,
                "class_list": self.count_class_list
            }
        elif self.usecase_mode == 2:
            usecase_config = {
                "usecase_mode": self.usecase_mode,
                "path_to_color_template": self.path_to_color_template,
                "color_match_histogram_metric":
                    self.color_match_histogram_metric
            }
        elif self.usecase_mode == 3:
            usecase_config = {
                "usecase_mode": self.usecase_mode
            }
        elif self.usecase_mode == 4:
            usecase_config = {
                "usecase_mode": self.usecase_mode,
                "track_type": self.track_type
            }
        else:
            usecase_config = {"usecase_mode": self.usecase_mode}

        json_object_2 = json.dumps(usecase_config, indent=4)
        with open(usecase_config_filepath, 'w') as outfile_2:
            outfile_2.write(json_object_2)

        topic_config = {
            "input_image_topic": self.input_image_topic
            }
        json_object_3 = json.dumps(topic_config, indent=4)

        with open(self.inputimagetopic_config_filepath, 'w') as outfile_3:
            outfile_3.write(json_object_3)


def main(args=None):
    # Checks if this script is run in the root of the
    # easy_perception_deployment ROS2 package.
    start_dirpath = os.getcwd()
    # Check if the following folders/files are in the directory
    # when script is being executed:
    # /data, /scripts, /launch, CMakeLists.txt and package.xml
    try:
        if args is None:
            args = sys.argv
        elif not isinstance(args, (list, tuple)):
            raise EPDConfigError(
                "Malformed CLI args: expected list/tuple like sys.argv")
        EPDConfigurator(start_dirpath, args)
    except EPDConfigError as exc:
        print(f"[ config_epd ] - Fatal: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
