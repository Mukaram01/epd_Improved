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

import json
import pytest
from pathlib import Path

from cli.config_epd import EPDConfigurator, EPDConfigError

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_START_DIR = str(PACKAGE_ROOT)

# Reset session_config.json and usecase_config.json to default.
if ((PACKAGE_ROOT / "config/session_config.json").exists() and
   (PACKAGE_ROOT / "config/usecase_config.json").exists()):
    (PACKAGE_ROOT / "config/session_config.json").unlink(missing_ok=True)
    (PACKAGE_ROOT / "config/usecase_config.json").unlink(missing_ok=True)

    dict = {
        "path_to_model": './data/model/squeezenet1.1-7.onnx',
        "path_to_label_list": './data/label_list/imagenet_classes.txt',
        "visualizeFlag": 'visualize',
        "useCPU": 'CPU'
        }
    json_object = json.dumps(dict, indent=4)
    with open(PACKAGE_ROOT / 'config/session_config.json', 'w') as outfile:
        outfile.write(json_object)

    dict = {"usecase_mode": 0}
    json_object = json.dumps(dict, indent=4)
    with open(PACKAGE_ROOT / 'config/usecase_config.json', 'w') as outfile:
        outfile.write(json_object)


def test_invalid_ExeDirectory():
    # Change to invalid directory.
    test_args = ['scripts/cli/config_epd.py', '-v']
    INVALID_START_DIR = str(PACKAGE_ROOT / "scripts")

    with pytest.raises(EPDConfigError):
        EPDConfigurator(INVALID_START_DIR, test_args)


def test_print_help_NoArgs(capfd):
    test_args = ['scripts/cli/config_epd.py']

    with pytest.raises(EPDConfigError):
        EPDConfigurator(REQUIRED_START_DIR, test_args)


def test_print_help_HelpArg(capfd):
    test_args = ['scripts/cli/config_epd.py', '-h']

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        EPDConfigurator(REQUIRED_START_DIR, test_args)

    assert pytest_wrapped_e.value.code == 0


def test_set_VisualizeMode_short():

    test_args = ['scripts/cli/config_epd.py', '-v']
    session_config_filepath = REQUIRED_START_DIR \
        + "/config/session_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load session_config.json
    f = open(session_config_filepath)
    data = json.load(f)
    visualizeFlag = data["visualizeFlag"]
    f.close()

    assert visualizeFlag == "visualize"


def test_set_VisualizeMode_long():

    test_args = ['scripts/cli/config_epd.py', '--visualize']
    session_config_filepath = REQUIRED_START_DIR \
        + "/config/session_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load session_config.json
    f = open(session_config_filepath)
    data = json.load(f)
    visualizeFlag = data["visualizeFlag"]
    f.close()

    assert visualizeFlag == "visualize"


def test_set_ActionMode_short():

    test_args = ['scripts/cli/config_epd.py', '-a']
    session_config_filepath = REQUIRED_START_DIR \
        + "/config/session_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load session_config.json
    f = open(session_config_filepath)
    data = json.load(f)
    visualizeFlag = data["visualizeFlag"]
    f.close()

    assert visualizeFlag == "robot"


def test_set_ActionMode_long():

    test_args = ['scripts/cli/config_epd.py', '--action']
    session_config_filepath = REQUIRED_START_DIR \
        + "/config/session_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load session_config.json
    f = open(session_config_filepath)
    data = json.load(f)
    visualizeFlag = data["visualizeFlag"]
    f.close()

    assert visualizeFlag == "robot"


def test_set_GPU_short():

    test_args = ['scripts/cli/config_epd.py', '-g']
    session_config_filepath = REQUIRED_START_DIR \
        + "/config/session_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load session_config.json
    f = open(session_config_filepath)
    data = json.load(f)
    useCPU = data["useCPU"]
    f.close()

    assert useCPU == "GPU"


def test_set_GPU_long():

    test_args = ['scripts/cli/config_epd.py', '--gpu']
    session_config_filepath = REQUIRED_START_DIR \
        + "/config/session_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load session_config.json
    f = open(session_config_filepath)
    data = json.load(f)
    useCPU = data["useCPU"]
    f.close()

    assert useCPU == "GPU"


def test_set_CPU_short():

    test_args = ['scripts/cli/config_epd.py', '-c']
    session_config_filepath = REQUIRED_START_DIR \
        + "/config/session_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load session_config.json
    f = open(session_config_filepath)
    data = json.load(f)
    useCPU = data["useCPU"]
    f.close()

    assert useCPU == "CPU"


def test_set_CPU_long():

    test_args = ['scripts/cli/config_epd.py', '--cpu']
    session_config_filepath = REQUIRED_START_DIR \
        + "/config/session_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load session_config.json
    f = open(session_config_filepath)
    data = json.load(f)
    useCPU = data["useCPU"]
    f.close()

    assert useCPU == "CPU"


@pytest.mark.parametrize(
    "filename,parser_name,required_key",
    [
        ("session_config.json", "parse_session_config", "path_to_model"),
        ("usecase_config.json", "parse_usecase_config", "usecase_mode"),
        ("input_image_topic.json", "parse_inputimagetopic_config", "input_image_topic"),
    ],
)
def test_parse_config_malformed_json_raises_epd_config_error(
        tmp_path, filename, parser_name, required_key):
    config_path = tmp_path / filename
    config_path.write_text("{ malformed json", encoding="utf-8")

    configurator = EPDConfigurator.__new__(EPDConfigurator)
    parser = getattr(configurator, parser_name)

    with pytest.raises(EPDConfigError):
        parser(str(config_path))


@pytest.mark.parametrize(
    "filename,parser_name,payload",
    [
        ("session_config.json", "parse_session_config", {}),
        ("usecase_config.json", "parse_usecase_config", {}),
        ("input_image_topic.json", "parse_inputimagetopic_config", {}),
    ],
)
def test_parse_config_missing_required_keys_raises_epd_config_error(
        tmp_path, filename, parser_name, payload):
    config_path = tmp_path / filename
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    configurator = EPDConfigurator.__new__(EPDConfigurator)
    parser = getattr(configurator, parser_name)

    with pytest.raises(EPDConfigError):
        parser(str(config_path))


@pytest.mark.parametrize(
    "filename,parser_name",
    [
        ("session_config.json", "parse_session_config"),
        ("usecase_config.json", "parse_usecase_config"),
        ("input_image_topic.json", "parse_inputimagetopic_config"),
    ],
)
def test_parse_config_empty_file_raises_epd_config_error(
        tmp_path, filename, parser_name):
    config_path = tmp_path / filename
    config_path.write_text("", encoding="utf-8")

    configurator = EPDConfigurator.__new__(EPDConfigurator)
    parser = getattr(configurator, parser_name)

    with pytest.raises(EPDConfigError):
        parser(str(config_path))


def test_set_ValidModel():

    # Create dummy model file in /data
    PATH_TO_DUMMY_MODEL = str(PACKAGE_ROOT / 'data/model/DUMMY_MODEL.onnx')
    (PACKAGE_ROOT / 'data/model/DUMMY_MODEL.onnx').touch()

    test_args = ['scripts/cli/config_epd.py', '--model', PATH_TO_DUMMY_MODEL]
    session_config_filepath = REQUIRED_START_DIR \
        + "/config/session_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load session_config.json
    f = open(session_config_filepath)
    data = json.load(f)
    model_path = data["path_to_model"]
    f.close()

    assert model_path == PATH_TO_DUMMY_MODEL

    # Remove dummy model file in /data
    (PACKAGE_ROOT / 'data/model/DUMMY_MODEL.onnx').unlink(missing_ok=True)


def test_set_InvalidModel():

    PATH_TO_INVALID_MODEL = str(PACKAGE_ROOT / 'data/model/NONEXISTENT_MODEL.onnx')

    test_args = ['scripts/cli/config_epd.py', '--model', PATH_TO_INVALID_MODEL]
    session_config_filepath = REQUIRED_START_DIR \
        + "/config/session_config.json"

    with pytest.raises(EPDConfigError):
        EPDConfigurator(REQUIRED_START_DIR, test_args)


def test_set_ValidLabelList():

    PATH_TO_VALID_LABEL_LIST = str(PACKAGE_ROOT / 'data/label_list/coco_classes.txt')

    test_args = [
        'scripts/cli/config_epd.py',
        '--label',
        PATH_TO_VALID_LABEL_LIST]
    session_config_filepath = REQUIRED_START_DIR \
        + "/config/session_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load session_config.json
    f = open(session_config_filepath)
    data = json.load(f)
    label_list_path = data["path_to_label_list"]
    f.close()

    assert label_list_path == PATH_TO_VALID_LABEL_LIST


def test_set_InvalidLabelList():

    PATH_TO_INVALID_LABEL_LIST = str(PACKAGE_ROOT / 'data/label_list/NONEXISTENT_LABEL_LIST.txt')

    test_args = [
        'scripts/cli/config_epd.py',
        '--label',
        PATH_TO_INVALID_LABEL_LIST]
    session_config_filepath = REQUIRED_START_DIR \
        + "/config/session_config.json"

    with pytest.raises(EPDConfigError):
        EPDConfigurator(REQUIRED_START_DIR, test_args)


def test_set_UseCase_Classification():

    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '0']
    usecase_config_filepath = REQUIRED_START_DIR \
        + "/config/usecase_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load usecase_config.json
    f = open(usecase_config_filepath)
    data = json.load(f)
    usecase_mode = data["usecase_mode"]
    f.close()

    assert usecase_mode == 0


def test_set_UseCase_Localization():

    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '3']
    usecase_config_filepath = REQUIRED_START_DIR \
        + "/config/usecase_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load usecase_config.json
    f = open(usecase_config_filepath)
    data = json.load(f)
    usecase_mode = data["usecase_mode"]
    f.close()

    assert usecase_mode == 3


def test_set_UseCase_Counting(monkeypatch):

    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '1']
    usecase_config_filepath = REQUIRED_START_DIR \
        + "/config/usecase_config.json"

    responses = iter(["2", "person", "dog"])
    monkeypatch.setattr('builtins.input', lambda _: next(responses))
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load usecase_config.json
    f = open(usecase_config_filepath)
    data = json.load(f)
    usecase_mode = data["usecase_mode"]
    count_class_list = data["class_list"]
    f.close()

    assert usecase_mode == 1
    assert len(count_class_list) == 2
    assert count_class_list[0] == "person"
    assert count_class_list[1] == "dog"


def test_set_UseCase_ColorMatching(monkeypatch, tmp_path):

    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '2']
    usecase_config_filepath = REQUIRED_START_DIR \
        + "/config/usecase_config.json"

    template_path = tmp_path / "orange.png"
    template_path.write_text("dummy", encoding="utf-8")
    responses = iter([str(template_path)])
    monkeypatch.setattr('builtins.input', lambda _: next(responses))
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load usecase_config.json
    f = open(usecase_config_filepath)
    data = json.load(f)
    usecase_mode = data["usecase_mode"]
    path_to_color_template = data["path_to_color_template"]
    f.close()

    assert usecase_mode == 2
    assert path_to_color_template == str(template_path)


def test_set_UseCase_Tracking(monkeypatch):

    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '4']
    usecase_config_filepath = REQUIRED_START_DIR \
        + "/config/usecase_config.json"

    responses = iter(["KCF"])
    monkeypatch.setattr('builtins.input', lambda _: next(responses))
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load usecase_config.json
    f = open(usecase_config_filepath)
    data = json.load(f)
    usecase_mode = data["usecase_mode"]
    track_type = data["track_type"]
    f.close()

    assert usecase_mode == 4
    assert track_type == "KCF"


def test_set_Invalid_UseCase():

    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '5']
    usecase_config_filepath = REQUIRED_START_DIR \
        + "/config/usecase_config.json"

    with pytest.raises(EPDConfigError):
        EPDConfigurator(REQUIRED_START_DIR, test_args)


def test_set_InputImageTopic():

    test_args = [
        'scripts/cli/config_epd.py',
        '--topic',
        '/virtual_camera/image_raw']
    inputimagetopic_config_filepath = REQUIRED_START_DIR \
        + "/config/input_image_topic.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    # Load usecase_config.json
    f = open(inputimagetopic_config_filepath)
    data = json.load(f)
    input_image_topic = data["input_image_topic"]
    f.close()

    assert input_image_topic == '/virtual_camera/image_raw'


def test_set_UseCase_Counting_WithClassListCli():

    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '1',
        '--class-list',
        'person, dog,  bottle  ']
    usecase_config_filepath = REQUIRED_START_DIR \
        + "/config/usecase_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    with open(usecase_config_filepath, encoding="utf-8") as f:
        data = json.load(f)

    assert data["usecase_mode"] == 1
    assert data["class_list"] == ["person", "dog", "bottle"]


def test_set_UseCase_Counting_InvalidClassList():

    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '1',
        '--class-list',
        ',  ,']

    with pytest.raises(EPDConfigError):
        EPDConfigurator(REQUIRED_START_DIR, test_args)


def test_set_UseCase_ColorMatching_WithTemplateAndMetric(tmp_path):

    template_path = tmp_path / "template.png"
    template_path.write_text("dummy", encoding="utf-8")
    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '2',
        '--color-template',
        str(template_path),
        '--color-hist-metric',
        '3']
    usecase_config_filepath = REQUIRED_START_DIR \
        + "/config/usecase_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    with open(usecase_config_filepath, encoding="utf-8") as f:
        data = json.load(f)

    assert data["usecase_mode"] == 2
    assert data["path_to_color_template"] == str(template_path)
    assert data["color_match_histogram_metric"] == "Bhattacharyya"


def test_set_UseCase_ColorMatching_InvalidTemplate():

    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '2',
        '--color-template',
        str(PACKAGE_ROOT / "data/missing_color_template.png")]

    with pytest.raises(EPDConfigError):
        EPDConfigurator(REQUIRED_START_DIR, test_args)


def test_set_UseCase_ColorMatching_InvalidHistogramMetric(tmp_path):

    template_path = tmp_path / "template.png"
    template_path.write_text("dummy", encoding="utf-8")

    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '2',
        '--color-template',
        str(template_path),
        '--color-hist-metric',
        'bogus']

    with pytest.raises(EPDConfigError):
        EPDConfigurator(REQUIRED_START_DIR, test_args)


def test_set_UseCase_Tracking_WithTrackTypeCli():

    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '4',
        '--track-type',
        'medianflow']
    usecase_config_filepath = REQUIRED_START_DIR \
        + "/config/usecase_config.json"

    EPDConfigurator(REQUIRED_START_DIR, test_args)

    with open(usecase_config_filepath, encoding="utf-8") as f:
        data = json.load(f)

    assert data["usecase_mode"] == 4
    assert data["track_type"] == "MEDIANFLOW"


def test_set_UseCase_Tracking_InvalidTrackType():

    test_args = [
        'scripts/cli/config_epd.py',
        '--use',
        '4',
        '--track-type',
        'invalid_tracker']

    with pytest.raises(EPDConfigError):
        EPDConfigurator(REQUIRED_START_DIR, test_args)
