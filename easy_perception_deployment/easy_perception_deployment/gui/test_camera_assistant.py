from windows.camera_assistant import (
    CAMERA_INFO_TYPE,
    DEFAULT_CAMERA_INFO_TOPIC,
    DEFAULT_DEPTH_TOPIC,
    DEFAULT_RGB_TOPIC,
    IMAGE_TYPE,
    infer_camera_topics,
    parse_average_rate,
    parse_sample_metadata,
    parse_topic_list,
)


def test_parse_topic_list_extracts_sensor_types():
    output = """
/camera/camera/color/image_raw [sensor_msgs/msg/Image]
/camera/camera/color/camera_info [sensor_msgs/msg/CameraInfo]
/chatter [std_msgs/msg/String]
"""
    parsed = parse_topic_list(output)
    assert parsed[DEFAULT_RGB_TOPIC] == IMAGE_TYPE
    assert parsed[DEFAULT_CAMERA_INFO_TOPIC] == CAMERA_INFO_TYPE
    assert parsed["/chatter"] == "std_msgs/msg/String"


def test_infer_camera_topics_prefers_realsense_defaults():
    topics = {
        DEFAULT_RGB_TOPIC: IMAGE_TYPE,
        DEFAULT_DEPTH_TOPIC: IMAGE_TYPE,
        DEFAULT_CAMERA_INFO_TOPIC: CAMERA_INFO_TYPE,
    }
    inferred = infer_camera_topics(topics, DEFAULT_RGB_TOPIC)
    assert inferred["rgb"] == DEFAULT_RGB_TOPIC
    assert inferred["depth"] == DEFAULT_DEPTH_TOPIC
    assert inferred["camera_info"] == DEFAULT_CAMERA_INFO_TOPIC


def test_infer_camera_topics_derives_custom_namespace():
    rgb = "/cell/cam/color/image_raw"
    depth = "/cell/cam/aligned_depth_to_color/image_raw"
    info = "/cell/cam/color/camera_info"
    topics = {
        rgb: IMAGE_TYPE,
        depth: IMAGE_TYPE,
        info: CAMERA_INFO_TYPE,
    }
    inferred = infer_camera_topics(topics, rgb)
    assert inferred["depth"] == depth
    assert inferred["camera_info"] == info


def test_parse_sample_metadata_extracts_resolution_encoding_and_stamp():
    output = """
header:
  stamp:
    sec: 100
    nanosec: 500000000
  frame_id: camera_color_optical_frame
height: 480
width: 640
encoding: rgb8
"""
    metadata = parse_sample_metadata(output)
    assert metadata["width"] == 640
    assert metadata["height"] == 480
    assert metadata["encoding"] == "rgb8"
    assert metadata["stamp_seconds"] == 100.5


def test_parse_average_rate_uses_latest_measurement():
    output = "average rate: 29.82\naverage rate: 30.04\n"
    assert parse_average_rate(output) == 30.04
    assert parse_average_rate("no rate yet") is None
