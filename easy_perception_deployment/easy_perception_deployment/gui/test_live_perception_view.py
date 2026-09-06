from types import SimpleNamespace

from windows.epd2_productization import (
    OUTPUT_IMAGE_TOPIC,
    comparable_header_age_ms,
    object_count_from_message,
    qimage_from_packet,
    raw_packet_from_message,
    select_preview_source,
    transport_topic,
)


def _stamp(sec, nanosec=0):
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _header(sec, nanosec=0):
    return SimpleNamespace(stamp=_stamp(sec, nanosec))


def test_transport_topic_keeps_raw_and_appends_compressed():
    base = "/camera/camera/color/image_raw"
    assert transport_topic(base, "raw") == base
    assert transport_topic(base, "compressed") == base + "/compressed"
    assert transport_topic(base + "/compressed", "compressed") == base + "/compressed"


def test_preview_source_uses_camera_when_stopped():
    source = select_preview_source(
        "/camera/color/image_raw",
        "raw",
        perception_active=False,
        overlay_enabled=True,
    )
    assert source["source"] == "Camera RGB"
    assert source["topic"] == "/camera/color/image_raw"
    assert source["compressed"] is False


def test_preview_source_uses_output_when_running_with_overlay():
    source = select_preview_source(
        "/camera/color/image_raw",
        "compressed",
        perception_active=True,
        overlay_enabled=True,
    )
    assert source["source"] == "Detection overlay"
    assert source["topic"] == OUTPUT_IMAGE_TOPIC + "/compressed"
    assert source["compressed"] is True


def test_preview_source_falls_back_to_camera_when_overlay_off():
    source = select_preview_source(
        "/camera/color/image_raw",
        "raw",
        perception_active=True,
        overlay_enabled=False,
    )
    assert source["source"] == "Camera RGB"
    assert source["topic"] == "/camera/color/image_raw"


def test_object_count_supports_detection_and_3d_messages():
    detection = SimpleNamespace(class_indices=[1, 2, 3])
    localization = SimpleNamespace(objects=[object(), object()])
    unknown = SimpleNamespace()

    assert object_count_from_message(detection) == 3
    assert object_count_from_message(localization) == 2
    assert object_count_from_message(unknown) is None


def test_comparable_header_age_rejects_unrelated_clocks():
    message = SimpleNamespace(header=_header(100, 500_000_000))
    assert comparable_header_age_ms(message, now=101.0) == 500.0
    assert comparable_header_age_ms(message, now=1000.0) is None
    assert comparable_header_age_ms(
        SimpleNamespace(header=_header(0, 0)),
        now=100.0,
    ) is None


def test_raw_rgb_packet_converts_to_qimage():
    message = SimpleNamespace(
        width=2,
        height=1,
        step=6,
        encoding="rgb8",
        data=bytes([255, 0, 0, 0, 255, 0]),
    )
    packet = raw_packet_from_message(message)
    image, error = qimage_from_packet(packet)

    assert error == ""
    assert image is not None
    assert image.width() == 2
    assert image.height() == 1
    assert image.pixelColor(0, 0).red() == 255
    assert image.pixelColor(1, 0).green() == 255


def test_raw_bgr_packet_converts_without_cv_bridge():
    message = SimpleNamespace(
        width=1,
        height=1,
        step=3,
        encoding="bgr8",
        data=bytes([0, 0, 255]),
    )
    packet = raw_packet_from_message(message)
    image, error = qimage_from_packet(packet)

    assert error == ""
    assert image is not None
    assert image.pixelColor(0, 0).red() == 255


def test_raw_packet_rejects_short_data():
    message = SimpleNamespace(
        width=2,
        height=2,
        step=6,
        encoding="rgb8",
        data=bytes([0, 0, 0]),
    )
    assert raw_packet_from_message(message) is None
