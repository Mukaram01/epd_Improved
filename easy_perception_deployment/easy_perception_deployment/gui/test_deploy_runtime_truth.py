from pathlib import Path

from windows.deploy_runtime_truth import (
    augment_local_cpu_readiness,
    filter_camera_source_topics,
    is_camera_source_topic,
)


def test_local_compiled_cpu_is_ready_without_docker_image():
    probe = {
        "ready": {"cpu": False, "cuda": False, "tensorrt": False},
        "compiled": {"cpu": True, "cuda": False, "tensorrt": False},
        "recommended": "cpu",
        "docker_ok": False,
    }

    result = augment_local_cpu_readiness(probe)

    assert result["ready"]["cpu"] is True
    assert result["local_provider_ready"]["cpu"] is True
    assert result["recommended"] == "cpu"


def test_unknown_compiled_provider_does_not_fake_cpu_readiness():
    probe = {
        "ready": {"cpu": False, "cuda": False, "tensorrt": False},
        "compiled": None,
        "recommended": "cpu",
    }

    result = augment_local_cpu_readiness(probe)

    assert result["ready"]["cpu"] is False
    assert result["local_provider_ready"]["cpu"] is False


def test_epd_owned_image_topics_are_not_camera_sources():
    assert is_camera_source_topic("/camera/camera/color/image_raw") is True
    assert is_camera_source_topic("/sim/camera/image") is True
    assert is_camera_source_topic("/easy_perception_deployment/image_output") is False
    assert is_camera_source_topic(
        "/easy_perception_deployment/ingress/color/image_raw"
    ) is False
    assert is_camera_source_topic("") is False


def test_camera_discovery_filters_epd_outputs_and_keeps_order():
    topics = [
        "/easy_perception_deployment/image_output",
        "/camera/camera/color/image_raw",
        "/easy_perception_deployment/ingress/color/image_raw",
        "/sim/camera/image",
        "/camera/camera/color/image_raw",
    ]

    assert filter_camera_source_topics(topics) == [
        "/camera/camera/color/image_raw",
        "/sim/camera/image",
    ]


def test_main_installs_stability_before_epd0_camera_truth():
    main_py = Path(__file__).with_name("main.py").read_text(encoding="utf-8")

    stability = main_py.index("apply_deploy_ui_stability(window1)")
    epd0 = main_py.index("apply_epd0_productization(window1)")
    runtime_truth = main_py.index("apply_deploy_runtime_truth(window1)")
    epd8 = main_py.index("apply_epd8_productization(window1)")

    assert stability < epd0 < runtime_truth < epd8
