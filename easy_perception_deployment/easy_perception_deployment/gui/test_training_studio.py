import json
import os
import time

from windows.training_studio import (
    dataset_summary,
    list_checkpoints,
    parse_training_line,
    training_guidance,
)


def _write_coco(path, image_names, annotations, categories):
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": index + 1, "file_name": name}
                    for index, name in enumerate(image_names)
                ],
                "annotations": annotations,
                "categories": categories,
            }
        ),
        encoding="utf-8",
    )


def test_dataset_summary_counts_images_annotations_and_imbalance(tmp_path):
    root = tmp_path / "custom_dataset"
    train = root / "train_dataset"
    val = root / "val_dataset"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    for directory, names in ((train, ["a.jpg", "b.png"]), (val, ["c.jpg"])):
        for name in names:
            (directory / name).write_bytes(b"image")

    categories = [{"id": 1, "name": "part"}, {"id": 2, "name": "defect"}]
    train_annotations = [
        {"id": index, "image_id": 1, "category_id": 1}
        for index in range(1, 12)
    ] + [{"id": 20, "image_id": 2, "category_id": 2}]
    _write_coco(train / "annotations.json", ["a.jpg", "b.png"], train_annotations, categories)
    _write_coco(
        val / "annotations.json",
        ["c.jpg"],
        [{"id": 21, "image_id": 1, "category_id": 1}],
        categories,
    )

    summary = dataset_summary(root, ["part", "defect"])
    assert summary["valid"] is True
    assert summary["splits"]["train_dataset"]["images_on_disk"] == 2
    assert summary["splits"]["train_dataset"]["annotations"] == 12
    assert summary["class_counts"]["part"] == 12
    assert summary["class_counts"]["defect"] == 1
    assert any("imbalance" in warning.lower() for warning in summary["warnings"])


def test_dataset_summary_reports_missing_structure(tmp_path):
    root = tmp_path / "custom_dataset"
    root.mkdir()
    summary = dataset_summary(root, [])
    assert summary["valid"] is False
    assert any("train_dataset" in warning for warning in summary["warnings"])
    assert any("val_dataset" in warning for warning in summary["warnings"])


def test_parse_training_line_extracts_progress():
    event = parse_training_line(
        "eta: 0:12:34  iter: 420  loss: 0.4382  lr: 0.00025"
    )
    assert event["iteration"] == 420
    assert event["loss"] == 0.4382
    assert event["lr"] == 0.00025
    assert event["eta"] == "0:12:34"


def test_parse_training_line_extracts_validation_ap():
    event = parse_training_line(
        "Average Precision  (AP) @[ IoU=0.50:0.95 ] = 0.372"
    )
    assert event["validation_ap"] == 0.372


def test_training_guidance_does_not_invent_validation_evidence():
    history = [
        {"iteration": index * 100, "loss": 2.0 - index * 0.2}
        for index in range(1, 7)
    ]
    guidance = training_guidance(history, 1000)
    assert "Validation loss is not emitted" in guidance


def test_list_checkpoints_marks_last_checkpoint(tmp_path):
    gui = tmp_path
    weights = gui / "p2_trainer" / "weights" / "custom"
    weights.mkdir(parents=True)
    first = weights / "model_0000200.pth"
    second = weights / "model_0000400.pth"
    first.write_bytes(b"a")
    time.sleep(0.01)
    second.write_bytes(b"bb")
    (weights / "last_checkpoint").write_text(
        "weights/custom/model_0000400.pth\n",
        encoding="utf-8",
    )
    os.utime(second, None)

    records = list_checkpoints(gui, 2)
    assert records[0]["name"] == "model_0000400.pth"
    assert records[0]["iteration"] == 400
    assert records[0]["latest"] is True
