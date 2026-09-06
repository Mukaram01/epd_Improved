from types import SimpleNamespace

from windows.acceptance_stability import _set_state, _stable_deploy_sync


class FakeStyle:
    def __init__(self):
        self.unpolish_calls = 0
        self.polish_calls = 0

    def unpolish(self, widget):
        self.unpolish_calls += 1

    def polish(self, widget):
        self.polish_calls += 1


class FakeWidget:
    def __init__(self, text="", enabled=True):
        self._text = text
        self._tooltip = ""
        self._properties = {}
        self._style = FakeStyle()
        self._enabled = enabled
        self.set_text_calls = 0
        self.update_calls = 0

    def text(self):
        return self._text

    def setText(self, text):
        self._text = text
        self.set_text_calls += 1

    def currentText(self):
        return self._text

    def toolTip(self):
        return self._tooltip

    def setToolTip(self, text):
        self._tooltip = text

    def property(self, name):
        return self._properties.get(name)

    def setProperty(self, name, value):
        self._properties[name] = value

    def style(self):
        return self._style

    def update(self):
        self.update_calls += 1

    def isEnabled(self):
        return self._enabled


def test_state_repolishes_only_when_style_state_changes():
    label = FakeWidget()

    _set_state(label, "ready")
    assert label.text() == "✓ Ready"
    assert label.style().unpolish_calls == 1
    assert label.style().polish_calls == 1

    _set_state(label, "ready")
    assert label.style().unpolish_calls == 1
    assert label.style().polish_calls == 1

    _set_state(label, "blocked")
    assert label.text() == "! Check"
    assert label.style().unpolish_calls == 2
    assert label.style().polish_calls == 2


def test_deploy_sync_does_not_repaint_unchanged_state():
    backend_label = "Backend  •  AUTO • CHECK"
    window = SimpleNamespace(
        _path_to_model="./data/model/MaskRCNN-10.onnx",
        _path_to_label_list="./data/label_list/coco_classes.txt",
        model_readiness_label=FakeWidget("Ready"),
        label_list_readiness_label=FakeWidget("Ready"),
        topic_readiness_label=FakeWidget("Configured"),
        topic_button=FakeWidget("/camera/camera/color/image_raw"),
        visualizeFlag=True,
        publish_detection_segmentation=True,
        useCPU=True,
        visualize_button=FakeWidget(),
        segmentation_button=FakeWidget(),
        docker_button=FakeWidget(backend_label),
        run_button=FakeWidget(enabled=True),
        _is_running=False,
    )
    controller = SimpleNamespace(
        window=window,
        model_value=FakeWidget(),
        labels_value=FakeWidget(),
        topic_value=FakeWidget(),
        readiness_model_value=FakeWidget(),
        readiness_labels_value=FakeWidget(),
        model_state=FakeWidget(),
        labels_state=FakeWidget(),
        topic_state=FakeWidget(),
        readiness_model_state=FakeWidget(),
        readiness_labels_state=FakeWidget(),
        header_badge=FakeWidget(),
        _status_from_text=lambda text: (
            "ready" if "ready" in text.lower() else "unknown"
        ),
    )

    _stable_deploy_sync(controller)
    first_header_polish = controller.header_badge.style().polish_calls
    first_model_polish = controller.model_state.style().polish_calls
    first_run_text_calls = window.run_button.set_text_calls

    _stable_deploy_sync(controller)

    assert controller.header_badge.style().polish_calls == first_header_polish
    assert controller.model_state.style().polish_calls == first_model_polish
    assert window.run_button.set_text_calls == first_run_text_calls
    assert window.docker_button.text() == backend_label
    assert window.docker_button.set_text_calls == 0
