from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QWidget


class _TrainWorkflowStepper(QObject):
    """Add a lightweight workflow guide to the refreshed Train window.

    This is presentation-only. It mirrors the existing TrainWindow readiness
    flags and never changes training state, validation rules, or job control.
    """

    _STAGES = (
        ("Model", "_is_model_ready"),
        ("Dataset", "_is_dataset_linked"),
        ("Labels", "_is_labellist_linked"),
        ("Annotations", "_is_dataset_labelled"),
        ("Train", None),
    )

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.labels = []
        self._timer = QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self.sync)

    def apply(self):
        content = self.window.findChild(QWidget, "trainContent")
        if content is None or not isinstance(content.layout(), QGridLayout):
            return self

        grid = content.layout()
        if content.findChild(QFrame, "trainWorkflowStepper") is not None:
            return self

        # Move the four existing cards down one row. Re-adding an existing
        # widget to a QGridLayout safely relocates it; no widget is recreated.
        model_card = grid.itemAtPosition(0, 0)
        dataset_card = grid.itemAtPosition(0, 1)
        params_card = grid.itemAtPosition(1, 0)
        readiness_card = grid.itemAtPosition(1, 1)

        model_widget = model_card.widget() if model_card else None
        dataset_widget = dataset_card.widget() if dataset_card else None
        params_widget = params_card.widget() if params_card else None
        readiness_widget = readiness_card.widget() if readiness_card else None

        stepper = QFrame(content)
        stepper.setObjectName("trainWorkflowStepper")
        row = QHBoxLayout(stepper)
        row.setContentsMargins(16, 10, 16, 10)
        row.setSpacing(8)

        for index, (name, _flag) in enumerate(self._STAGES, start=1):
            label = QLabel(f"{index}  {name}", stepper)
            label.setObjectName("trainWorkflowStep")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(34)
            label.setMinimumWidth(112 if name != "Annotations" else 130)
            row.addWidget(label, 1)
            self.labels.append(label)

            if index < len(self._STAGES):
                arrow = QLabel("→", stepper)
                arrow.setObjectName("trainWorkflowArrow")
                arrow.setAlignment(Qt.AlignCenter)
                arrow.setFixedWidth(16)
                row.addWidget(arrow, 0)

        stepper.setStyleSheet(
            """
            QFrame#trainWorkflowStepper {
                background-color: #151a22;
                border: 1px solid #293342;
                border-radius: 11px;
            }
            QLabel#trainWorkflowArrow {
                color: #536176;
                font-size: 13px;
                font-weight: 700;
            }
            """
        )

        grid.addWidget(stepper, 0, 0, 1, 2)
        if model_widget is not None:
            grid.addWidget(model_widget, 1, 0)
        if dataset_widget is not None:
            grid.addWidget(dataset_widget, 1, 1)
        if params_widget is not None:
            grid.addWidget(params_widget, 2, 0)
        if readiness_widget is not None:
            grid.addWidget(readiness_widget, 2, 1)

        self.sync()
        self._timer.start()
        return self

    def sync(self):
        states = []
        for name, flag in self._STAGES[:-1]:
            states.append(bool(getattr(self.window, flag, False)))

        # The Train step becomes available only when every actual prerequisite
        # flag is ready and the existing Train button is enabled.
        train_ready = all(states) and bool(self.window.train_button.isEnabled())
        stage_ready = states + [train_ready]

        first_incomplete = next(
            (index for index, ready in enumerate(stage_ready[:-1]) if not ready),
            4,
        )

        for index, label in enumerate(self.labels):
            if stage_ready[index]:
                text = f"✓  {self._STAGES[index][0]}"
                state = "complete"
            elif index == first_incomplete:
                text = f"{index + 1}  {self._STAGES[index][0]}"
                state = "active"
            elif index == 4 and all(states):
                text = "5  Train"
                state = "active"
            else:
                text = f"{index + 1}  {self._STAGES[index][0]}"
                state = "pending"

            label.setText(text)
            label.setProperty("state", state)
            self._style_step(label, state)

    @staticmethod
    def _style_step(label, state):
        if state == "complete":
            label.setStyleSheet(
                "color:#9de2b2; background:#14231a; border:1px solid #2b5b3a; "
                "border-radius:7px; padding:5px 8px; font-size:11px; font-weight:700;"
            )
        elif state == "active":
            label.setStyleSheet(
                "color:#dce5ff; background:#202b45; border:1px solid #5369a3; "
                "border-radius:7px; padding:5px 8px; font-size:11px; font-weight:700;"
            )
        else:
            label.setStyleSheet(
                "color:#778399; background:#121720; border:1px solid #293342; "
                "border-radius:7px; padding:5px 8px; font-size:11px; font-weight:600;"
            )


def apply_train_workflow_stepper(window):
    controller = _TrainWorkflowStepper(window)
    controller.apply()
    window._train_workflow_stepper = controller
    return controller
