from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QFrame, QWidget


def apply_deploy_ui_scale(window):
    """Apply a readability pass after the Deploy presentation refresh.

    This keeps behaviour untouched while making the operational screen easier
    to read on a normal workstation. The window still clamps to the usable
    desktop and the existing scroll area remains the small-screen fallback.
    """

    screen = QApplication.primaryScreen()
    available = screen.availableGeometry() if screen else None

    preferred = QSize(1080, 820)
    if available is not None:
        max_width = max(860, int(available.width() * 0.96))
        max_height = max(650, int(available.height() * 0.94))
        width = min(max(preferred.width(), 860), max_width)
        height = min(max(preferred.height(), 650), max_height)
        min_width = min(860, max_width)
        min_height = min(650, max_height)
    else:
        width, height = preferred.width(), preferred.height()
        min_width, min_height = 860, 650

    window.setMinimumSize(min_width, min_height)

    # Do not shrink a user-resized window, but raise older/smaller remembered
    # sizes to the new readability floor.
    window.resize(max(window.width(), width), max(window.height(), height))

    # Give the grouped cards and page chrome a little more breathing room.
    for frame in window.findChildren(QFrame):
        if frame.objectName() == "sectionCard" and frame.layout() is not None:
            frame.layout().setContentsMargins(20, 18, 20, 18)
            frame.layout().setSpacing(12)
        elif frame.objectName() == "deployHeader" and frame.layout() is not None:
            frame.layout().setContentsMargins(28, 22, 28, 20)
        elif frame.objectName() == "deployFooter" and frame.layout() is not None:
            frame.layout().setContentsMargins(26, 14, 26, 14)

    content = window.findChild(QWidget, "deployContent")
    if content is not None and content.layout() is not None:
        content.layout().setContentsMargins(26, 22, 26, 22)
        content.layout().setHorizontalSpacing(18)
        content.layout().setVerticalSpacing(18)

    # Larger interaction targets pair with the larger typography.
    for widget in (
        window.model_button,
        window.list_button,
        window.visualize_button,
        window.segmentation_button,
        window.docker_button,
        window.usecase_config_button,
        window.topic_button,
        window.transport_combo,
        window.confidence_spinbox,
        window.max_detections_spinbox,
    ):
        widget.setMinimumHeight(42)

    window.topic_button.setFixedHeight(42)
    window.refresh_topics_button.setMinimumHeight(42)
    window.use_defaults_button.setMinimumHeight(38)
    window.run_button.setMinimumHeight(52)
    window.run_button.setIconSize(QSize(26, 26))

    # Append targeted overrides so the existing visual language remains intact.
    window.setStyleSheet(
        window.styleSheet()
        + """

        QLabel#deployEyebrow {
            font-size: 10px;
        }

        QLabel#deployTitle {
            font-size: 25px;
        }

        QLabel#deploySubtitle {
            font-size: 13px;
        }

        QLabel#headerBadge {
            font-size: 10px;
            padding: 6px 12px;
        }

        QLabel#sectionTitle {
            font-size: 10px;
        }

        QLabel#sectionSubtitle {
            font-size: 11px;
        }

        QLabel#fieldLabel,
        QLabel#summaryValue {
            font-size: 12px;
        }

        QLabel#summaryState {
            font-size: 10px;
            padding: 5px 8px;
        }

        QLabel#validationMessage {
            font-size: 11px;
            padding: 9px 11px;
        }

        QPushButton,
        QComboBox,
        QDoubleSpinBox,
        QSpinBox {
            font-size: 12px;
            padding: 7px 10px;
        }

        QLabel#runtimeStatus {
            font-size: 12px;
        }

        QLabel#runtimeMetrics {
            font-size: 11px;
        }

        QPushButton#runButton {
            font-size: 13px;
            padding: 10px 20px;
        }
        """
    )
