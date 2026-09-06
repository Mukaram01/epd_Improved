from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QTextBrowser,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)


_ORIGINAL_DOCS = "https://easy-perception-deployment.readthedocs.io/en/latest/"


class HelpWindow(QWidget):
    """Offline-first EPD guide with clearly labelled upstream reference links."""

    TOPICS = {
        "Quick Start": (
            "<h2>Quick Start</h2>"
            "<p><b>EPD turns camera images into ROS 2 perception results.</b></p>"
            "<pre>Camera → Model → Mode → Validate → Run → ROS 2 output</pre>"
            "<ol>"
            "<li>Start your ROS 2 camera node.</li>"
            "<li>Open <b>Deploy</b>.</li>"
            "<li>Select an ONNX model and matching labels.</li>"
            "<li>Select or type the RGB image topic.</li>"
            "<li>Open <b>Camera Assistant</b> and verify camera health.</li>"
            "<li>Choose the perception mode.</li>"
            "<li>Check readiness, then run perception.</li>"
            "</ol>"
            "<p>RealSense RGB default: "
            "<code>/camera/camera/color/image_raw</code>.</p>"
            "<p>Press <b>F1</b> from Launcher, Train or Deploy to reopen Help.</p>"
        ),
        "Camera & ROS 2": (
            "<h2>Camera & ROS 2</h2>"
            "<p>Deploy scans ROS 2 for <code>sensor_msgs/msg/Image</code> topics.</p>"
            "<p><b>RealSense D435i defaults used by this fork:</b></p>"
            "<ul>"
            "<li>RGB: <code>/camera/camera/color/image_raw</code></li>"
            "<li>Depth: "
            "<code>/camera/camera/aligned_depth_to_color/image_raw</code></li>"
            "<li>CameraInfo: <code>/camera/camera/color/camera_info</code></li>"
            "</ul>"
            "<p><b>Detected</b> means the RGB topic appeared in the latest scan.</p>"
            "<p><b>Configured</b> means a saved topic exists but live discovery "
            "has not verified it yet.</p>"
        ),
        "Camera Assistant": (
            "<h2>Camera Assistant</h2>"
            "<p>EPD-1 adds a dedicated camera-health view. Open it from the "
            "Camera Input card or press <b>Ctrl+Shift+C</b> in Deploy.</p>"
            "<p>The assistant checks:</p>"
            "<ul>"
            "<li>ROS 2 graph availability and ROS distribution.</li>"
            "<li>Detected Image and CameraInfo topics.</li>"
            "<li>Whether RGB, depth and CameraInfo actually deliver a sample.</li>"
            "<li>Resolution, encoding, rate and message age where available.</li>"
            "</ul>"
            "<p><b>Live</b> means a message was sampled successfully.</p>"
            "<p><b>No sample</b> means the topic exists but did not deliver before "
            "the health-check timeout.</p>"
            "<p><b>Missing</b> means the expected topic is absent from the graph.</p>"
            "<p>Localization and Tracking require RGB, aligned depth and CameraInfo. "
            "For 2D modes, depth and CameraInfo are shown as optional.</p>"
            "<p>The assistant does not silently rewrite Deploy camera settings.</p>"
        ),
        "RealSense Setup": (
            "<h2>RealSense Setup</h2>"
            "<p>Localization and tracking need colour, aligned depth and CameraInfo.</p>"
            "<pre>ros2 topic list -t\n"
            "ros2 topic hz /camera/camera/color/image_raw\n"
            "ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw\n"
            "ros2 topic echo /camera/camera/color/camera_info --once</pre>"
            "<p>If the camera is stopped, EPD keeps the saved RGB topic and marks "
            "it as <b>Configured</b>.</p>"
        ),
        "Deploy Step-by-Step": (
            "<h2>Deploy Step-by-Step</h2>"
            "<ol>"
            "<li><b>Model:</b> choose the ONNX model.</li>"
            "<li><b>Labels:</b> choose the matching class list.</li>"
            "<li><b>Mode:</b> choose the required perception behaviour.</li>"
            "<li><b>Camera:</b> select a detected topic or type one manually.</li>"
            "<li><b>Camera Assistant:</b> verify the streams required by the mode.</li>"
            "<li><b>Detection overlay:</b> enable it for human visual checking.</li>"
            "<li><b>Object masks:</b> enable segmentation output when useful.</li>"
            "<li><b>Device:</b> start with CPU unless GPU is configured.</li>"
            "<li><b>Confidence:</b> 0.50 is a sensible starting point.</li>"
            "</ol>"
        ),
        "Perception Modes": (
            "<h2>Perception Modes</h2>"
            "<p><b>Classification:</b> base model inference.</p>"
            "<p><b>Counting:</b> count or filter selected object classes.</p>"
            "<p><b>Color-Matching:</b> compare detections with a colour template.</p>"
            "<p><b>Localization:</b> add 3D geometry using depth.</p>"
            "<p><b>Tracking:</b> localization plus persistent object IDs.</p>"
            "<p>For Workcell Studio manipulation, Localization and Tracking are "
            "usually the most useful modes.</p>"
        ),
        "Detection Overlay": (
            "<h2>Detection Overlay</h2>"
            "<p>This is the operator-facing name for the legacy "
            "<code>visualizeFlag</code>.</p>"
            "<p><b>On:</b> generate visualization output for human inspection.</p>"
            "<p><b>Off:</b> reduce visualization overhead.</p>"
            "<p>Turning it off does <b>not</b> stop ROS perception results.</p>"
        ),
        "Object Masks / Segmentation": (
            "<h2>Object Masks / Segmentation</h2>"
            "<p>Object masks expose segmentation-related output where supported.</p>"
            "<p>Use masks when precise shape matters, such as irregular objects "
            "or manipulation tasks.</p>"
            "<p>Mask R-CNN is the natural choice when instance masks are needed.</p>"
        ),
        "Confidence & Limits": (
            "<h2>Confidence & Detection Limits</h2>"
            "<p><b>Confidence threshold</b> is the minimum accepted score.</p>"
            "<ul>"
            "<li>Raise it to reduce false positives.</li>"
            "<li>Lower it if valid difficult objects are missed.</li>"
            "<li>Start around <b>0.50</b>, then tune on the real workcell.</li>"
            "</ul>"
            "<p><b>Max detections</b> caps detections processed per frame.</p>"
        ),
        "CPU, GPU & Transport": (
            "<h2>CPU, GPU & Image Transport</h2>"
            "<p><b>CPU</b> is the safest default.</p>"
            "<p><b>GPU</b> needs a deployment runtime configured for acceleration.</p>"
            "<p><b>raw</b> is simple and avoids compression work.</p>"
            "<p><b>compressed</b> saves network bandwidth but adds codec overhead.</p>"
        ),
        "Choosing a Model": (
            "<h2>Choosing a Model</h2>"
            "<p><b>Faster R-CNN:</b> bounding-box detection.</p>"
            "<p><b>Mask R-CNN:</b> instance segmentation when shape matters.</p>"
            "<p>Do not add model complexity that the application does not need.</p>"
            "<p>The deployed label list must match the model class order.</p>"
        ),
        "Model Formats": (
            "<h2>Model Formats</h2>"
            "<p><code>.onnx</code> — recommended EPD deployment format.</p>"
            "<p><code>.pt / .pth</code> — common PyTorch training formats.</p>"
            "<p><code>.engine</code> — TensorRT-specific optimized engine.</p>"
            "<p>Typical flow: train in PyTorch, export to ONNX, deploy in EPD.</p>"
        ),
        "Pretrained Models": (
            "<h2>Pretrained Models</h2>"
            "<p>Start from a trusted pretrained model when possible.</p>"
            "<p>Common sources include ONNX repositories, torchvision model "
            "collections and NVIDIA model resources.</p>"
            "<p>Verify input, outputs, labels and ONNX compatibility before use.</p>"
        ),
        "Training a Model": (
            "<h2>Training a Model</h2>"
            "<pre>Images → Labels → Validate → Train → Checkpoint → ONNX</pre>"
            "<ol>"
            "<li>Collect representative workcell images.</li>"
            "<li>Annotate classes consistently.</li>"
            "<li>Generate and validate the dataset.</li>"
            "<li>Choose the architecture.</li>"
            "<li>Train and inspect checkpoints.</li>"
            "<li>Export ONNX and deploy with the same labels.</li>"
            "</ol>"
        ),
        "Training Parameters": (
            "<h2>Training Parameters</h2>"
            "<p><b>Max iterations:</b> total optimizer learning steps.</p>"
            "<p><b>Checkpoint interval:</b> how often training state is saved.</p>"
            "<p>These are different: max iterations ends training; checkpoint "
            "interval controls recovery points.</p>"
            "<p><b>Learning-rate steps:</b> when learning speed changes.</p>"
            "<p><b>Test period:</b> how often evaluation runs.</p>"
        ),
        "ROS 2 Outputs": (
            "<h2>ROS 2 Outputs</h2>"
            "<p>EPD publishes mode-specific perception messages plus image and "
            "diagnostic outputs.</p>"
            "<pre>ros2 topic list\n"
            "ros2 topic hz /easy_perception_deployment/image_output\n"
            "ros2 topic echo "
            "/easy_perception_deployment/epd_tracking_output --once</pre>"
        ),
        "Troubleshooting": (
            "<h2>Troubleshooting</h2>"
            "<h3>No camera topics</h3>"
            "<p>Check ROS 2 sourcing, camera node state, and "
            "<code>ros2 topic list -t</code>.</p>"
            "<p>Open <b>Camera Assistant</b> for graph and live-sample checks.</p>"
            "<p>You can type the expected RGB topic manually.</p>"
            "<h3>Configured but not detected</h3>"
            "<p>The topic is saved but the latest ROS graph scan did not verify it.</p>"
            "<h3>Detected but no sample</h3>"
            "<p>The topic exists, but Camera Assistant could not receive a message "
            "before timeout. Check the camera publisher and topic rate.</p>"
            "<h3>No detections</h3>"
            "<p>Check camera, model, labels, object classes and confidence threshold.</p>"
        ),
        "EPD + Workcell Studio": (
            "<h2>EPD + Workcell Studio</h2>"
            "<p>EPD remains the perception subsystem.</p>"
            "<p>Workcell Studio owns scene, task, planning and simulation.</p>"
            "<pre>Camera → EPD → perceived objects → Workcell Studio / EMD → "
            "grasp → MoveIt</pre>"
            "<p>Tracking IDs help downstream systems reason across frames.</p>"
        ),
        "What Comes Next": (
            "<h2>Product Roadmap</h2>"
            "<p><b>EPD-0:</b> camera truth, clearer controls and practical Help.</p>"
            "<p><b>EPD-1:</b> Camera Assistant with RGB/depth/CameraInfo health.</p>"
            "<p><b>Next:</b> embedded live perception preview, smart model "
            "validation, better training visibility, profiles/replay and 3D tools.</p>"
        ),
        "Original EPD Documentation": (
            "<h2>Original EPD Documentation</h2>"
            "<p>This fork is inspired by the original Easy Perception Deployment "
            "project.</p>"
            "<p>The upstream docs are useful reference material, while this fork's "
            "local UI and repository docs describe current behaviour.</p>"
            f'<p><a href="{_ORIGINAL_DOCS}">{_ORIGINAL_DOCS}</a></p>'
        ),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EPD Help & Guides")
        self.resize(980, 680)
        self.setMinimumSize(760, 520)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(10)

        title = QLabel("EPD Help & Guides")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        subtitle = QLabel(
            "Offline guidance for training, camera setup, deployment and "
            "troubleshooting. The ReadTheDocs site is upstream reference material."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #9aa7b8;")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        body = QHBoxLayout()
        body.setSpacing(12)
        self.topic_list = QListWidget()
        self.topic_list.setMinimumWidth(245)
        self.viewer = QTextBrowser()
        self.viewer.setOpenExternalLinks(True)

        for topic in self.TOPICS:
            self.topic_list.addItem(topic)

        self.topic_list.currentTextChanged.connect(self.show_topic)
        body.addWidget(self.topic_list, 1)
        body.addWidget(self.viewer, 3)
        outer.addLayout(body, 1)

        footer = QLabel(
            f'Upstream reference: <a href="{_ORIGINAL_DOCS}">{_ORIGINAL_DOCS}</a>'
        )
        footer.setOpenExternalLinks(True)
        footer.setTextInteractionFlags(Qt.TextBrowserInteraction)
        outer.addWidget(footer)

        self.topic_list.setCurrentRow(0)

    def show_topic(self, topic):
        self.viewer.setHtml(self.TOPICS.get(topic, ""))

    def select_topic(self, topic):
        matches = self.topic_list.findItems(topic, Qt.MatchExactly)
        if matches:
            self.topic_list.setCurrentItem(matches[0])
        elif self.topic_list.count() > 0:
            self.topic_list.setCurrentRow(0)
