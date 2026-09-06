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
    """Offline-first EPD user guide with links to the original upstream docs.

    The local topics explain the current fork's operator workflow. The ReadTheDocs link
    is labelled as original/upstream documentation because this fork has evolved beyond
    the older project in ROS version, runtime controls and Workcell Studio integration.
    """

    TOPICS = {
        "Quick Start": f"""
        <h2>Quick Start</h2>
        <p><b>EPD turns camera images into ROS 2 perception results.</b></p>
        <pre>Camera → Model → Perception mode → Validate → Run → ROS 2 output</pre>
        <ol>
          <li>Start your ROS 2 camera node.</li>
          <li>Open <b>Deploy</b>.</li>
          <li>Select an ONNX model and the matching label list.</li>
          <li>Select or type the RGB image topic.</li>
          <li>Choose Classification, Counting, Color-Matching, Localization or Tracking.</li>
          <li>Check the readiness panel, then run perception.</li>
        </ol>
        <p>For a RealSense D435i, the common RGB topic is
        <code>/camera/camera/color/image_raw</code>.</p>
        <p><b>Tip:</b> press <b>F1</b> from the launcher, Train window or Deploy window to reopen this guide.</p>
        <p><a href="{_ORIGINAL_DOCS}">Read the original Easy Perception Deployment documentation</a></p>
        """,
        "Camera & ROS 2": """
        <h2>Camera & ROS 2</h2>
        <p>The Deploy window scans the ROS 2 graph for <code>sensor_msgs/msg/Image</code> topics.</p>
        <p><b>RealSense D435i defaults used by this fork:</b></p>
        <ul>
          <li>RGB: <code>/camera/camera/color/image_raw</code></li>
          <li>Aligned depth: <code>/camera/camera/aligned_depth_to_color/image_raw</code></li>
          <li>CameraInfo: <code>/camera/camera/color/camera_info</code></li>
        </ul>
        <p>The main Deploy selector is the <b>RGB input topic</b>. Localization and tracking also rely on
        depth and camera calibration through the runtime pipeline.</p>
        <h3>Configured vs detected</h3>
        <p><b>Detected</b> means the selected RGB topic appeared in the latest ROS 2 topic scan.</p>
        <p><b>Configured</b> means EPD has a saved topic, but live topic discovery did not verify it yet.
        This is useful when you configure EPD before starting the camera.</p>
        """,
        "RealSense Setup": """
        <h2>RealSense Setup</h2>
        <p>For 3D localization/tracking, EPD expects colour, aligned depth and CameraInfo to be available.</p>
        <p>Typical checks:</p>
        <pre>ros2 topic list -t
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/camera/color/camera_info --once</pre>
        <p>If the RGB topic is saved but the camera is currently stopped, EPD will show it as
        <b>Configured</b> rather than falsely claiming that it is live.</p>
        """,
        "Deploy Step-by-Step": """
        <h2>Deploy Step-by-Step</h2>
        <ol>
          <li><b>Model:</b> choose the ONNX model used for inference.</li>
          <li><b>Labels:</b> choose the text label list that matches the model classes and order.</li>
          <li><b>Mode:</b> select the perception behaviour needed by the application.</li>
          <li><b>Camera:</b> select a detected RGB topic or type one manually.</li>
          <li><b>Detection overlay:</b> enable it when a human needs visual output; disable it to reduce overhead.</li>
          <li><b>Object masks:</b> enable segmentation-related output where supported.</li>
          <li><b>Device:</b> start with CPU unless GPU support is already configured.</li>
          <li><b>Confidence:</b> 0.50 is a reasonable starting point; tune from observed false positives/misses.</li>
          <li>Run only after the readiness panel matches the intended configuration.</li>
        </ol>
        """,
        "Perception Modes": """
        <h2>Perception Modes</h2>
        <table cellspacing="8">
          <tr><td><b>Classification</b></td><td>Run the base model inference path.</td></tr>
          <tr><td><b>Counting</b></td><td>Count/filter selected object classes.</td></tr>
          <tr><td><b>Color-Matching</b></td><td>Compare detected objects against a reference colour template.</td></tr>
          <tr><td><b>Localization</b></td><td>Add 3D geometry/position information using depth.</td></tr>
          <tr><td><b>Tracking</b></td><td>Localization plus persistent object IDs across frames.</td></tr>
        </table>
        <p>For Workcell Studio manipulation, <b>Localization</b> and <b>Tracking</b> are normally the most
        relevant because downstream grasp planning needs 3D scene information.</p>
        """,
        "Detection Overlay": """
        <h2>Detection Overlay</h2>
        <p>This is the current user-facing name for the legacy <code>visualizeFlag</code>.</p>
        <p><b>On:</b> generate visualization output for human inspection.</p>
        <p><b>Off:</b> reduce visualization overhead. This does <b>not</b> mean that ROS perception results
        stop publishing.</p>
        <p>For high-rate robot pipelines, turning the overlay off can be useful after the system is validated.</p>
        """,
        "Object Masks / Segmentation": """
        <h2>Object Masks / Segmentation</h2>
        <p>Object masks expose segmentation-related per-object output where the selected model/runtime supports it.</p>
        <p>Use masks when precise object shape matters, for example irregular objects or manipulation where a
        bounding box is not enough. Masks can increase compute and message bandwidth.</p>
        <p>Mask R-CNN is the natural choice when instance masks are needed.</p>
        """,
        "Confidence & Detection Limits": """
        <h2>Confidence & Detection Limits</h2>
        <p><b>Confidence threshold</b> is the minimum score accepted as a detection.</p>
        <ul>
          <li>Raise it to reduce false positives.</li>
          <li>Lower it if difficult real objects are being missed.</li>
          <li>Start around <b>0.50</b>, then tune using your own camera and environment.</li>
        </ul>
        <p><b>Max detections</b> limits how many detections are processed per frame. Use a sensible cap for
        crowded scenes when runtime predictability matters.</p>
        """,
        "CPU, GPU & Image Transport": """
        <h2>CPU, GPU & Image Transport</h2>
        <p><b>CPU</b> is the safest default because it requires the least deployment-specific acceleration setup.</p>
        <p><b>GPU</b> should be used only when the EPD runtime/container and inference backend are configured for it.</p>
        <p><b>raw image transport</b> is simple and avoids compression work.</p>
        <p><b>compressed image transport</b> can reduce network bandwidth but adds encode/decode overhead.</p>
        """,
        "Choosing a Model": """
        <h2>Choosing a Model</h2>
        <p>Choose the model according to the information the robot actually needs.</p>
        <ul>
          <li><b>Faster R-CNN:</b> bounding-box detection when object extent is sufficient.</li>
          <li><b>Mask R-CNN:</b> instance segmentation when object shape/boundary matters.</li>
          <li><b>Classification models:</b> image/category classification rather than object geometry.</li>
        </ul>
        <p>The selected label list must match the model class order.</p>
        """,
        "Faster R-CNN vs Mask R-CNN": """
        <h2>Faster R-CNN vs Mask R-CNN</h2>
        <p><b>Faster R-CNN</b> predicts object classes and bounding boxes. Choose it when you need reliable
        detection but not a pixel-accurate silhouette.</p>
        <p><b>Mask R-CNN</b> adds an instance mask for each object. Choose it when manipulation, irregular
        shapes or precise boundaries make segmentation valuable.</p>
        <p>Mask output can be more expensive, so do not enable complexity that the application does not need.</p>
        """,
        "Model Formats": """
        <h2>Model Formats</h2>
        <table cellspacing="8">
          <tr><td><code>.onnx</code></td><td>Recommended EPD deployment format.</td></tr>
          <tr><td><code>.pt / .pth</code></td><td>Typical PyTorch training/checkpoint formats; convert/export before EPD deployment.</td></tr>
          <tr><td><code>.engine</code></td><td>TensorRT-specific optimized engine; not the general EPD interchange format.</td></tr>
        </table>
        <p>ONNX is useful because it separates deployment from the original training framework.</p>
        """,
        "Pretrained Models": """
        <h2>Pretrained Models</h2>
        <p>Start from a trusted pretrained model when possible, then fine-tune on your own labelled data.</p>
        <p>Common sources include ONNX model repositories, PyTorch/torchvision model collections and NVIDIA model resources.</p>
        <p>Always verify architecture, expected input, output contract, labels and ONNX compatibility before deployment.</p>
        """,
        "Training a Model": """
        <h2>Training a Model</h2>
        <pre>Images → Annotations → Dataset generation → Validation → Train → Checkpoint → Export ONNX</pre>
        <ol>
          <li>Collect images that represent the real workcell conditions.</li>
          <li>Annotate every class consistently.</li>
          <li>Generate the training dataset in the format expected by EPD.</li>
          <li>Validate labels and dataset structure before training.</li>
          <li>Choose the architecture that matches the required output.</li>
          <li>Train, inspect checkpoints, and deploy the exported ONNX model with the same class list.</li>
        </ol>
        """,
        "Training Parameters": """
        <h2>Training Parameters</h2>
        <p><b>Max iterations:</b> total number of optimizer learning steps. Too few may under-train; too many
        can waste time or overfit.</p>
        <p><b>Checkpoint interval:</b> how often the training state is saved. It is not an alternative to max
        iterations: max iterations controls when training ends; checkpoint interval controls when recovery points are written.</p>
        <p><b>Learning-rate steps:</b> when the optimizer learning rate changes.</p>
        <p><b>Test period:</b> how often evaluation runs during training.</p>
        """,
        "Dataset & Labels": """
        <h2>Dataset & Labels</h2>
        <p>Training quality depends more on representative, consistent data than on simply increasing iteration count.</p>
        <ul>
          <li>Include lighting, backgrounds, orientations and partial occlusions expected in the real workcell.</li>
          <li>Keep class names and annotation policy consistent.</li>
          <li>Keep a validation set separate from training data.</li>
          <li>Use the same label ordering when deploying the exported model.</li>
        </ul>
        """,
        "ROS 2 Outputs": """
        <h2>ROS 2 Outputs</h2>
        <p>EPD publishes mode-specific ROS 2 perception messages plus image/diagnostic outputs. This fork also
        supports normalized live/replay workflows used by Workcell Studio integration.</p>
        <p>Useful inspection commands include:</p>
        <pre>ros2 topic list
ros2 topic hz /easy_perception_deployment/image_output
ros2 topic echo /easy_perception_deployment/epd_tracking_output --once</pre>
        <p>The exact output topic depends on the selected perception mode.</p>
        """,
        "Troubleshooting": """
        <h2>Troubleshooting</h2>
        <h3>No camera topics</h3>
        <ul>
          <li>Confirm ROS 2 and the workspace are sourced.</li>
          <li>Confirm the camera node is running.</li>
          <li>Use <code>ros2 topic list -t</code>.</li>
          <li>You can type the expected RGB topic manually; EPD preserves it when discovery fails.</li>
        </ul>
        <h3>Configured but not detected</h3>
        <p>The topic is saved, but the latest ROS graph scan did not verify it. This can be normal if the camera
        is started later. Refresh after the camera is live.</p>
        <h3>No detections</h3>
        <ul>
          <li>Verify the model and matching labels.</li>
          <li>Check the RGB stream.</li>
          <li>Lower the confidence threshold temporarily.</li>
          <li>Confirm the model classes actually include the object you expect.</li>
        </ul>
        """,
        "EPD + Workcell Studio": """
        <h2>EPD + Workcell Studio</h2>
        <p>EPD remains the perception subsystem. Workcell Studio owns scene/workcell definition, task definition,
        planning and simulation.</p>
        <pre>Camera → EPD perception → normalized perceived objects → Workcell Studio / EMD → grasp planning → MoveIt</pre>
        <p>Tracking is particularly useful because stable IDs help downstream systems reason about the same object across frames.</p>
        """,
        "What Comes Next": """
        <h2>Product Roadmap</h2>
        <p><b>EPD-0 (this increment):</b> truthful camera state, preserved manual/saved topics, clearer runtime controls,
        F1 help, and practical deployment guidance.</p>
        <p><b>Next:</b> Camera Assistant with RGB/depth/CameraInfo health, then embedded live perception preview,
        smart model validation, improved training observability, saved profiles/replay, stronger 3D tools and the
        normalized Workcell Studio bridge.</p>
        """,
        "Original EPD Documentation": f"""
        <h2>Original Easy Perception Deployment Documentation</h2>
        <p>This fork is inspired by the original Easy Perception Deployment project. The upstream documentation is
        valuable for the original concepts and workflows, but this fork has evolved and its local UI/runtime behaviour
        is the source of truth for current features.</p>
        <p><a href="{_ORIGINAL_DOCS}">{_ORIGINAL_DOCS}</a></p>
        """,
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
            "Offline guidance for training, camera setup, deployment and troubleshooting. "
            "Use the original ReadTheDocs site as upstream reference material."
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
