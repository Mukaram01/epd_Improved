from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QListWidget, QTextBrowser, QHBoxLayout, QVBoxLayout, QWidget


class HelpWindow(QWidget):
    """In-app EPD user guide window.

    Provides beginner-friendly explanations for training and deployment.
    Content is intentionally static so it remains available offline.
    """

    TOPICS = {
        "Getting Started": """
        <h2>Getting Started</h2>
        <p>EPD workflow:</p>
        <p>Camera → Dataset → Training → ONNX Model → Deployment → ROS 2 Output</p>
        """,
        "Training a Model": """
        <h2>Training a Model</h2>
        <p>Prepare images and annotations, select an architecture, validate the dataset,
        then start training.</p>
        """,
        "Faster R-CNN vs Mask R-CNN": """
        <h2>Model Selection</h2>
        <p><b>Faster R-CNN</b>: object detection using bounding boxes. Good when location
        is enough.</p>
        <p><b>Mask R-CNN</b>: adds pixel-level segmentation. Recommended when accurate
        object shape matters, such as robotic manipulation.</p>
        """,
        "Training Parameters": """
        <h2>Training Parameters</h2>
        <p><b>Max iterations</b>: number of learning steps performed.</p>
        <p><b>Checkpoint interval</b>: how often progress is saved.</p>
        <p><b>Learning rate steps</b>: controls how learning speed changes.</p>
        """,
        "Deployment": """
        <h2>Deployment</h2>
        <p>Select an ONNX model, matching labels, camera topic and runtime settings.</p>
        """,
        "Model Formats": """
        <h2>Model Formats</h2>
        <p>Deployment recommendation: ONNX.</p>
        <p>PyTorch .pt/.pth files are normally converted before deployment.</p>
        """,
        "Pretrained Models": """
        <h2>Pretrained Models</h2>
        <p>Models can be obtained from model repositories such as ONNX Model Zoo,
        then converted or fine-tuned for your application.</p>
        """,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EPD Help & Guides")
        self.resize(900, 600)

        layout = QHBoxLayout(self)

        self.topic_list = QListWidget()
        self.viewer = QTextBrowser()

        for topic in self.TOPICS:
            self.topic_list.addItem(topic)

        self.topic_list.currentTextChanged.connect(self.show_topic)

        layout.addWidget(self.topic_list, 1)
        layout.addWidget(self.viewer, 3)

        self.topic_list.setCurrentRow(0)

    def show_topic(self, topic):
        self.viewer.setHtml(self.TOPICS.get(topic, ""))
