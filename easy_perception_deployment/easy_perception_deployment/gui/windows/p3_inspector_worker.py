"""ROS worker for the EPD-6 3D Inspector using production topic names."""

import threading

from windows.three_d_diagnostics import (
    InspectorEvent,
    diagnostics_values,
    summarize_p3_message,
)


LOCALIZATION_TOPIC = "/easy_perception_deployment/epd_localize_output"
TRACKING_TOPIC = "/easy_perception_deployment/epd_tracking_output"
DIAGNOSTICS_TOPIC = "/easy_perception_deployment/inference_diagnostics"


class ProductionP3InspectorWorker:
    """Listen to existing P3 outputs without touching Qt from the ROS thread."""

    def __init__(self, events):
        self.events = events
        self._thread = None
        self._running = False
        self._context = None
        self._node = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="EPD3DInspector",
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        self._thread = None

    def _emit(self, kind, payload):
        self.events.put(InspectorEvent(kind, payload))

    def _run(self):
        try:
            import rclpy
            from diagnostic_msgs.msg import DiagnosticArray
            from epd_msgs.msg import EPDObjectLocalization, EPDObjectTracking
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy
        except ImportError as exc:
            self._emit("unavailable", f"ROS 2 Python messages unavailable: {exc}")
            return

        try:
            context = rclpy.context.Context()
            context.init(args=None)
            self._context = context
            node = Node("epd_3d_diagnostics_gui", context=context)
            self._node = node
            qos = QoSProfile(depth=10)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            node.create_subscription(
                EPDObjectLocalization,
                LOCALIZATION_TOPIC,
                lambda msg: self._emit("p3", summarize_p3_message(msg, False)),
                qos,
            )
            node.create_subscription(
                EPDObjectTracking,
                TRACKING_TOPIC,
                lambda msg: self._emit("p3", summarize_p3_message(msg, True)),
                qos,
            )
            node.create_subscription(
                DiagnosticArray,
                DIAGNOSTICS_TOPIC,
                lambda msg: self._emit("diagnostics", diagnostics_values(msg)),
                10,
            )
            self._emit("ready", "Listening for Localization/Tracking results.")
            while self._running and rclpy.ok(context=context):
                rclpy.spin_once(node, timeout_sec=0.1)
        except Exception as exc:
            self._emit("unavailable", f"3D inspector ROS error: {exc}")
        finally:
            try:
                if self._node is not None:
                    self._node.destroy_node()
                if self._context is not None:
                    self._context.shutdown()
            except Exception:
                pass
            self._node = None
            self._context = None
