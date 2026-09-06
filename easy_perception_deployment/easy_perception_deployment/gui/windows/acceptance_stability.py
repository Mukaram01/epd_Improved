"""Acceptance-run stability fixes for the EPD GUI.

This module is intentionally narrow. It keeps the current EPD-0..9 behaviour
while avoiding two acceptance findings:

* repeated Qt style repolishing that can make Deploy visibly flicker;
* use of rclpy's implicit/global executor from several GUI worker threads.

The ROS workers keep private contexts and explicit single-threaded executors so
one GUI subsystem cannot accidentally share the global executor with another.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import MethodType


_PATCHED = False


def _shutdown_executor(executor, node):
    if executor is None:
        return
    if node is not None:
        try:
            executor.remove_node(node)
        except Exception:
            pass
    try:
        executor.shutdown(timeout_sec=0.5)
    except TypeError:
        try:
            executor.shutdown()
        except Exception:
            pass
    except Exception:
        pass


def install_ros_executor_stability():
    """Patch GUI ROS workers before any worker instance is started."""
    global _PATCHED
    if _PATCHED:
        return

    from rclpy.executors import SingleThreadedExecutor

    from windows import Deploy as deploy_module
    from windows import epd2_productization as preview_module
    from windows import p3_inspector_worker as production_p3_module
    from windows import three_d_diagnostics as three_d_module

    def fps_run(self):
        if not deploy_module._RCLPY_AVAILABLE:
            self._set_latest_text('FPS: N/A | Latency: N/A (ROS unavailable)')
            return

        executor = None
        try:
            # Always use a private context/executor. rclpy.spin_once(node) would
            # otherwise use the process-global executor and is unsafe here
            # because Deploy has several independent ROS background threads.
            self._context = deploy_module.rclpy.context.Context()
            self._context.init(args=None)
            self._owns_context = True
            self._node = deploy_module.Node(
                'epd_fps_monitor',
                context=self._context,
            )
            executor = SingleThreadedExecutor(context=self._context)
            executor.add_node(self._node)
            self._update_subscription(self._usecase_mode)
            while (
                    self._running
                    and deploy_module.rclpy.ok(context=self._context)):
                self._maybe_update_subscription()
                executor.spin_once(timeout_sec=0.1)
        except Exception as exc:
            logging.getLogger('deploy').warning(
                'FPS monitor thread failed: %s', exc)
            self._set_latest_text('FPS: N/A | Latency: N/A (ROS error)')
        finally:
            _shutdown_executor(executor, self._node)
            if self._subscription is not None and self._node is not None:
                try:
                    self._node.destroy_subscription(self._subscription)
                except Exception as exc:
                    logging.getLogger('deploy').debug(
                        'FPS monitor: error destroying subscription: %s', exc)
                self._subscription = None
            if self._node is not None:
                try:
                    self._node.destroy_node()
                except Exception as exc:
                    logging.getLogger('deploy').debug(
                        'FPS monitor: error destroying node: %s', exc)
                self._node = None
            if self._owns_context and self._context is not None:
                try:
                    self._context.shutdown()
                except Exception as exc:
                    logging.getLogger('deploy').debug(
                        'FPS monitor: error during context shutdown: %s', exc)
            self._context = None

    def preview_run(self):
        if not preview_module._ROS_IMAGE_AVAILABLE:
            self._set_error(
                'ROS Python image support is unavailable. Source ROS 2 and the '
                'EPD workspace before launching the GUI.'
            )
            return

        context = None
        node = None
        executor = None
        subscriptions = []
        applied_revision = -1
        try:
            context = preview_module.rclpy.context.Context()
            context.init(args=None)
            node = preview_module.rclpy.create_node(
                'epd_live_perception_preview',
                context=context,
            )
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            while not self._stop_event.is_set() and context.ok():
                with self._lock:
                    revision = self._config_revision
                    config = dict(self._config)
                if revision != applied_revision:
                    for subscription in subscriptions:
                        node.destroy_subscription(subscription)
                    subscriptions = self._create_subscriptions(node, config)
                    applied_revision = revision
                executor.spin_once(timeout_sec=0.08)
        except Exception as exc:
            self._set_error(f'Live preview ROS subscriber failed: {exc}')
        finally:
            _shutdown_executor(executor, node)
            if node is not None:
                for subscription in subscriptions:
                    try:
                        node.destroy_subscription(subscription)
                    except Exception:
                        pass
                try:
                    node.destroy_node()
                except Exception:
                    pass
            if context is not None:
                try:
                    context.shutdown()
                except Exception:
                    pass

    def make_p3_run(localization_topic, tracking_topic, diagnostics_topic):
        def p3_run(self):
            try:
                import rclpy
                from diagnostic_msgs.msg import DiagnosticArray
                from epd_msgs.msg import EPDObjectLocalization, EPDObjectTracking
                from rclpy.executors import SingleThreadedExecutor as Executor
                from rclpy.node import Node
                from rclpy.qos import QoSProfile, ReliabilityPolicy
            except ImportError as exc:
                self._emit('unavailable', f'ROS 2 Python messages unavailable: {exc}')
                return

            executor = None
            try:
                context = rclpy.context.Context()
                context.init(args=None)
                self._context = context
                node = Node('epd_3d_diagnostics_gui', context=context)
                self._node = node
                executor = Executor(context=context)
                executor.add_node(node)
                qos = QoSProfile(depth=10)
                qos.reliability = ReliabilityPolicy.BEST_EFFORT
                node.create_subscription(
                    EPDObjectLocalization,
                    localization_topic,
                    lambda msg: self._emit(
                        'p3', three_d_module.summarize_p3_message(msg, False)),
                    qos,
                )
                node.create_subscription(
                    EPDObjectTracking,
                    tracking_topic,
                    lambda msg: self._emit(
                        'p3', three_d_module.summarize_p3_message(msg, True)),
                    qos,
                )
                node.create_subscription(
                    DiagnosticArray,
                    diagnostics_topic,
                    lambda msg: self._emit(
                        'diagnostics', three_d_module.diagnostics_values(msg)),
                    10,
                )
                self._emit(
                    'ready',
                    'Listening for Localization/Tracking results.',
                )
                while self._running and rclpy.ok(context=context):
                    executor.spin_once(timeout_sec=0.1)
            except Exception as exc:
                self._emit('unavailable', f'3D inspector ROS error: {exc}')
            finally:
                _shutdown_executor(executor, self._node)
                try:
                    if self._node is not None:
                        self._node.destroy_node()
                    if self._context is not None:
                        self._context.shutdown()
                except Exception:
                    pass
                self._node = None
                self._context = None

        return p3_run

    deploy_module.FPSMonitorThread._run = fps_run
    preview_module.LivePerceptionMonitor._run = preview_run

    p3_run = make_p3_run(
        production_p3_module.LOCALIZATION_TOPIC,
        production_p3_module.TRACKING_TOPIC,
        production_p3_module.DIAGNOSTICS_TOPIC,
    )
    production_p3_module.ProductionP3InspectorWorker._run = p3_run
    # The controller creates the legacy worker before EPD-6 swaps in the
    # production worker. Patch it as well so future direct use remains safe.
    three_d_module.P3InspectorWorker._run = p3_run

    def stable_preview_state(self, text, state):
        if self.state_badge.text() != text:
            self.state_badge.setText(text)
        if self.state_badge.property('previewState') != state:
            self.state_badge.setProperty('previewState', state)
            self.state_badge.style().unpolish(self.state_badge)
            self.state_badge.style().polish(self.state_badge)
            self.state_badge.update()

    def stable_placeholder(self, text):
        text = str(text)
        if self._image is None and self.text() == text:
            return
        self._image = None
        self.clear()
        self.setText(text)

    preview_module._LivePerceptionController._set_state = stable_preview_state
    preview_module._PreviewSurface.set_placeholder = stable_placeholder
    _PATCHED = True


def _set_text(widget, text):
    if widget.text() != text:
        widget.setText(text)


def _set_tooltip(widget, text):
    if widget.toolTip() != text:
        widget.setToolTip(text)


def _set_state(label, state):
    if state == 'ready':
        text = '✓ Ready'
        style_state = 'ready'
    elif state == 'blocked':
        text = '! Check'
        style_state = 'blocked'
    else:
        text = '• Checking'
        style_state = 'unknown'

    _set_text(label, text)
    if label.property('state') != style_state:
        label.setProperty('state', style_state)
        label.style().unpolish(label)
        label.style().polish(label)
        label.update()


def _stable_deploy_sync(self):
    """Mirror Deploy truth without repainting unchanged widgets."""
    w = self.window
    model_path = str(getattr(w, '_path_to_model', ''))
    labels_path = str(getattr(w, '_path_to_label_list', ''))
    model_name = Path(model_path).name or 'Not configured'
    labels_name = Path(labels_path).name or 'Not configured'
    topic = w.topic_button.currentText().strip() or 'Not configured'

    model_status = self._status_from_text(w.model_readiness_label.text())
    labels_status = self._status_from_text(w.label_list_readiness_label.text())
    topic_status = self._status_from_text(w.topic_readiness_label.text())

    for widget, text, tooltip in (
        (self.model_value, model_name, model_path),
        (self.labels_value, labels_name, labels_path),
        (self.topic_value, topic, topic),
        (self.readiness_model_value, model_name, model_path),
        (self.readiness_labels_value, labels_name, labels_path),
    ):
        _set_text(widget, text)
        _set_tooltip(widget, tooltip)

    _set_state(self.model_state, model_status)
    _set_state(self.labels_state, labels_status)
    _set_state(self.topic_state, topic_status)
    _set_state(self.readiness_model_state, model_status)
    _set_state(self.readiness_labels_state, labels_status)

    _set_text(
        w.visualize_button,
        'Visual output  •  On'
        if getattr(w, 'visualizeFlag', False)
        else 'Visual output  •  Off',
    )
    _set_text(
        w.segmentation_button,
        'Segmentation  •  On'
        if getattr(w, 'publish_detection_segmentation', False)
        else 'Segmentation  •  Off',
    )

    # EPD-8 is the single owner of docker_button text. Its compact backend
    # label carries richer truth (AUTO/CPU/CUDA/TensorRT + READY/CHECK/BLOCKED).
    # Do not overwrite that with the legacy useCPU-derived Device label.

    is_running = bool(getattr(w, '_is_running', False))
    if is_running:
        badge_text, badge_state, run_text = (
            'RUNNING', 'running', 'Stop Perception')
    elif w.run_button.isEnabled():
        badge_text, badge_state, run_text = (
            'READY', 'ready', 'Run Perception')
    else:
        badge_text, badge_state, run_text = (
            'SETUP REQUIRED', 'blocked', 'Run Perception')

    _set_text(self.header_badge, badge_text)
    _set_text(w.run_button, run_text)
    if self.header_badge.property('state') != badge_state:
        self.header_badge.setProperty('state', badge_state)
        self.header_badge.style().unpolish(self.header_badge)
        self.header_badge.style().polish(self.header_badge)
        self.header_badge.update()


def apply_deploy_ui_stability(main_window):
    """Replace high-frequency unconditional repainting with change-only sync."""
    controller = getattr(main_window, '_deploy_ui_controller', None)
    if controller is None or getattr(controller, '_acceptance_stability', False):
        return controller

    timer = getattr(controller, '_summary_timer', None)
    if timer is not None:
        timer.stop()
        try:
            timer.timeout.disconnect()
        except (RuntimeError, TypeError):
            pass

    controller.sync = MethodType(_stable_deploy_sync, controller)
    controller._acceptance_stability = True

    if timer is not None:
        # Readiness does not need a 300 ms full-card refresh. Event-driven
        # hooks still request immediate sync after operator actions.
        timer.setInterval(750)
        timer.timeout.connect(controller.sync)

    controller.sync()
    if timer is not None:
        timer.start()
    return controller
