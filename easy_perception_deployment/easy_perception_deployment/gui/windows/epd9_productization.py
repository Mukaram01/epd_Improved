"""EPD-9 in-app release/demo guidance."""

from PySide6.QtGui import QKeySequence, QShortcut


_RELEASE_HELP = (
    "<h2>Release & Demo</h2>"
    "<p>EPD-9 turns the EPD-0 → EPD-8 capabilities into a repeatable handoff "
    "workflow with acceptance evidence, diagnostics and known limitations.</p>"
    "<p><b>Reference sequence:</b></p>"
    "<pre>Camera Assistant → Model Manager → Tracking → Live Preview → "
    "3D Inspector → Profile → Replay → Workcell Contract → Acceptance</pre>"
    "<p>Recommended release command:</p>"
    "<pre>ros2 run easy_perception_deployment epd_release_acceptance.py "
    "--with-replay --output /tmp/epd_release_acceptance.json</pre>"
    "<p>Full repository guide: <code>docs/EPD_RELEASE_DEMO_GUIDE.md</code>.</p>"
)

_DIAGNOSTICS_HELP = (
    "<h2>Diagnostics Bundle</h2>"
    "<p>Create a read-only support bundle without changing EPD configuration:</p>"
    "<pre>ros2 run easy_perception_deployment epd_diagnostics_bundle.py "
    "--output /tmp/epd_diagnostics.zip</pre>"
    "<p>The bundle records available configs, recent GUI logs, ROS graph/output "
    "samples, backend capability and Docker/NVIDIA evidence. Missing commands or "
    "topics are recorded instead of making bundle creation fail.</p>"
    "<p>HOME/user strings are redacted by default. Review a bundle before sharing.</p>"
    "<p>A diagnostics bundle is engineering evidence, not a safety certificate.</p>"
)

_ACCEPTANCE_HELP = (
    "<h2>Acceptance & Limitations</h2>"
    "<p>Use <code>docs/EPD_ACCEPTANCE_CHECKLIST.md</code> on the actual target "
    "workstation and record PASS/WARN/FAIL with evidence.</p>"
    "<p>Review <code>docs/EPD_KNOWN_LIMITATIONS.md</code> before claiming a "
    "capability release-ready.</p>"
    "<ul>"
    "<li>CPU is the portable/recovery backend.</li>"
    "<li>CUDA/Jetson/TensorRT need target-specific evidence.</li>"
    "<li>Replay does not replace a live-camera smoke test.</li>"
    "<li>3D quality still depends on valid aligned depth/intrinsics.</li>"
    "<li>EPD health does not authorize robot motion.</li>"
    "</ul>"
)

_ROADMAP_HELP = (
    "<h2>Product Roadmap</h2>"
    "<p>The EPD productization sequence is now implemented through EPD-9:</p>"
    "<pre>EPD-0 camera truth\n"
    "EPD-1 camera health\n"
    "EPD-2 live perception view\n"
    "EPD-3 smart model manager\n"
    "EPD-4 training studio\n"
    "EPD-5 profiles + replay\n"
    "EPD-6 3D inspector\n"
    "EPD-7 Workcell contract\n"
    "EPD-8 performance backends\n"
    "EPD-9 release/demo evidence</pre>"
    "<p>Future work should be driven by measured product/workcell failures rather "
    "than adding capabilities without acceptance evidence.</p>"
)


def apply_epd9_productization(main_window):
    """Install EPD-9 help additions once."""
    existing = getattr(main_window, "_epd9_productization", None)
    if existing is not None:
        return existing

    help_window = main_window.help_window
    help_window.TOPICS["Release & Demo"] = _RELEASE_HELP
    help_window.TOPICS["Diagnostics Bundle"] = _DIAGNOSTICS_HELP
    help_window.TOPICS["Acceptance & Limitations"] = _ACCEPTANCE_HELP
    help_window.TOPICS["What Comes Next"] = _ROADMAP_HELP

    existing_topics = {
        help_window.topic_list.item(index).text()
        for index in range(help_window.topic_list.count())
    }
    for topic in ("Release & Demo", "Diagnostics Bundle", "Acceptance & Limitations"):
        if topic not in existing_topics:
            help_window.topic_list.addItem(topic)

    shortcut = QShortcut(QKeySequence("Ctrl+Shift+R"), main_window)
    shortcut.activated.connect(
        lambda: _open_release_help(main_window)
    )
    main_window._epd9_productization = shortcut
    return shortcut


def _open_release_help(main_window):
    main_window.help_window.select_topic("Release & Demo")
    main_window.help_window.show()
    main_window.help_window.raise_()
    main_window.help_window.activateWindow()
