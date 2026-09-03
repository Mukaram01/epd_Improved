from PySide6.QtCore import QEvent, QObject, QTimer


class _LegacyInlineStyleGuard(QObject):
    """Keep legacy state colours from overriding refreshed widget styling.

    Older Train/Deploy handlers still call ``setStyleSheet`` to paint entire
    controls red/green/white. The refreshed UI communicates state with badges
    and readiness chips instead. This event filter clears those legacy inline
    styles immediately after they are applied, without polling the widgets.
    """

    def __init__(self, parent, widgets):
        super().__init__(parent)
        self._widgets = tuple(widget for widget in widgets if widget is not None)
        self._widget_ids = {id(widget) for widget in self._widgets}
        self._pending_ids = set()

    def install(self):
        for widget in self._widgets:
            widget.installEventFilter(self)
            if widget.styleSheet():
                widget.setStyleSheet("")
        return self

    def eventFilter(self, obj, event):
        obj_id = id(obj)
        if (
                obj_id in self._widget_ids
                and event.type() == QEvent.StyleChange
                and obj.styleSheet()
                and obj_id not in self._pending_ids):
            self._pending_ids.add(obj_id)
            QTimer.singleShot(0, lambda widget=obj: self._clear(widget))
        return super().eventFilter(obj, event)

    def _clear(self, widget):
        self._pending_ids.discard(id(widget))
        if widget.styleSheet():
            widget.setStyleSheet("")


def install_legacy_style_guard(parent, widgets):
    guard = _LegacyInlineStyleGuard(parent, widgets)
    guard.install()
    return guard
