"""Target-object picker: which label-list classes Localization/Tracking publish."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


def target_objects_summary(target_classes, limit=3):
    if not target_classes:
        return 'All classes'
    shown = ', '.join(target_classes[:limit])
    extra = len(target_classes) - limit
    return f'{shown} +{extra}' if extra > 0 else shown


class TargetObjectsDialog(QDialog):
    """Searchable checklist built from the active label list."""

    def __init__(self, label_names, selected, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Target objects')
        self.setMinimumSize(360, 460)
        # Match the dark Deploy window (its stylesheet does not cascade here).
        self.setStyleSheet(
            """
            QDialog { background-color: #101319; color: #f4f7fb; }
            QLabel { color: #c9d1dd; }
            QLineEdit, QListWidget {
                background-color: #171c24; color: #f4f7fb;
                border: 1px solid #2b3441; border-radius: 7px; padding: 5px;
            }
            QListWidget::item { padding: 3px; }
            QListWidget::indicator {
                width: 14px; height: 14px;
                border: 1px solid #6b7b94; border-radius: 3px;
                background-color: #101319;
            }
            QListWidget::indicator:checked {
                background-color: #5367d8; border-color: #8796ee;
            }
            QPushButton {
                background-color: #1d2430; color: #f4f7fb;
                border: 1px solid #344052; border-radius: 7px; padding: 6px 12px;
            }
            QPushButton:hover { border-color: #526176; }
            """)

        layout = QVBoxLayout(self)
        intro = QLabel(
            'Tick the objects Localization and Tracking should report as pick '
            'targets. Other detections stay visible in readiness diagnostics '
            'but are never published. Leave everything unticked to publish '
            'all classes.')
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.search = QLineEdit(self)
        self.search.setPlaceholderText('Search objects, e.g. banana')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)

        self.list_widget = QListWidget(self)
        selected_set = set(selected or [])
        for name in label_names:
            item = QListWidgetItem(name, self.list_widget)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in selected_set else Qt.Unchecked)
        self.list_widget.itemChanged.connect(self._update_count)
        layout.addWidget(self.list_widget, 1)

        footer = QHBoxLayout()
        self.count_label = QLabel(self)
        footer.addWidget(self.count_label, 1)
        clear_button = QPushButton('All classes', self)
        clear_button.setToolTip('Clear the selection so every class is published.')
        clear_button.clicked.connect(self._clear)
        footer.addWidget(clear_button)
        layout.addLayout(footer)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_count()

    def selected(self):
        """Checked names in label-list order."""
        return [
            self.list_widget.item(row).text()
            for row in range(self.list_widget.count())
            if self.list_widget.item(row).checkState() == Qt.Checked
        ]

    def _filter(self, text):
        needle = text.strip().lower()
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _clear(self):
        for row in range(self.list_widget.count()):
            self.list_widget.item(row).setCheckState(Qt.Unchecked)

    def _update_count(self, *_):
        selected = self.selected()
        self.count_label.setText(
            f'{len(selected)} selected' if selected else 'All classes will be published')
