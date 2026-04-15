from enum import Enum, auto

from PySide6.QtCore import QObject, Signal


class JobState(Enum):
    IDLE = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    FAILED = auto()


class JobController(QObject):
    """Small shared state container for long-running GUI jobs."""

    state_changed = Signal(object, str)

    def __init__(self, owner_name, parent=None):
        super().__init__(parent)
        self._owner_name = owner_name
        self._state = JobState.IDLE
        self._message = f'{owner_name} idle.'

    @property
    def state(self):
        return self._state

    @property
    def message(self):
        return self._message

    def set_state(self, state, message=None):
        if message is None:
            message = self._default_message(state)
        changed = (state != self._state) or (message != self._message)
        self._state = state
        self._message = message
        if changed:
            self.state_changed.emit(self._state, self._message)

    def _default_message(self, state):
        if state == JobState.IDLE:
            return f'{self._owner_name} idle.'
        if state == JobState.STARTING:
            return f'{self._owner_name} starting...'
        if state == JobState.RUNNING:
            return f'{self._owner_name} running.'
        if state == JobState.STOPPING:
            return f'{self._owner_name} stopping...'
        if state == JobState.FAILED:
            return f'{self._owner_name} failed.'
        return f'{self._owner_name} unknown state.'

    def is_busy(self):
        return self._state in (JobState.STARTING, JobState.RUNNING, JobState.STOPPING)
