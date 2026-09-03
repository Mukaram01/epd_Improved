from types import MethodType

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QMessageBox


_EXTRA_REMEDIATIONS = {
    'EPD_ERR_BAD_BACKEND':
        'Use the automatic backend, or choose either local or Docker explicitly.',
    'EPD_ERR_LOCAL_RUNTIME_UNAVAILABLE':
        'Build the EPD workspace and ensure the local launch script is executable, then retry.',
    'EPD_ERR_DEPLOY_BACKEND_UNAVAILABLE':
        'Build the EPD workspace locally or install/configure Docker, then retry.',
}


def _reason_from_log(log_tail, error_code):
    if not error_code:
        return None
    prefix = f'{error_code}:'
    for line in log_tail.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split(':', 1)[1].strip() or None
    return None


def _remediation_from_log(log_tail, error_code):
    if not error_code:
        return None
    prefix = f'{error_code}_REMEDIATION:'
    for line in log_tail.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split(':', 1)[1].strip() or None
    return None


def _show_process_error(window, process_type):
    if process_type == 'kill' and window._stop_timeout_timer is not None:
        window._stop_timeout_timer.stop()

    label = 'Deployment' if process_type == 'deploy' else 'Stop'
    window._job_controller.set_state(
        window._job_controller.state.__class__.FAILED,
        f'{label} failed.')

    log_tail = window._tail_process_log(process_type)
    error_code = window._extract_epd_error_code(log_tail)
    reason = _reason_from_log(log_tail, error_code)
    remediation = _remediation_from_log(log_tail, error_code)
    if not remediation:
        remediation = _EXTRA_REMEDIATIONS.get(error_code)
    if not remediation:
        remediation = window._epd_remediation_message(error_code)

    if process_type == 'deploy':
        title = 'Deployment failed'
        primary = 'Could not start the perception pipeline.'
    else:
        title = 'Stop incomplete'
        primary = 'Could not stop the perception pipeline cleanly.'

    info_parts = []
    if reason:
        info_parts.append(f'Reason\n{reason}')
    elif error_code:
        info_parts.append(f'Reason\n{error_code}')
    if remediation:
        info_parts.append(f'Suggested action\n{remediation}')
    if not info_parts:
        info_parts.append('Open technical details for the process output.')

    details = []
    if error_code:
        details.append(f'Error code: {error_code}')
    details.append(f'Process: {process_type}')
    details.append('')
    details.append('Log output:')
    details.append(log_tail)

    message = QMessageBox(window)
    message.setIcon(QMessageBox.Critical)
    message.setWindowTitle(title)
    message.setText(primary)
    message.setInformativeText('\n\n'.join(info_parts))
    message.setDetailedText('\n'.join(details))
    message.setStandardButtons(QMessageBox.Ok)
    message.exec()


def _show_partial_cleanup_warning(window):
    log_tail = window._tail_process_log('kill')
    message = QMessageBox(window)
    message.setIcon(QMessageBox.Warning)
    message.setWindowTitle('Stop incomplete')
    message.setText('Perception stopped, but cleanup was only partially completed.')
    message.setInformativeText(
        'Some EPD processes may still be running. Check technical details before restarting the pipeline.'
    )
    message.setDetailedText(log_tail)
    message.setStandardButtons(QMessageBox.Ok)
    message.exec()


def install_deploy_runtime_ui(window):
    """Polish runtime presentation without changing deployment semantics."""

    def keep_run_button_compact(*_args):
        window.run_button.setIconSize(QSize(24, 24))
        window.run_button.setMinimumHeight(50)
        window.run_button.updateGeometry()

    # DeployWindow's legacy state callback resets the icon to 100x100 on every
    # state change. Connect after it so the refreshed UI always wins.
    window._job_controller.state_changed.connect(keep_run_button_compact)
    keep_run_button_compact()

    def handle_process_error(_self, process_type):
        _show_process_error(window, process_type)

    def show_partial_cleanup_warning(_self):
        _show_partial_cleanup_warning(window)

    window._handle_process_error = MethodType(handle_process_error, window)
    window._show_kill_partial_cleanup_warning = MethodType(
        show_partial_cleanup_warning, window)

    return keep_run_button_compact
