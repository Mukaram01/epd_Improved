"""EPD-4 productization entry point."""

from windows.training_studio import apply_training_studio


def apply_epd4_productization(main_window):
    """Install the EPD-4 Training Studio on the current launcher."""
    return apply_training_studio(main_window)
