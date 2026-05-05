"""Smoke tests for widgets extracted from the legacy GUI module."""

from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")

    app = QApplication.instance()
    if app is not None:
        return app
    try:
        return QApplication([])
    except Exception as exc:
        pytest.skip(f"QApplication unavailable: {exc}")


def _theme() -> dict[str, str]:
    return {
        "bg": "#1e1e2e",
        "fg": "#cdd6f4",
        "sidebar_bg": "#181825",
    }


def test_processing_panel_imports_from_new_module_and_gui_package(qapp):
    from probeflow.gui import ProcessingControlPanel as PublicPanel
    from probeflow.gui.processing import ProcessingControlPanel

    assert PublicPanel is ProcessingControlPanel

    quick = ProcessingControlPanel("browse_quick")
    full = ProcessingControlPanel("viewer_full")

    assert quick.state() == {"align_rows": None, "remove_bad_lines": None}
    assert full.state()["align_rows"] is None

    quick.close()
    full.close()


def test_terminal_widgets_import_from_new_module_and_gui_package(qapp):
    from probeflow.gui import DeveloperTerminalWidget as PublicTerminal
    from probeflow.gui import _TerminalPane as PublicPane
    from probeflow.gui.terminal import DeveloperTerminalWidget, _TerminalPane

    assert PublicTerminal is DeveloperTerminalWidget
    assert PublicPane is _TerminalPane

    pane = _TerminalPane()
    pane.set_prompt("test $ ")
    pane.show_prompt()
    assert pane.toPlainText().startswith("test $ ")

    widget = DeveloperTerminalWidget(_theme())
    assert widget.findChild(_TerminalPane) is not None

    pane.close()
    widget.close()
