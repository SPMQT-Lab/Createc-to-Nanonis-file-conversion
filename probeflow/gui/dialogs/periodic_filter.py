from __future__ import annotations

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout,
)


class PeriodicFilterDialog(QDialog):
    """Interactive centred-FFT peak picker for periodic notch filtering."""

    def __init__(self, arr: np.ndarray, peaks=None, radius_px: float = 3.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Periodic FFT filter")
        self.resize(680, 620)
        self._arr = arr.astype(np.float64, copy=True)
        self._peaks: list[tuple[int, int]] = [
            (int(p[0]), int(p[1])) for p in (peaks or [])
        ]

        lay = QVBoxLayout(self)
        help_lbl = QLabel(
            "Click bright periodic peaks in the FFT power spectrum. "
            "Each click suppresses that peak and its conjugate in the processed image."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setFont(QFont("Helvetica", 9))
        lay.addWidget(help_lbl)

        self._fig = Figure(figsize=(6.0, 5.0), dpi=90)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111)
        lay.addWidget(self._canvas, 1)

        radius_row = QHBoxLayout()
        radius_lbl = QLabel("Notch radius:")
        radius_lbl.setFont(QFont("Helvetica", 8))
        self._radius_sl = QSlider(Qt.Horizontal)
        self._radius_sl.setRange(1, 20)
        self._radius_sl.setValue(max(1, min(20, int(round(radius_px)))))
        self._radius_val = QLabel(f"{self._radius_sl.value()} px")
        self._radius_val.setFont(QFont("Helvetica", 8))
        self._radius_sl.valueChanged.connect(
            lambda v: self._radius_val.setText(f"{v} px"))
        radius_row.addWidget(radius_lbl)
        radius_row.addWidget(self._radius_sl, 1)
        radius_row.addWidget(self._radius_val)
        lay.addLayout(radius_row)

        self._selected_lbl = QLabel("")
        self._selected_lbl.setWordWrap(True)
        self._selected_lbl.setFont(QFont("Helvetica", 8))
        lay.addWidget(self._selected_lbl)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear peaks")
        clear_btn.clicked.connect(self._clear)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Use selected peaks")
        apply_btn.setObjectName("accentBtn")
        apply_btn.clicked.connect(self.accept)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(apply_btn)
        lay.addLayout(btn_row)

        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._draw()

    def selected_peaks(self) -> list[tuple[int, int]]:
        return list(self._peaks)

    def radius_px(self) -> float:
        return float(self._radius_sl.value())

    def _spectrum(self) -> np.ndarray:
        a = self._arr
        nan_mask = ~np.isfinite(a)
        fill = float(np.nanmean(a)) if (~nan_mask).any() else 0.0
        centered = np.where(nan_mask, fill, a) - fill
        win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
        F = np.fft.fftshift(np.fft.fft2(centered * win))
        return np.log1p(np.abs(F) ** 2)

    def _draw(self):
        power = self._spectrum()
        Ny, Nx = power.shape
        cx, cy = Nx // 2, Ny // 2
        self._ax.clear()
        self._ax.imshow(power, cmap="magma", origin="upper")
        self._ax.set_title("FFT power spectrum")
        self._ax.set_xlabel("kx")
        self._ax.set_ylabel("ky")
        self._ax.axvline(cx, color="white", alpha=0.25, linewidth=0.8)
        self._ax.axhline(cy, color="white", alpha=0.25, linewidth=0.8)
        for dx, dy in self._peaks:
            for sx, sy in ((dx, dy), (-dx, -dy)):
                self._ax.plot(cx + sx, cy + sy, "o", color="#89b4fa",
                              markerfacecolor="none", markersize=9, markeredgewidth=1.8)
        self._fig.tight_layout()
        self._canvas.draw_idle()
        if self._peaks:
            text = ", ".join(f"({dx:+d}, {dy:+d})" for dx, dy in self._peaks)
            self._selected_lbl.setText(f"Selected peaks: {text}")
        else:
            self._selected_lbl.setText("Selected peaks: none")

    def _on_click(self, event):
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        Ny, Nx = self._arr.shape
        cx, cy = Nx // 2, Ny // 2
        dx = int(round(event.xdata)) - cx
        dy = int(round(event.ydata)) - cy
        if dx == 0 and dy == 0:
            return
        canonical = (dx, dy)
        conjugate = (-dx, -dy)
        if conjugate in self._peaks:
            canonical = conjugate
        if canonical in self._peaks:
            self._peaks.remove(canonical)
        else:
            self._peaks.append(canonical)
        self._draw()

    def _clear(self):
        self._peaks.clear()
        self._draw()
