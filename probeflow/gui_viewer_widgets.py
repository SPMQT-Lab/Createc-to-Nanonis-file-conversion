"""Viewer-only widgets used by ImageViewerDialog."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QPointF, QRect, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF, QWheelEvent,
)
from PySide6.QtWidgets import QLabel, QToolTip, QWidget

# ── Physical-axis ruler (top / left of the image) ───────────────────────────
class RulerWidget(QWidget):
    """Thin tick-mark ruler showing physical nm extent of the image.

    Placed in the scroll viewport alongside the image so it scrolls/zooms
    in step with it. ``orientation`` is "horizontal" (top, runs left→right)
    or "vertical" (left, runs top→bottom).
    """

    THICKNESS_PX = 26  # height for horizontal, width for vertical

    def __init__(self, orientation: str = "horizontal", parent=None):
        super().__init__(parent)
        if orientation not in ("horizontal", "vertical"):
            raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation!r}")
        self._orient = orientation
        self._scan_nm: float = 0.0
        self._extent_px: int = 0  # image pixel extent in this direction
        if orientation == "horizontal":
            self.setFixedHeight(self.THICKNESS_PX)
        else:
            self.setFixedWidth(self.THICKNESS_PX)

    def set_extent(self, scan_nm: float, extent_px: int) -> None:
        """Bind to scan physical size and current pixmap extent (px)."""
        self._scan_nm = float(scan_nm) if scan_nm and scan_nm > 0 else 0.0
        self._extent_px = max(0, int(extent_px))
        if self._orient == "horizontal":
            self.setFixedWidth(max(1, self._extent_px))
        else:
            self.setFixedHeight(max(1, self._extent_px))
        self.update()

    @staticmethod
    def _nice_step(scan_nm: float) -> float:
        """Pick a tick step roughly scan/5, snapped to {1, 2, 5} × 10^k."""
        if scan_nm <= 0:
            return 1.0
        target = scan_nm / 5.0
        import math
        exp = math.floor(math.log10(target))
        base = target / (10 ** exp)
        if base < 1.5:
            mult = 1
        elif base < 3.5:
            mult = 2
        elif base < 7.5:
            mult = 5
        else:
            mult = 10
        return mult * (10 ** exp)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._scan_nm <= 0 or self._extent_px <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#cdd6f4"), 1))
        painter.setFont(QFont("Helvetica", 7))

        step = self._nice_step(self._scan_nm)
        if step <= 0:
            painter.end()
            return
        n_steps = int(self._scan_nm / step) + 1

        if self._orient == "horizontal":
            # Border line at bottom of the ruler.
            y_baseline = self.height() - 1
            painter.drawLine(0, y_baseline, self.width(), y_baseline)
            for i in range(n_steps + 1):
                nm = i * step
                if nm > self._scan_nm + 1e-9:
                    break
                x = int(round(nm / self._scan_nm * self._extent_px))
                tick_h = 6 if i % 1 == 0 else 3
                painter.drawLine(x, y_baseline - tick_h, x, y_baseline)
                lbl = f"{nm:g}"
                fm = painter.fontMetrics()
                w = fm.horizontalAdvance(lbl)
                tx = max(0, min(self.width() - w, x - w // 2))
                painter.drawText(tx, y_baseline - tick_h - 2, lbl)
            # Unit label at far right, drawn against the right edge.
            unit = "nm"
            fm = painter.fontMetrics()
            uw = fm.horizontalAdvance(unit)
            painter.drawText(self.width() - uw - 2, y_baseline - 2, unit)
        else:  # vertical
            x_baseline = self.width() - 1
            painter.drawLine(x_baseline, 0, x_baseline, self.height())
            for i in range(n_steps + 1):
                nm = i * step
                if nm > self._scan_nm + 1e-9:
                    break
                y = int(round(nm / self._scan_nm * self._extent_px))
                tick_w = 6 if i % 1 == 0 else 3
                painter.drawLine(x_baseline - tick_w, y, x_baseline, y)
                lbl = f"{nm:g}"
                fm = painter.fontMetrics()
                h = fm.height()
                # Right-align label to the tick start.
                w = fm.horizontalAdvance(lbl)
                tx = max(0, x_baseline - tick_w - 2 - w)
                ty = max(h, min(self.height(), y + h // 2 - 1))
                painter.drawText(tx, ty, lbl)
        painter.end()


# ── Scale-bar widget (lives below the image, separate from the pixmap) ──────
class ScaleBarWidget(QWidget):
    """Independent scale bar drawn underneath the image (not on the pixmap).

    Defaults to an integer-nm length computed as roughly 20-30% of the scan
    width, rounded down to the nearest integer nm. The user can override that
    with a custom value (which can be any positive integer nm). Hidden when
    ``visible`` is False; nothing is painted at all in that case.
    """

    BAR_HEIGHT_PX = 6
    LABEL_GAP_PX  = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scan_nm: float = 0.0
        self._image_pixel_width: int = 0
        self._visible: bool = False
        self._custom_nm: Optional[float] = None  # None → auto integer length
        self.setFixedHeight(28)
        self.setMinimumWidth(40)

    def set_scan_size(self, scan_nm: float, image_pixel_width: int) -> None:
        """Tell the widget the scan's physical width and the on-screen width
        of the image pixmap (so the bar can be sized in proportion).
        """
        self._scan_nm = float(scan_nm) if scan_nm and scan_nm > 0 else 0.0
        self._image_pixel_width = max(0, int(image_pixel_width))
        self.update()

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        self.update()

    def set_custom_length_nm(self, length_nm: Optional[float]) -> None:
        """Override the auto length. Pass None to revert to auto."""
        if length_nm is None or length_nm <= 0:
            self._custom_nm = None
        else:
            self._custom_nm = float(length_nm)
        self.update()

    def auto_length_nm(self) -> float:
        """Default integer-nm bar length (~25% of scan), floored to integer."""
        if self._scan_nm <= 0:
            return 0.0
        target = self._scan_nm * 0.25
        if target >= 1.0:
            return float(int(target))  # floor to integer nm
        # Sub-nm: fall back to a single nm if scan_nm >= 1, else just half.
        if self._scan_nm >= 1.0:
            return 1.0
        return max(0.1, round(self._scan_nm * 0.25, 2))

    def current_length_nm(self) -> float:
        if self._custom_nm is not None:
            return self._custom_nm
        return self.auto_length_nm()

    def paintEvent(self, event):
        super().paintEvent(event)
        if (not self._visible) or self._scan_nm <= 0 or self._image_pixel_width <= 0:
            return
        length_nm = self.current_length_nm()
        if length_nm <= 0 or length_nm > self._scan_nm:
            return
        bar_px = int(round(length_nm / self._scan_nm * self._image_pixel_width))
        if bar_px <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Centre the bar horizontally over the image (image is centered in
        # the scroll area by Qt.AlignCenter; we approximate by centering on
        # this widget's full width — close enough since the widget is set
        # to match the scroll area width).
        x0 = (self.width() - bar_px) // 2
        y0 = 4
        painter.setPen(QPen(QColor("white"), 0))
        painter.setBrush(QBrush(QColor("black")))
        painter.drawRect(x0, y0, bar_px, self.BAR_HEIGHT_PX)

        # Label "X nm" centred below.
        if length_nm == int(length_nm):
            txt = f"{int(length_nm)} nm"
        else:
            txt = f"{length_nm:g} nm"
        painter.setPen(QPen(QColor("black"), 0))
        painter.setFont(QFont("Helvetica", 10, QFont.Bold))
        painter.drawText(QRect(x0, y0 + self.BAR_HEIGHT_PX + self.LABEL_GAP_PX,
                                bar_px, 16),
                         Qt.AlignCenter, txt)
        painter.end()


# ── Full-size image viewer dialog ─────────────────────────────────────────────
class _ZoomLabel(QLabel):
    """QLabel inside a scroll area that supports Ctrl+Wheel zoom and spec-position markers."""

    marker_clicked = Signal(object)  # emits VertFile when user clicks a marker
    pixel_clicked  = Signal(float, float)  # (frac_x, frac_y) — only when set_zero_mode is on
    selection_changed = Signal(object)  # structured ROI geometry
    pixmap_resized = Signal(int)  # new pixmap width in pixels (zoom changes, source changes)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap_orig: Optional[QPixmap] = None
        self._zoom = 1.0
        self._markers: list[dict] = []   # each: {frac_x, frac_y, entry}
        self._show_markers: bool = True
        self._zero_markers: list[dict] = []  # each: {frac_x, frac_y, label}
        self._set_zero_mode: bool = False
        self._selection_tool: str = "none"
        self._selection_start = None
        self._selection_drag = None
        self._selection_geometry = None
        self._polygon_points: list[tuple[float, float]] = []
        self.setMouseTracking(True)

    def set_set_zero_mode(self, enabled: bool):
        self._set_zero_mode = bool(enabled)
        self._update_cursor()

    def set_selection_tool(self, kind: str):
        kind = str(kind or "none").lower()
        if kind not in {"none", "rectangle", "ellipse", "polygon", "line"}:
            kind = "none"
        self._selection_tool = kind
        self._selection_start = None
        self._selection_drag = None
        self._polygon_points = []
        self.update()
        self._update_cursor()

    def clear_roi(self):
        self._selection_start = None
        self._selection_drag = None
        self._selection_geometry = None
        self._polygon_points = []
        self.update()

    def _update_cursor(self):
        self.setCursor(
            Qt.CrossCursor if (
                self._set_zero_mode or self._selection_tool != "none"
            )
            else Qt.ArrowCursor
        )

    def set_source(self, pixmap: QPixmap, reset_zoom: bool = True):
        self._pixmap_orig = pixmap
        if reset_zoom:
            self._zoom = 1.0
        self._apply_zoom()

    def zoom_by(self, factor: float):
        self._zoom = max(0.25, min(8.0, self._zoom * factor))
        self._apply_zoom()

    def reset_zoom(self):
        self._zoom = 1.0
        self._apply_zoom()

    def zoom(self) -> float:
        return self._zoom

    def set_markers(self, markers: list[dict]):
        self._markers = markers
        self.update()

    def set_show_markers(self, visible: bool):
        self._show_markers = visible
        self.update()

    def set_zero_markers(self, markers: list[dict]):
        """Mark click locations used for set-zero / set-zero-plane interaction.

        Each marker is a dict with ``frac_x``, ``frac_y`` in [0,1] and a
        ``label`` string drawn inside the dot.
        """
        self._zero_markers = list(markers or [])
        self.update()

    def _apply_zoom(self):
        if self._pixmap_orig is None:
            return
        w = int(self._pixmap_orig.width()  * self._zoom)
        h = int(self._pixmap_orig.height() * self._zoom)
        # SPM image pixels are measured samples, not presentation pixels.
        # Preserve the raster honestly: 1:1 is native data size, and zoom uses
        # nearest-neighbour style scaling instead of smoothing/interpolation.
        scaled = self._pixmap_orig.scaled(w, h, Qt.KeepAspectRatio,
                                           Qt.FastTransformation)
        self.setPixmap(scaled)
        # The label sits inside a ruler QGridLayout. A plain resize() can be
        # overridden by the layout, leaving a large pixmap squeezed into a tiny
        # label. Fixed size is what we want in this scroll-area context: zoom
        # changes should alter the scrollable image extent, not let the layout
        # compress the image to fit beside the rulers.
        self.setFixedSize(scaled.size())
        self.update()
        self.pixmap_resized.emit(scaled.width())

    def _marker_px(self, frac_x: float, frac_y: float) -> tuple[int, int]:
        """Fractional image coords → label pixel coords."""
        return int(frac_x * self.width()), int(frac_y * self.height())

    def _norm_bounds(self, x0, y0, x1, y1) -> tuple[float, float, float, float]:
        return (
            max(0.0, min(1.0, min(float(x0), float(x1)))),
            max(0.0, min(1.0, min(float(y0), float(y1)))),
            max(0.0, min(1.0, max(float(x0), float(x1)))),
            max(0.0, min(1.0, max(float(y0), float(y1)))),
        )

    def _frac_from_pos(self, pos) -> tuple[float, float]:
        return (
            max(0.0, min(1.0, pos.x() / float(max(1, self.width())))),
            max(0.0, min(1.0, pos.y() / float(max(1, self.height())))),
        )

    def _rect_from_bounds(self, bounds) -> QRect:
        x0, y0, x1, y1 = bounds
        return QRect(
            int(round(x0 * self.width())),
            int(round(y0 * self.height())),
            int(round((x1 - x0) * self.width())),
            int(round((y1 - y0) * self.height())),
        ).normalized()

    def _active_selection(self):
        if self._selection_drag is not None:
            return self._selection_drag
        return self._selection_geometry

    def _selection_points_px(self, points):
        return [
            QPointF(float(x) * self.width(), float(y) * self.height())
            for x, y in points
        ]

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap_orig is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._show_markers and self._markers:
            r = 7
            for i, m in enumerate(self._markers):
                sx, sy = self._marker_px(m["frac_x"], m["frac_y"])
                painter.setBrush(QBrush(QColor("#FFD700")))
                painter.setPen(QPen(QColor("black"), 1.5))
                painter.drawEllipse(sx - r, sy - r, 2 * r, 2 * r)
                painter.setFont(QFont("Helvetica", 6, QFont.Bold))
                painter.setPen(QPen(QColor("black")))
                painter.drawText(QRect(sx - r, sy - r, 2 * r, 2 * r),
                                 Qt.AlignCenter, str(i + 1))
        if self._zero_markers:
            r = 8
            for m in self._zero_markers:
                sx, sy = self._marker_px(m["frac_x"], m["frac_y"])
                # Cross hairs for precision, then a filled dot on top.
                painter.setPen(QPen(QColor(0, 0, 0, 200), 1))
                painter.drawLine(sx - r - 4, sy, sx - r, sy)
                painter.drawLine(sx + r, sy, sx + r + 4, sy)
                painter.drawLine(sx, sy - r - 4, sx, sy - r)
                painter.drawLine(sx, sy + r, sx, sy + r + 4)
                painter.setBrush(QBrush(QColor("#22D3EE")))  # cyan
                painter.setPen(QPen(QColor("black"), 1.5))
                painter.drawEllipse(sx - r, sy - r, 2 * r, 2 * r)
                label = str(m.get("label", ""))
                if label:
                    painter.setFont(QFont("Helvetica", 6, QFont.Bold))
                    painter.setPen(QPen(QColor("black")))
                    painter.drawText(QRect(sx - r, sy - r, 2 * r, 2 * r),
                                     Qt.AlignCenter, label)
        geometry = self._active_selection()
        if geometry is not None:
            painter.setBrush(QBrush(QColor(137, 180, 250, 45)))
            painter.setPen(QPen(QColor("#89b4fa"), 2))
            kind = geometry.get("kind")
            if kind == "ellipse" and geometry.get("bounds_frac"):
                rect = self._rect_from_bounds(geometry["bounds_frac"])
                painter.drawEllipse(rect)
                self._draw_selection_handles(painter, [
                    (rect.left(), rect.top()), (rect.right(), rect.top()),
                    (rect.right(), rect.bottom()), (rect.left(), rect.bottom()),
                ])
            elif kind == "polygon" and geometry.get("points_frac"):
                pts = QPolygonF(self._selection_points_px(geometry["points_frac"]))
                if len(pts) >= 2:
                    painter.drawPolyline(pts)
                if len(pts) >= 3 and geometry is self._selection_geometry:
                    painter.drawPolygon(pts)
                self._draw_selection_handles(painter, [(p.x(), p.y()) for p in pts])
            elif kind == "line" and geometry.get("points_frac"):
                pts = self._selection_points_px(geometry["points_frac"])
                if len(pts) >= 2:
                    painter.drawLine(pts[0], pts[-1])
                    self._draw_selection_handles(
                        painter,
                        [(pts[0].x(), pts[0].y()), (pts[-1].x(), pts[-1].y())],
                    )
            elif geometry.get("bounds_frac"):
                rect = self._rect_from_bounds(geometry["bounds_frac"])
                painter.drawRect(rect)
                self._draw_selection_handles(painter, [
                    (rect.left(), rect.top()), (rect.right(), rect.top()),
                    (rect.right(), rect.bottom()), (rect.left(), rect.bottom()),
                ])
        painter.end()

    def _draw_selection_handles(self, painter: QPainter, points) -> None:
        painter.save()
        painter.setBrush(QBrush(QColor("#89b4fa")))
        painter.setPen(QPen(QColor("#11111b"), 1))
        for x, y in points:
            painter.drawEllipse(QPointF(float(x), float(y)), 3.5, 3.5)
        painter.restore()

    def mouseMoveEvent(self, event):
        if (
            self._selection_tool in {"rectangle", "ellipse", "line"}
            and self._selection_start is not None
        ):
            fx0, fy0 = self._selection_start
            fx1, fy1 = self._frac_from_pos(event.pos())
            if self._selection_tool == "line":
                self._selection_drag = {
                    "kind": "line",
                    "points_frac": [(fx0, fy0), (fx1, fy1)],
                }
            else:
                self._selection_drag = {
                    "kind": self._selection_tool,
                    "bounds_frac": self._norm_bounds(fx0, fy0, fx1, fy1),
                }
            self.update()
            return
        if self._show_markers and self._markers and self._pixmap_orig is not None:
            pos = event.pos()
            for m in self._markers:
                sx, sy = self._marker_px(m["frac_x"], m["frac_y"])
                if abs(pos.x() - sx) <= 10 and abs(pos.y() - sy) <= 10:
                    entry = m["entry"]
                    lines = [entry.stem]
                    if getattr(entry, "measurement_label", None):
                        lines.append(entry.measurement_label)
                    if entry.sweep_type and entry.sweep_type != "unknown":
                        lines.append(entry.sweep_type)
                    if entry.bias_mv is not None:
                        lines.append(f"Bias: {entry.bias_mv:.0f} mV")
                    QToolTip.showText(event.globalPosition().toPoint(),
                                      "\n".join(lines), self)
                    return
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton and self._selection_tool == "polygon"
                and self._pixmap_orig is not None):
            self._polygon_points.append(self._frac_from_pos(event.pos()))
            self._selection_drag = {
                "kind": "polygon",
                "points_frac": list(self._polygon_points),
            }
            self.update()
            return
        if (event.button() == Qt.LeftButton
                and self._selection_tool in {"rectangle", "ellipse", "line"}
                and self._pixmap_orig is not None):
            self._selection_start = self._frac_from_pos(event.pos())
            self._selection_drag = None
            self.update()
            return
        if (event.button() == Qt.LeftButton and self._show_markers
                and self._markers and self._pixmap_orig is not None):
            pos = event.pos()
            for m in self._markers:
                sx, sy = self._marker_px(m["frac_x"], m["frac_y"])
                if abs(pos.x() - sx) <= 12 and abs(pos.y() - sy) <= 12:
                    self.marker_clicked.emit(m["entry"])
                    return
        if (event.button() == Qt.LeftButton and self._set_zero_mode
                and self._pixmap_orig is not None
                and self.width() > 0 and self.height() > 0):
            pos = event.pos()
            fx = max(0.0, min(1.0, pos.x() / float(self.width())))
            fy = max(0.0, min(1.0, pos.y() / float(self.height())))
            self.pixel_clicked.emit(fx, fy)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and self._selection_tool in {"rectangle", "ellipse", "line"}
            and self._selection_start is not None
            and self._pixmap_orig is not None
        ):
            fx0, fy0 = self._selection_start
            fx1, fy1 = self._frac_from_pos(event.pos())
            self._selection_start = None
            if self._selection_tool == "line":
                geometry = {
                    "kind": "line",
                    "points_frac": [(fx0, fy0), (fx1, fy1)],
                }
            else:
                bounds = self._norm_bounds(fx0, fy0, fx1, fy1)
                if (
                    abs(bounds[2] - bounds[0]) * self.width() < 3
                    or abs(bounds[3] - bounds[1]) * self.height() < 3
                ):
                    self._selection_drag = None
                    self.update()
                    return
                geometry = {
                    "kind": self._selection_tool,
                    "bounds_frac": bounds,
                }
            self._selection_drag = None
            self._selection_geometry = geometry
            self.selection_changed.emit(dict(geometry))
            self.update()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and self._selection_tool == "polygon"
            and len(self._polygon_points) >= 3
        ):
            self._selection_geometry = {
                "kind": "polygon",
                "points_frac": list(self._polygon_points),
            }
            self._selection_drag = None
            self._polygon_points = []
            self.selection_changed.emit(dict(self._selection_geometry))
            self.update()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            self.zoom_by(1.12 if delta > 0 else 1 / 1.12)
            event.accept()
        else:
            super().wheelEvent(event)
