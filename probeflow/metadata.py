"""Lightweight metadata for Createc DAT and Nanonis SXM image scan files.

Public API
----------
read_scan_metadata(path) -> ScanMetadata
    Return metadata for a supported image scan file without exposing the
    internal Scan representation to callers that only need header info.

metadata_from_scan(scan) -> ScanMetadata
    Build metadata from an already-loaded Scan object.

ScanMetadata
    Frozen dataclass holding the stable, format-agnostic summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from probeflow.common import _f


# ── ScanMetadata dataclass ────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScanMetadata:
    """Lightweight, immutable summary of a single STM image scan.

    source_format is "createc_dat" or "nanonis_sxm" (independent of the
    internal Scan.source_format strings "dat" / "sxm").
    """

    path: Path
    source_format: str                          # "createc_dat" | "nanonis_sxm"
    item_type: str = "scan"
    display_name: str = ""
    shape: Optional[tuple[int, int]] = None     # (Ny, Nx)
    plane_names: tuple[str, ...] = ()
    units: tuple[str, ...] = ()                 # parallel to plane_names
    scan_range: Optional[tuple[float, float]] = None  # (width_m, height_m)
    bias: Optional[float] = None                # V
    setpoint: Optional[float] = None            # A (tunnel current setpoint)
    comment: Optional[str] = None
    acquisition_datetime: Optional[str] = None
    raw_header: dict[str, Any] = field(default_factory=dict)


# ── Format string mapping ────────────────────────────────────────────────────

_FORMAT_MAP = {"dat": "createc_dat", "sxm": "nanonis_sxm"}


# ── metadata_from_scan ───────────────────────────────────────────────────────

def metadata_from_scan(scan) -> ScanMetadata:
    """Build a :class:`ScanMetadata` from an already-loaded ``Scan``."""
    source_format = _FORMAT_MAP.get(scan.source_format, scan.source_format)

    shape = scan.planes[0].shape if scan.planes else None
    plane_names = tuple(scan.plane_names)
    units = tuple(scan.plane_units)
    scan_range = tuple(scan.scan_range_m) if scan.scan_range_m else None
    hdr = dict(scan.header)

    display_name = Path(scan.source_path).stem if scan.source_path else ""

    if scan.source_format == "dat":
        bias, setpoint, comment, acq_dt = _extract_createc_fields(hdr)
    elif scan.source_format == "sxm":
        bias, setpoint, comment, acq_dt = _extract_nanonis_fields(hdr)
    else:
        bias, setpoint, comment, acq_dt = None, None, None, None

    return ScanMetadata(
        path=Path(scan.source_path),
        source_format=source_format,
        item_type="scan",
        display_name=display_name,
        shape=shape,
        plane_names=plane_names,
        units=units,
        scan_range=scan_range,
        bias=bias,
        setpoint=setpoint,
        comment=comment,
        acquisition_datetime=acq_dt,
        raw_header=hdr,
    )


def _extract_createc_fields(hdr: dict) -> tuple:
    """Extract bias, setpoint, comment, datetime from a Createc header."""
    # Bias: "BiasVolt.[mV]" or "Biasvolt[mV]" (mV → V)
    bias_mv = _f(hdr.get("BiasVolt.[mV]") or hdr.get("Biasvolt[mV]"))
    bias = bias_mv / 1000.0 if bias_mv is not None else None

    # Setpoint current: "Current[A]"
    setpoint = _f(hdr.get("Current[A]"))

    # Comment / title: "Titel" (German for title)
    raw_titel = hdr.get("Titel", "")
    comment = str(raw_titel).strip() or None

    # Date/time: "PSTMAFM.EXE_Date"
    acq_dt = str(hdr.get("PSTMAFM.EXE_Date", "")).strip() or None

    return bias, setpoint, comment, acq_dt


def _extract_nanonis_fields(hdr: dict) -> tuple:
    """Extract bias, setpoint, comment, datetime from a Nanonis SXM header."""
    # Bias: prefer "Bias>Bias (V)", fall back to "BIAS"
    bias = _f(hdr.get("Bias>Bias (V)") or hdr.get("BIAS"))

    # Setpoint current: "Current>Current (A)"
    setpoint = _f(hdr.get("Current>Current (A)"))

    # Comment
    raw_comment = hdr.get("COMMENT", "")
    comment = str(raw_comment).strip() or None

    # Date + time
    date = str(hdr.get("REC_DATE", "")).strip()
    time = str(hdr.get("REC_TIME", "")).strip()
    if date:
        acq_dt = f"{date} {time}".strip()
    else:
        acq_dt = None

    return bias, setpoint, comment, acq_dt


# ── read_scan_metadata ───────────────────────────────────────────────────────

def read_scan_metadata(path) -> ScanMetadata:
    """Return :class:`ScanMetadata` for a Createc DAT or Nanonis SXM image file.

    Spectroscopy files and unknown file types raise ``ValueError`` with a
    descriptive message.  Currently delegates to the full image loader
    internally, but the public API is designed so header-only parsing can be
    substituted later without changing callers.
    """
    from probeflow.file_type import FileType, sniff_file_type
    from probeflow.scan import load_scan

    p = Path(path)
    ft = sniff_file_type(p)

    if ft == FileType.NANONIS_IMAGE:
        return metadata_from_scan(load_scan(p))
    if ft == FileType.CREATEC_IMAGE:
        return metadata_from_scan(load_scan(p))
    if ft == FileType.NANONIS_SPEC:
        raise ValueError(
            f"{p.name}: this is a Nanonis spectroscopy file — "
            "read_scan_metadata only supports image scans"
        )
    if ft == FileType.CREATEC_SPEC:
        raise ValueError(
            f"{p.name}: this is a Createc .VERT spectroscopy file — "
            "read_scan_metadata only supports image scans"
        )
    raise ValueError(
        f"Unsupported or unrecognised file for metadata: {p}"
    )
