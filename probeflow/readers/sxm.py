"""Reader for Nanonis ``.sxm`` files — returns a :class:`probeflow.scan.Scan`."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from probeflow.scan import PLANE_CANON_NAMES, PLANE_CANON_UNITS, Scan
from probeflow.sxm_io import read_all_sxm_planes, sxm_scan_range


def read_sxm(path) -> Scan:
    """Load a Nanonis ``.sxm`` into a Scan (display-oriented, SI units)."""
    path = Path(path)
    hdr, planes = read_all_sxm_planes(path, orient=True)
    if not planes:
        raise ValueError(f"{path}: no data planes could be read")

    n = len(planes)
    names = list(PLANE_CANON_NAMES[:n])
    units = list(PLANE_CANON_UNITS[:n])
    synthetic = [False] * n

    # Pad names/units with placeholders if the file has an unusual plane count
    while len(names) < n:
        names.append(f"Channel {len(names)}")
        units.append("")

    scan_range_m = sxm_scan_range(hdr)

    return Scan(
        planes=[p.astype(np.float64) for p in planes],
        plane_names=names,
        plane_units=units,
        plane_synthetic=synthetic,
        header=dict(hdr),
        scan_range_m=scan_range_m,
        source_path=path,
        source_format="sxm",
    )


def read_sxm_metadata(path):
    """Return :class:`~probeflow.metadata.ScanMetadata` for a Nanonis ``.sxm``."""
    from probeflow.metadata import metadata_from_scan
    return metadata_from_scan(read_sxm(path))
