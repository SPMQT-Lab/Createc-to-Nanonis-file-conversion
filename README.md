<p align="center">
  <img src="assets/logo.gif" alt="ProbeFlow logo" width="100%"/>
</p>

<h1 align="center">ProbeFlow</h1>
<p align="center"><em>A scanning-tunneling-microscopy toolkit — browse, process, convert.</em></p>

---

## What It Is

**ProbeFlow** is a scanning-tunneling-microscopy (STM) toolkit built around three everyday lab needs:

1. **Getting scans out of Createc and into Nanonis.**
   Createc `.dat` files are decoded, scaled to SI units, and repackaged as Nanonis-compatible `.sxm` files or PNG previews.
2. **Looking at and cleaning up the scans.**
   Browse a folder of `.sxm` files in a thumbnail grid, switch colormaps, flatten with plane / facet levelling, fix bad scan lines, smooth, FFT-filter, detect grains, measure lattice periodicities, and export publication-ready PNGs with scale bars — **from a GUI *or* from the shell.**
3. **Reading and processing point-measurement spectroscopy.**
   Createc `.VERT` files (bias sweeps, time traces, Z spectroscopy) are read into a unit-aware data model, smoothed, differentiated, overlaid, and plotted — with tip positions optionally shown as markers on a topography image.

Everything is available as a CLI subcommand, so corrections and conversions can be scripted across hundreds of files or wired into a processing pipeline.

---

## Installation

```bash
git clone https://github.com/SPMQT-Lab/ProbeFlow.git
cd ProbeFlow
python -m pip install -e .
```

Python 3.11+ is required. The install pulls in `numpy`, `scipy`, `pillow`, `matplotlib`, and `PySide6`.

---

## Quick Start

### Launch the GUI

```bash
probeflow gui
```

### One-shot from the shell

```bash
# Convert a folder of Createc .dat scans into .sxm
probeflow dat2sxm -- --input-dir data/scans --output-dir out/sxm

# Export a single .sxm plane as a colour PNG with an auto-sized nm scale bar
probeflow sxm2png scan.sxm --colormap inferno --clip-low 2 --clip-high 98

# Flatten with a linear plane background, write a new .sxm
probeflow plane-bg scan.sxm --order 1 -o scan_flat.sxm

# Chain three corrections and write a PNG straight out
probeflow pipeline scan.sxm \
    --steps align-rows:median plane-bg:1 smooth:1.5 \
    --png --colormap viridis -o scan_clean.png
```

---

## Commands

The top-level command is `probeflow`. Every subcommand accepts `--help`.

### Conversion

| Command   | Purpose                                                            |
|-----------|--------------------------------------------------------------------|
| `dat2sxm` | Createc `.dat` → Nanonis `.sxm` (use `--` to pass through flags)   |
| `dat2png` | Createc `.dat` → preview PNGs                                      |
| `sxm2png` | Nanonis `.sxm` → colorised PNG with optional scale bar             |

Legacy shortcuts `dat-sxm` and `dat-png` remain available for backward compatibility.

### Processing (`.sxm` in → `.sxm` or `.png` out)

Each of these reads an `.sxm`, applies a single operation to the selected plane (0 = Z forward by default), and writes a new `.sxm` — or a PNG with `--png`.

| Command            | Operation                                                         |
|--------------------|-------------------------------------------------------------------|
| `plane-bg`         | Subtract polynomial plane background (`--order 1` or `2`)         |
| `align-rows`       | Per-row offset / slope removal (`--method median|mean|linear`)    |
| `remove-bad-lines` | MAD-based outlier-row interpolation                               |
| `facet-level`      | Plane fit using only flat-terrace pixels — good for stepped surfaces |
| `smooth`           | Isotropic Gaussian smoothing (`--sigma` in pixels)                |
| `edge`             | Laplacian / LoG / DoG edge detection                              |
| `fft`              | 2-D FFT low-pass or high-pass filter                              |

Common options across the processing commands:

```
--plane N              # 0=Z-fwd, 1=Z-bwd, 2=I-fwd, 3=I-bwd  (default 0)
--png                  # write a colorised PNG instead of a new .sxm
--colormap NAME        # any matplotlib colormap name
--clip-low  P          # lower percentile for PNG contrast (default 1.0)
--clip-high P          # upper percentile for PNG contrast (default 99.0)
--no-scalebar
--scalebar-unit nm|Å|pm
--scalebar-pos  bottom-right|bottom-left
```

### Analysis / inspection

| Command       | Purpose                                                               |
|---------------|-----------------------------------------------------------------------|
| `grains`      | Detect islands or depressions and print per-grain area / centroid     |
| `autoclip`    | Suggest GMM-based clip percentiles for display                        |
| `periodicity` | Find dominant spatial periods via the power spectrum                  |
| `info`        | Print header metadata (`--json` for machine-readable output)          |

### Chain several steps: `pipeline`

`pipeline` runs any ordered sequence of processing atoms in a single invocation:

```bash
probeflow pipeline scan.sxm \
    --steps remove-bad-lines align-rows:median plane-bg:1 smooth:1.2 \
    -o scan_processed.sxm
```

Step syntax is `name[:param1,param2,…]`:

| Step               | Parameters                                      | Example                  |
|--------------------|-------------------------------------------------|--------------------------|
| `remove-bad-lines` | `mad_threshold` (default `5.0`)                 | `remove-bad-lines:4.0`   |
| `align-rows`       | `median` / `mean` / `linear`                    | `align-rows:linear`      |
| `plane-bg`         | `order` (`1` or `2`)                            | `plane-bg:2`             |
| `facet-level`      | `threshold_deg`                                 | `facet-level:2.0`        |
| `smooth`           | `sigma_px`                                      | `smooth:1.5`             |
| `edge`             | `method,sigma,sigma2`                           | `edge:log,1.0`           |
| `fft`              | `mode,cutoff,window`                            | `fft:low_pass,0.08`      |

Add `--png` to the `pipeline` command to skip `.sxm` output and write a colorised PNG directly.

### Spectroscopy (Createc `.VERT` files)

ProbeFlow reads Createc vertical-spectroscopy files and auto-detects the sweep type from the data:

| Sweep type   | X-axis         | Typical use                                    |
|--------------|----------------|------------------------------------------------|
| Bias sweep   | Bias (V)       | I(V) / Z(V) tunnelling spectroscopy            |
| Time trace   | Time (s)       | I(t) / Z(t) at fixed bias — telegraph noise    |

```bash
# Print header metadata from a .VERT file
probeflow spec-info spectrum.VERT

# Quick plot of the Z channel vs. bias or time
probeflow spec-plot spectrum.VERT --channel Z -o spectrum.png

# Overlay multiple spectra with a waterfall offset; also show the mean
probeflow spec-overlay *.VERT --channel Z --offset 1e-10 --average -o stack.png

# Mark tip positions of a set of spectra on a topography image
probeflow spec-positions scan.sxm *.VERT -o positions.png
```

Available channels per file: `I` (current, A), `Z` (tip-sample distance, m), `V` (bias, V).

**Programmatic API:**

```python
from probeflow.spec_io import read_spec_file
from probeflow.spec_processing import smooth_spectrum, numeric_derivative, crop
from probeflow.spec_plot import plot_spectrum, plot_spectra

spec = read_spec_file("spectrum.VERT")
print(spec.metadata["sweep_type"])  # "bias_sweep" or "time_trace"
print(spec.position)                # (x_m, y_m) tip position in metres

# Smooth the Z channel and compute dZ/dV
z_smooth = smooth_spectrum(spec.channels["Z"], method="savgol", window_length=21)
dzdv = numeric_derivative(spec.x_array, z_smooth)

# Crop to a sub-range and plot
x_crop, z_crop = crop(spec.x_array, z_smooth, x_min=-0.3, x_max=-0.05)

# Overlay multiple spectra
specs = [read_spec_file(p) for p in sorted(Path(".").glob("*.VERT"))]
ax = plot_spectra(specs, channel="Z", offset=5e-10)
```

### GUI

```bash
probeflow gui
```

Opens the browser view with thumbnail grid, colormap gallery, live clip sliders, per-scan undo, an image viewer with interactive histogram, processing panel, and PNG export dialog. Preferences (folders, theme, clip values) are saved to `~/.probeflow_config.json`.

---

## Bash-driven workflows

### Batch-process a folder

```bash
for f in data/sxm/*.sxm; do
    probeflow pipeline "$f" \
        --steps align-rows:median plane-bg:1 \
        -o "processed/${f##*/}"
done
```

### Convert → flatten → export for publication

```bash
probeflow dat2sxm -- --input-dir raw --output-dir sxm
for s in sxm/*.sxm; do
    probeflow pipeline "$s" \
        --steps remove-bad-lines align-rows:median plane-bg:1 smooth:1.0 \
        --png --colormap inferno --scalebar-unit nm \
        -o "figures/${s##*/}.png"
done
```

### Auto-suggest contrast across a dataset (JSON out)

```bash
for s in sxm/*.sxm; do
    echo -n "$s  "
    probeflow autoclip "$s" --json
done
```

### Machine-readable lattice periods

```bash
probeflow periodicity scan.sxm --n-peaks 3 --json \
    | jq '.[] | {period_nm: (.period_m*1e9), angle_deg}'
```

---

## Programmatic use

The package is importable without pulling in the GUI:

```python
# Topographic image processing
from probeflow import processing
from probeflow.sxm_io import (
    read_sxm_plane, write_sxm_with_planes, read_all_sxm_planes,
    parse_sxm_header, sxm_dims, sxm_scan_range,
)

arr = read_sxm_plane("scan.sxm", plane_idx=0)
arr = processing.align_rows(arr, method="median")
arr = processing.subtract_background(arr, order=1)

# Spectroscopy
from probeflow.spec_io import read_spec_file
from probeflow.spec_processing import smooth_spectrum, numeric_derivative

spec = read_spec_file("spectrum.VERT")
z_smooth = smooth_spectrum(spec.channels["Z"], method="savgol")
dzdv = numeric_derivative(spec.x_array, z_smooth)
```

---

## Repository layout

```
probeflow/              # installable package
├── __init__.py
├── common.py           # DAC / header utilities used by both conversion paths
├── dat_sxm.py          # Createc .dat → Nanonis .sxm
├── dat_png.py          # Createc .dat → PNG previews
├── sxm_io.py           # .sxm read / write (GUI-free)
├── processing.py       # image-processing pipeline (GUI-free)
├── spec_io.py          # Createc .VERT reader → SpecData (GUI-free)
├── spec_processing.py  # spectroscopy processing functions (GUI-free)
├── spec_plot.py        # spectroscopy matplotlib plots (GUI-free)
├── gui.py              # PySide6 desktop interface
└── cli.py              # unified "probeflow" command

src/file_cushions/      # binary layout captured from a reference .sxm file
data/                   # sample input / output for manual runs + tests
tests/                  # pytest suite (conversion, processing, .sxm round-trip,
                        #               spectroscopy reader + processing)
assets/                 # logo artwork
```

The `src/file_cushions/` directory holds the byte-level layout used to reconstruct `.sxm` files (header padding, `:SCANIT_END:` marker position, tail bytes, fixed data offset). These were reverse-engineered once from a reference Nanonis file and should be regenerated only if a future Nanonis version shifts the binary layout.

---

## Tests

```bash
pip install -e '.[dev]'
pytest
```

Covers:

* Conversion (`.dat` → `.sxm` and `.dat` → PNG) against the bundled sample scans.
* All ten functions in `probeflow.processing`.
* `.sxm` header parsing, plane reading, and write-then-read round-trip.
* `.VERT` header parsing, unit conversion, sweep-type detection, and error handling.
* All six functions in `probeflow.spec_processing`.

---

## Notes

* The `.sxm` timestamp parser expects Createc filenames of the form `AyyMMdd.HHmmss.dat`.
* Failed batch conversions are logged to `errors.json` in the output directory.
* Sine-mode Createc scans are currently rejected with a clear error (not silently corrupted).

---

## Acknowledgements

**ProbeFlow** is developed at **[SPMQT-Lab](https://github.com/SPMQT-Lab)**, under the supervision of **Dr. Peter Jacobson** at **The University of Queensland**.

The core Createc-decoding algorithms were originally written by **[Rohan Platts](https://github.com/rohanplatts)**. ProbeFlow is a refactored, extended, and GUI-enabled evolution of that work — his contributions are the foundation of the conversion pipeline.

> *"Standing on the shoulders of giants."*
