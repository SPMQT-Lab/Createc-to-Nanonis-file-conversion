"""Tests for probeflow.spec_io — Createc .VERT file reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probeflow.spec_io import SpecData, parse_spec_header, read_spec_file

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VERT_TIME_TRACE = DATA_DIR / "A180201.152542.M0001.VERT"
VERT_BIAS_SWEEP = DATA_DIR / "A180201.151737.M0001.VERT"


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def time_trace_spec():
    return read_spec_file(VERT_TIME_TRACE)


@pytest.fixture(scope="module")
def bias_sweep_spec():
    return read_spec_file(VERT_BIAS_SWEEP)


# ─── parse_spec_header ───────────────────────────────────────────────────────

class TestParseSpecHeader:
    def test_returns_dict(self):
        hdr = parse_spec_header(VERT_TIME_TRACE)
        assert isinstance(hdr, dict)
        assert len(hdr) > 10

    def test_dac_type_present(self):
        hdr = parse_spec_header(VERT_TIME_TRACE)
        assert "DAC-Type" in hdr
        assert "20bit" in hdr["DAC-Type"]

    def test_dac_to_a_xy_present(self):
        hdr = parse_spec_header(VERT_TIME_TRACE)
        assert "Dacto[A]xy" in hdr
        val = float(hdr["Dacto[A]xy"])
        assert 0 < val < 1.0  # Å/DAC, physically reasonable

    def test_offset_xy_present(self):
        hdr = parse_spec_header(VERT_TIME_TRACE)
        assert "OffsetX" in hdr
        assert "OffsetY" in hdr

    def test_spec_freq_present(self):
        hdr = parse_spec_header(VERT_TIME_TRACE)
        assert "SpecFreq" in hdr
        assert float(hdr["SpecFreq"]) > 0

    def test_does_not_require_full_file_read(self):
        # parse_spec_header should succeed even on a file that is mostly data.
        hdr = parse_spec_header(VERT_BIAS_SWEEP)
        assert "DAC-Type" in hdr


# ─── read_spec_file — time trace ─────────────────────────────────────────────

class TestReadSpecFileTimeTrace:
    def test_returns_specdata(self, time_trace_spec):
        assert isinstance(time_trace_spec, SpecData)

    def test_sweep_type(self, time_trace_spec):
        assert time_trace_spec.metadata["sweep_type"] == "time_trace"

    def test_n_points(self, time_trace_spec):
        assert time_trace_spec.metadata["n_points"] == 5000

    def test_x_axis_is_time(self, time_trace_spec):
        assert time_trace_spec.x_unit == "s"
        assert "Time" in time_trace_spec.x_label

    def test_x_array_shape(self, time_trace_spec):
        assert time_trace_spec.x_array.shape == (5000,)

    def test_x_array_monotonic(self, time_trace_spec):
        assert np.all(np.diff(time_trace_spec.x_array) >= 0)

    def test_x_range_seconds(self, time_trace_spec):
        # SpecFreq=1000 Hz, 5000 pts → 0 to 4.999 s
        assert time_trace_spec.x_array[0] == pytest.approx(0.0)
        assert time_trace_spec.x_array[-1] == pytest.approx(4.999, rel=1e-3)

    def test_channels_present(self, time_trace_spec):
        for ch in ("I", "Z", "V"):
            assert ch in time_trace_spec.channels

    def test_channel_shapes(self, time_trace_spec):
        for arr in time_trace_spec.channels.values():
            assert arr.shape == (5000,)

    def test_z_channel_units_metres(self, time_trace_spec):
        z = time_trace_spec.channels["Z"]
        assert z.min() > -20e-10  # >-20 Å
        assert z.max() < 20e-10   # <+20 Å

    def test_position_is_tuple_of_floats(self, time_trace_spec):
        px, py = time_trace_spec.position
        assert isinstance(px, float)
        assert isinstance(py, float)

    def test_position_in_metres(self, time_trace_spec):
        px, py = time_trace_spec.position
        assert abs(px) < 1e-6
        assert abs(py) < 1e-6

    def test_y_units_dict(self, time_trace_spec):
        assert time_trace_spec.y_units["I"] == "A"
        assert time_trace_spec.y_units["Z"] == "m"

    def test_bias_constant(self, time_trace_spec):
        v = time_trace_spec.channels["V"]
        assert v.max() - v.min() < 1e-3  # < 1 mV variation


# ─── read_spec_file — bias sweep ─────────────────────────────────────────────

class TestReadSpecFileBiasSweep:
    def test_returns_specdata(self, bias_sweep_spec):
        assert isinstance(bias_sweep_spec, SpecData)

    def test_sweep_type(self, bias_sweep_spec):
        assert bias_sweep_spec.metadata["sweep_type"] == "bias_sweep"

    def test_x_axis_is_bias(self, bias_sweep_spec):
        assert bias_sweep_spec.x_unit == "V"
        assert "Bias" in bias_sweep_spec.x_label

    def test_x_array_shape(self, bias_sweep_spec):
        assert bias_sweep_spec.x_array.shape == (5000,)

    def test_x_range_volts(self, bias_sweep_spec):
        x = bias_sweep_spec.x_array
        assert x.min() == pytest.approx(-0.300, abs=0.01)
        assert x.max() == pytest.approx(-0.050, abs=0.01)

    def test_channels_present(self, bias_sweep_spec):
        for ch in ("I", "Z", "V"):
            assert ch in bias_sweep_spec.channels

    def test_z_varies(self, bias_sweep_spec):
        z = bias_sweep_spec.channels["Z"]
        assert float(z.max() - z.min()) > 0


# ─── error handling ──────────────────────────────────────────────────────────

class TestReadSpecFileErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_spec_file(tmp_path / "nonexistent.VERT")

    def test_missing_data_marker(self, tmp_path):
        bad = tmp_path / "bad.VERT"
        bad.write_bytes(b"key=val\r\nother=stuff\r\n")
        with pytest.raises(ValueError, match="DATA"):
            read_spec_file(bad)

    def test_too_few_columns(self, tmp_path):
        # 3 columns instead of required 4
        body = "DATA\r\n    3    0    0    1\r\n"
        body += "0\t-50.0\t0.0\r\n" * 3
        bad = tmp_path / "short.VERT"
        bad.write_bytes(body.encode())
        with pytest.raises(ValueError, match="4"):
            read_spec_file(bad)

    def test_threshold_kwarg_classifies_short_sweep(self, tmp_path):
        # A 0.5 mV sweep should be classified as a sweep if threshold is 0.1 mV.
        # Build a minimal header with Vpoint entries spanning 0.5 mV.
        hdr = (
            "[ParVERT30]\r\n"
            "DAC-Type=20bit\r\n"
            "GainPre 10^=9\r\n"
            "Dacto[A]xy=0.00083\r\n"
            "Dacto[A]z=0.00018\r\n"
            "OffsetX=0\r\nOffsetY=0\r\n"
            "SpecFreq=1000\r\n"
            "Vpoint0.t=0\r\nVpoint0.V=-50.0\r\n"
            "Vpoint1.t=100\r\nVpoint1.V=-50.5\r\n"  # 0.5 mV span
            "Vpoint2.t=0\r\nVpoint2.V=0\r\n"
        )
        body = hdr + "DATA\r\n    10    0    0    1\r\n"
        for i in range(10):
            v = -50.0 - i * 0.05
            body += f"{i}\t{v}\t0.0\t-5000.0\r\n"
        f = tmp_path / "short_sweep.VERT"
        f.write_bytes(body.encode())
        # Default threshold (1 mV) would classify this as time trace
        spec_default = read_spec_file(f)
        assert spec_default.metadata["sweep_type"] == "time_trace"
        # With tighter threshold (0.1 mV) it's a sweep
        spec_tight = read_spec_file(f, time_trace_threshold_mv=0.1)
        assert spec_tight.metadata["sweep_type"] == "bias_sweep"
