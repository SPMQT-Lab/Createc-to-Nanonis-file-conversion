"""Tests for Phase 3: ROI-aware background subtraction, geometry algebra."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from probeflow.core.roi import ROI, ROISet, invert, combine
from probeflow.processing.image import subtract_background
from probeflow.processing.state import (
    ProcessingState,
    ProcessingStep,
    apply_processing_state,
    resolve_roi_expr,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rect(x, y, w, h, name="r"):
    return ROI.new("rectangle", {"x": float(x), "y": float(y),
                                  "width": float(w), "height": float(h)}, name=name)


def _poly(verts, name="p"):
    return ROI.new("polygon", {"vertices": [[float(v[0]), float(v[1])] for v in verts]},
                   name=name)


def _tilted_plane(Ny=64, Nx=64, *, slope_y=0.5, slope_x=0.3, intercept=1.0):
    """Pure linear background with no noise for deterministic tests."""
    ys = np.linspace(0, 1, Ny)
    xs = np.linspace(0, 1, Nx)
    X, Y = np.meshgrid(xs, ys)
    return slope_y * Y + slope_x * X + intercept


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Geometry algebra — invert()
# ═════════════════════════════════════════════════════════════════════════════

class TestInvert:
    def test_invert_rect_produces_roi(self):
        roi = _rect(10, 10, 20, 20)
        result = invert(roi, (64, 64))
        assert result is not None
        assert result.kind in ("polygon", "multipolygon")

    def test_invert_complement_mask(self):
        """Inverted ROI mask is the logical complement of the original."""
        roi = _rect(10, 10, 40, 30, name="inner")
        image_shape = (64, 64)
        inv = invert(roi, image_shape)
        original_mask = roi.to_mask(image_shape)
        inverted_mask = inv.to_mask(image_shape)
        # No pixel should be in both; total should cover whole image
        assert not np.any(original_mask & inverted_mask)
        assert np.all(original_mask | inverted_mask)

    def test_invert_name_prefix(self):
        roi = _rect(5, 5, 10, 10, name="terrace")
        inv = invert(roi, (32, 32))
        assert inv.name == "not_terrace"

    def test_invert_ellipse(self):
        roi = ROI.new("ellipse", {"cx": 32.0, "cy": 32.0, "rx": 10.0, "ry": 8.0},
                      name="blob")
        image_shape = (64, 64)
        inv = invert(roi, image_shape)
        original_mask = roi.to_mask(image_shape)
        inverted_mask = inv.to_mask(image_shape)
        # Ellipse uses a math formula; inverted mask uses a Shapely polygon
        # approximation (64 pts) — allow a small boundary mismatch
        overlap = original_mask & inverted_mask
        union = original_mask | inverted_mask
        assert overlap.sum() < 50  # boundary pixels only
        assert (union.sum() / (image_shape[0] * image_shape[1])) > 0.98

    def test_invert_polygon(self):
        roi = _poly([[10, 10], [30, 10], [30, 30], [10, 30]], name="quad")
        image_shape = (64, 64)
        inv = invert(roi, image_shape)
        assert inv.to_mask(image_shape).sum() < image_shape[0] * image_shape[1]

    def test_invert_line_raises(self):
        roi = ROI.new("line", {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0})
        with pytest.raises(ValueError, match="invert does not support"):
            invert(roi, (32, 32))

    def test_invert_point_raises(self):
        roi = ROI.new("point", {"x": 5.0, "y": 5.0})
        with pytest.raises(ValueError, match="invert does not support"):
            invert(roi, (32, 32))


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Geometry algebra — combine()
# ═════════════════════════════════════════════════════════════════════════════

class TestCombine:
    def test_union_larger_than_either(self):
        a = _rect(0, 0, 20, 20, name="a")
        b = _rect(10, 10, 20, 20, name="b")
        image_shape = (40, 40)
        u = combine([a, b], "union")
        assert u.to_mask(image_shape).sum() >= a.to_mask(image_shape).sum()
        assert u.to_mask(image_shape).sum() >= b.to_mask(image_shape).sum()

    def test_union_mask_equals_or_of_inputs(self):
        a = _rect(0, 0, 20, 20, name="a")
        b = _rect(25, 25, 10, 10, name="b")
        image_shape = (40, 40)
        u = combine([a, b], "union")
        expected = a.to_mask(image_shape) | b.to_mask(image_shape)
        result = u.to_mask(image_shape)
        # Allow small rasterization rounding at polygon boundaries
        assert np.sum(result ^ expected) < 10

    def test_intersection_smaller_than_either(self):
        a = _rect(0, 0, 20, 20, name="a")
        b = _rect(10, 10, 20, 20, name="b")
        image_shape = (40, 40)
        i = combine([a, b], "intersection")
        assert i.to_mask(image_shape).sum() <= a.to_mask(image_shape).sum()
        assert i.to_mask(image_shape).sum() <= b.to_mask(image_shape).sum()

    def test_difference_removes_overlap(self):
        a = _rect(0, 0, 30, 30, name="a")
        b = _rect(10, 10, 20, 20, name="b")
        image_shape = (40, 40)
        d = combine([a, b], "difference")
        b_mask = b.to_mask(image_shape)
        d_mask = d.to_mask(image_shape)
        # The difference result should not include any pixel from b's interior
        assert not np.any(b_mask & d_mask)

    def test_xor_symmetric(self):
        a = _rect(0, 0, 20, 20, name="a")
        b = _rect(10, 10, 20, 20, name="b")
        image_shape = (40, 40)
        xor_ab = combine([a, b], "xor")
        xor_ba = combine([b, a], "xor")
        # XOR is commutative
        assert np.array_equal(xor_ab.to_mask(image_shape), xor_ba.to_mask(image_shape))

    def test_combine_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            combine([], "union")

    def test_combine_line_roi_raises(self):
        line = ROI.new("line", {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0})
        rect = _rect(0, 0, 10, 10)
        with pytest.raises(ValueError, match="does not support"):
            combine([rect, line], "union")

    def test_single_roi_union_is_itself(self):
        a = _rect(5, 5, 20, 20, name="solo")
        image_shape = (40, 40)
        u = combine([a], "union")
        expected = a.to_mask(image_shape)
        result = u.to_mask(image_shape)
        assert np.sum(result ^ expected) < 10


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Multipolygon ROI
# ═════════════════════════════════════════════════════════════════════════════

class TestMultipolygon:
    def test_invert_produces_multipolygon(self):
        roi = _rect(10, 10, 20, 20, name="hole")
        result = invert(roi, (64, 64))
        # An interior rectangle inverted → polygon with hole → multipolygon
        assert result.kind == "multipolygon"

    def test_multipolygon_to_mask_shape(self):
        roi = _rect(10, 10, 20, 20, name="hole")
        inv = invert(roi, (64, 64))
        mask = inv.to_mask((64, 64))
        assert mask.shape == (64, 64)
        assert mask.dtype == bool

    def test_multipolygon_direct_construction(self):
        """Directly constructed multipolygon ROI masks correctly."""
        geom = {
            "components": [{
                "exterior": [[0, 0], [20, 0], [20, 20], [0, 20]],
                "holes": [[[5, 5], [15, 5], [15, 15], [5, 15]]],
            }]
        }
        roi = ROI.new("multipolygon", geom, name="donut")
        mask = roi.to_mask((25, 25))
        assert mask.dtype == bool
        # Centre of the hole should be False
        assert not mask[10, 10]
        # Corner of exterior should be True (roughly)
        assert mask[1, 1]

    def test_multipolygon_transform_lossless(self):
        roi = _rect(10, 10, 20, 20, name="hole")
        inv = invert(roi, (64, 64))
        transformed = inv._transform_lossless("flip_horizontal", (64, 64))
        assert transformed.kind == inv.kind
        # Flipping twice should round-trip
        roundtrip = transformed._transform_lossless("flip_horizontal", (64, 64))
        np.testing.assert_allclose(
            roundtrip.to_mask((64, 64)).astype(int),
            inv.to_mask((64, 64)).astype(int),
        )

    def test_multipolygon_transform_crop(self):
        roi = _rect(5, 5, 50, 50, name="big")
        inv = invert(roi, (64, 64))
        # Crop to lower-right 32x32 region — should not raise
        cropped = inv._transform_crop({"x0": 0, "y0": 0, "x1": 32, "y1": 32}, (64, 64))
        assert cropped is not None


# ═════════════════════════════════════════════════════════════════════════════
# 4.  subtract_background — ROI parameters
# ═════════════════════════════════════════════════════════════════════════════

class TestSubtractBackgroundGlobal:
    def test_global_removes_linear_tilt(self):
        arr = _tilted_plane(32, 32)
        result = subtract_background(arr, order=1)
        assert np.max(np.abs(result)) < 1e-9

    def test_global_order2_removes_quadratic(self):
        y = np.linspace(-1, 1, 32)
        x = np.linspace(-1, 1, 32)
        X, Y = np.meshgrid(x, y)
        arr = 0.5 * X**2 + 0.3 * Y**2 + 0.1 * X * Y + 1.0
        result = subtract_background(arr, order=2)
        assert np.max(np.abs(result)) < 1e-9

    def test_global_returns_float64(self):
        arr = np.ones((16, 16), dtype=np.float32)
        result = subtract_background(arr)
        assert result.dtype == np.float64

    def test_global_does_not_mutate_input(self):
        arr = _tilted_plane(16, 16)
        orig = arr.copy()
        subtract_background(arr)
        np.testing.assert_array_equal(arr, orig)


class TestSubtractBackgroundFitRoi:
    def test_fit_roi_fits_only_on_roi_pixels(self):
        """Fit on the clean left half, apply globally."""
        arr = _tilted_plane(32, 32)
        # Right half has a spike — should not affect fit if fit_roi is left half
        arr[:, 16:] += 100.0
        fit_roi = _rect(0, 0, 15, 31, name="left")  # columns 0–15
        result = subtract_background(arr, order=1, fit_roi=fit_roi)
        # Left half (where we fit) should be approximately flat
        assert np.max(np.abs(result[:, :16])) < 1e-6

    def test_fit_roi_result_same_shape(self):
        arr = _tilted_plane(32, 32)
        fit_roi = _rect(5, 5, 20, 20)
        result = subtract_background(arr, order=1, fit_roi=fit_roi)
        assert result.shape == arr.shape

    def test_fit_roi_globally_removes_plane(self):
        """Fit on a sub-region of a pure plane → global subtraction removes all tilt."""
        arr = _tilted_plane(64, 64)
        # ROI covers upper-left quadrant only
        fit_roi = _rect(0, 0, 32, 32, name="q1")
        result = subtract_background(arr, order=1, fit_roi=fit_roi)
        # Since arr is a perfect plane, fitting any sub-region extrapolates correctly
        assert np.max(np.abs(result)) < 1e-9

    def test_fit_roi_too_small_raises(self):
        arr = _tilted_plane(32, 32)
        # 1×1 ROI has fewer pixels than required for order-1 fit (3 terms)
        tiny = _rect(10, 10, 1, 1, name="tiny")
        with pytest.raises(ValueError, match="Fit region"):
            subtract_background(arr, order=1, fit_roi=tiny)


class TestSubtractBackgroundApplyRoi:
    def test_apply_roi_limits_output_pixels(self):
        """With apply_roi, pixels outside the ROI are returned unchanged."""
        arr = _tilted_plane(32, 32)
        apply_roi = _rect(8, 8, 16, 16, name="patch")
        result = subtract_background(arr, order=1, apply_roi=apply_roi)
        # Outside the apply ROI, arr should be unchanged
        mask = apply_roi.to_mask(arr.shape)
        np.testing.assert_array_almost_equal(result[~mask], arr[~mask])

    def test_apply_roi_modifies_inside(self):
        arr = _tilted_plane(32, 32)
        apply_roi = _rect(8, 8, 16, 16, name="patch")
        result = subtract_background(arr, order=1, apply_roi=apply_roi)
        # Inside the ROI the result should differ from input
        mask = apply_roi.to_mask(arr.shape)
        assert not np.allclose(result[mask], arr[mask])

    def test_patch_only_fit_and_apply_same_roi(self):
        """fit_roi == apply_roi: fit and subtract only within the patch."""
        arr = _tilted_plane(32, 32) + 10.0  # pure plane + offset
        patch = _rect(8, 8, 16, 16, name="patch")
        result = subtract_background(arr, order=1, fit_roi=patch, apply_roi=patch)
        mask = patch.to_mask(arr.shape)
        # Inside patch should be approximately flat after subtraction
        assert np.max(np.abs(result[mask])) < 1e-9
        # Outside patch should be unchanged
        np.testing.assert_array_almost_equal(result[~mask], arr[~mask])


class TestSubtractBackgroundExcludeRoi:
    def test_exclude_roi_removes_spike_from_fit(self):
        """Excluding a spike region lets the background fit proceed cleanly."""
        arr = _tilted_plane(32, 32)
        # Spike at centre
        arr[12:20, 12:20] += 1000.0
        spike_roi = _rect(12, 12, 8, 8, name="spike")
        result = subtract_background(arr, order=1, exclude_roi=spike_roi)
        # Away from the spike, result should be close to flat
        mask = spike_roi.to_mask(arr.shape)
        assert np.max(np.abs(result[~mask])) < 1e-6

    def test_exclude_plus_fit_intersection(self):
        """exclude_roi is intersected with fit_roi: only fit_roi AND NOT exclude_roi used."""
        arr = _tilted_plane(32, 32)
        fit_roi = _rect(0, 0, 32, 32, name="all")
        spike_roi = _rect(14, 14, 4, 4, name="spike")
        # Adding the spike should not affect the result if we exclude it
        arr_spiked = arr.copy()
        arr_spiked[14:18, 14:18] += 500.0
        result = subtract_background(
            arr_spiked, order=1, fit_roi=fit_roi, exclude_roi=spike_roi
        )
        assert np.max(np.abs(result[~spike_roi.to_mask(arr.shape)])) < 1e-6


# ═════════════════════════════════════════════════════════════════════════════
# 5.  resolve_roi_expr (state.py)
# ═════════════════════════════════════════════════════════════════════════════

class TestResolveRoiExpr:
    @pytest.fixture
    def roi_set(self):
        rs = ROISet(image_id="test-img")
        rs.add(_rect(0, 0, 20, 20, name="terrace"))
        rs.add(_rect(30, 30, 10, 10, name="patch"))
        return rs

    def test_none_returns_none(self, roi_set):
        assert resolve_roi_expr(None, roi_set, (64, 64)) is None

    def test_ref_by_name(self, roi_set):
        result = resolve_roi_expr({"ref": "terrace"}, roi_set, (64, 64))
        assert result.name == "terrace"

    def test_invert_by_name(self, roi_set):
        result = resolve_roi_expr({"invert": "terrace"}, roi_set, (64, 64))
        assert result.kind in ("polygon", "multipolygon")
        # Result should be the complement
        orig_mask = roi_set.get_by_name("terrace").to_mask((64, 64))
        inv_mask = result.to_mask((64, 64))
        assert not np.any(orig_mask & inv_mask)

    def test_combine_union(self, roi_set):
        result = resolve_roi_expr(
            {"combine": ["terrace", "patch"], "mode": "union"}, roi_set, (64, 64)
        )
        assert result.kind in ("polygon", "multipolygon")

    def test_missing_roi_raises_keyerror(self, roi_set):
        with pytest.raises(KeyError, match="nonexistent"):
            resolve_roi_expr({"ref": "nonexistent"}, roi_set, (64, 64))

    def test_no_roi_set_raises(self):
        with pytest.raises(ValueError, match="roi_set"):
            resolve_roi_expr({"ref": "terrace"}, None, (64, 64))

    def test_non_dict_raises(self, roi_set):
        with pytest.raises(ValueError, match="must be a dict"):
            resolve_roi_expr("terrace", roi_set, (64, 64))

    def test_unknown_keys_raise(self, roi_set):
        with pytest.raises(ValueError, match="Unrecognised"):
            resolve_roi_expr({"intersect": "terrace"}, roi_set, (64, 64))


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Processing state — plane_bg with ROI params
# ═════════════════════════════════════════════════════════════════════════════

class TestProcessingStatePlaneBgRoi:
    @pytest.fixture
    def roi_set(self):
        rs = ROISet(image_id="test-img")
        rs.add(_rect(0, 0, 32, 32, name="fit_region"))
        rs.add(_rect(8, 8, 16, 16, name="apply_region"))
        rs.add(_rect(12, 12, 8, 8, name="exclude_region"))
        return rs

    def test_plane_bg_with_fit_roi_id(self, roi_set):
        arr = _tilted_plane(32, 32)
        fit_roi = roi_set.get_by_name("fit_region")
        state = ProcessingState([
            ProcessingStep("plane_bg", {"order": 1, "fit_roi_id": fit_roi.id})
        ])
        result = apply_processing_state(arr, state, roi_set=roi_set)
        assert result.shape == arr.shape
        assert np.max(np.abs(result)) < 1e-9

    def test_plane_bg_with_fit_roi_expr_ref(self, roi_set):
        arr = _tilted_plane(32, 32)
        state = ProcessingState([
            ProcessingStep("plane_bg", {
                "order": 1,
                "fit_roi_expr": {"ref": "fit_region"},
            })
        ])
        result = apply_processing_state(arr, state, roi_set=roi_set)
        assert np.max(np.abs(result)) < 1e-9

    def test_plane_bg_with_fit_roi_expr_invert(self, roi_set):
        arr = _tilted_plane(32, 32)
        # Adding spike in a small region; we invert it to exclude just that region
        arr[12:20, 12:20] += 1000.0
        state = ProcessingState([
            ProcessingStep("plane_bg", {
                "order": 1,
                "fit_roi_expr": {"invert": "exclude_region"},
            })
        ])
        result = apply_processing_state(arr, state, roi_set=roi_set)
        excl_mask = roi_set.get_by_name("exclude_region").to_mask(arr.shape)
        assert np.max(np.abs(result[~excl_mask])) < 1e-6

    def test_plane_bg_roi_id_no_roi_set_warns(self):
        arr = _tilted_plane(32, 32)
        state = ProcessingState([
            ProcessingStep("plane_bg", {"order": 1, "fit_roi_id": "some-uuid"})
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_processing_state(arr, state, roi_set=None)
        assert any("no roi_set" in str(warning.message).lower() for warning in w)
        assert result.shape == arr.shape

    def test_plane_bg_roi_id_missing_warns(self, roi_set):
        arr = _tilted_plane(32, 32)
        state = ProcessingState([
            ProcessingStep("plane_bg", {"order": 1, "fit_roi_id": "nonexistent-id"})
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_processing_state(arr, state, roi_set=roi_set)
        assert any("not found" in str(warning.message).lower() for warning in w)
        assert result.shape == arr.shape

    def test_plane_bg_apply_roi_id(self, roi_set):
        arr = _tilted_plane(32, 32)
        apply_roi = roi_set.get_by_name("apply_region")
        state = ProcessingState([
            ProcessingStep("plane_bg", {
                "order": 1,
                "apply_roi_id": apply_roi.id,
            })
        ])
        result = apply_processing_state(arr, state, roi_set=roi_set)
        mask = apply_roi.to_mask(arr.shape)
        # Outside apply_roi — unchanged
        np.testing.assert_array_almost_equal(result[~mask], arr[~mask])

    def test_plane_bg_exclude_roi_expr(self, roi_set):
        arr = _tilted_plane(32, 32)
        arr[12:20, 12:20] += 500.0
        state = ProcessingState([
            ProcessingStep("plane_bg", {
                "order": 1,
                "exclude_roi_expr": {"ref": "exclude_region"},
            })
        ])
        result = apply_processing_state(arr, state, roi_set=roi_set)
        excl_mask = roi_set.get_by_name("exclude_region").to_mask(arr.shape)
        assert np.max(np.abs(result[~excl_mask])) < 1e-6


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Canonical use cases end-to-end
# ═════════════════════════════════════════════════════════════════════════════

class TestCanonicalUseCases:
    def test_case1_global_plane_fit(self):
        """Standard background removal — no ROI params."""
        arr = _tilted_plane(64, 64)
        result = subtract_background(arr, order=1)
        assert np.max(np.abs(result)) < 1e-9

    def test_case2_fit_terrace_apply_globally(self):
        """Fit on a clean terrace, subtract globally."""
        arr = _tilted_plane(64, 64)
        # Terrace is the left half; right half has adatoms (don't affect fit)
        arr[:, 32:] += 5.0  # step edge — but it's a constant offset, plane still fits
        terrace_roi = _rect(0, 0, 31, 63, name="terrace")
        result = subtract_background(arr, order=1, fit_roi=terrace_roi)
        # Left half (terrace) should be flat after subtraction
        assert np.max(np.abs(result[:, :32])) < 1e-6

    def test_case3_fit_terrace_apply_region(self):
        """Fit on terrace, apply only to an adjacent region."""
        arr = _tilted_plane(64, 64)
        terrace_roi = _rect(0, 0, 30, 63, name="terrace")
        target_roi = _rect(33, 0, 30, 63, name="target")
        result = subtract_background(
            arr, order=1, fit_roi=terrace_roi, apply_roi=target_roi
        )
        target_mask = target_roi.to_mask(arr.shape)
        terrace_mask = terrace_roi.to_mask(arr.shape)
        # Target region should be flat
        assert np.max(np.abs(result[target_mask])) < 1e-6
        # Terrace region should be unchanged
        np.testing.assert_array_almost_equal(result[terrace_mask], arr[terrace_mask])

    def test_case4_exclude_molecules(self):
        """Fit globally but exclude contaminated molecules."""
        arr = _tilted_plane(64, 64)
        arr[28:36, 28:36] += 20.0  # simulated adatom cluster
        molecules_roi = _rect(28, 28, 8, 8, name="molecules")
        result = subtract_background(arr, order=1, exclude_roi=molecules_roi)
        excl_mask = molecules_roi.to_mask(arr.shape)
        assert np.max(np.abs(result[~excl_mask])) < 1e-6

    def test_case5_patch_only(self):
        """Correct only a small patch — fit and apply to the same region."""
        arr = _tilted_plane(64, 64)
        patch_roi = _rect(20, 20, 20, 20, name="patch")
        result = subtract_background(arr, order=1, fit_roi=patch_roi, apply_roi=patch_roi)
        patch_mask = patch_roi.to_mask(arr.shape)
        outside_mask = ~patch_mask
        # Inside patch should be flat
        assert np.max(np.abs(result[patch_mask])) < 1e-9
        # Outside patch should be unchanged
        np.testing.assert_array_almost_equal(result[outside_mask], arr[outside_mask])
