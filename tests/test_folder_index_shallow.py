"""Tests for probeflow.core.indexing.index_folder_shallow."""

from __future__ import annotations

from pathlib import Path

import pytest

from probeflow.core.indexing import (
    ShallowFolderIndex,
    SubfolderEntry,
    index_folder_shallow,
)


TESTDATA = Path(__file__).resolve().parents[1] / "test_data"


def test_returns_shallow_index_object():
    idx = index_folder_shallow(TESTDATA)
    assert isinstance(idx, ShallowFolderIndex)
    assert idx.folder == TESTDATA


def test_files_are_only_at_immediate_level():
    idx = index_folder_shallow(TESTDATA)
    for item in idx.files:
        assert item.path.parent == TESTDATA


def test_subfolders_listed():
    idx = index_folder_shallow(TESTDATA)
    names = {sub.name for sub in idx.subfolders}
    # test_data/ ships with sample_input/, output_png/, output_sxm/
    assert "sample_input" in names
    assert all(isinstance(sub, SubfolderEntry) for sub in idx.subfolders)


def test_subfolder_sample_paths_are_scan_files():
    idx = index_folder_shallow(TESTDATA)
    sample_input = next(s for s in idx.subfolders if s.name == "sample_input")
    assert sample_input.n_scans >= 1
    # Up to 3 sample paths, each should point to a real scan file under that folder
    assert 0 < len(sample_input.sample_scan_paths) <= 3
    for p in sample_input.sample_scan_paths:
        assert p.exists()
        assert p.suffix.lower() in (".dat", ".sxm")


def test_subfolders_alpha_sorted():
    idx = index_folder_shallow(TESTDATA)
    names = [s.name for s in idx.subfolders]
    assert names == sorted(names, key=str.lower)


def test_skipdirs_are_omitted(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "real").mkdir()
    idx = index_folder_shallow(tmp_path)
    names = {sub.name for sub in idx.subfolders}
    assert names == {"real"}


def test_nonexistent_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        index_folder_shallow(tmp_path / "does_not_exist")


def test_file_path_raises(tmp_path):
    f = tmp_path / "not_a_dir.txt"
    f.write_text("hi")
    with pytest.raises(ValueError):
        index_folder_shallow(f)


def test_empty_folder(tmp_path):
    idx = index_folder_shallow(tmp_path)
    assert idx.files == []
    assert idx.subfolders == []
