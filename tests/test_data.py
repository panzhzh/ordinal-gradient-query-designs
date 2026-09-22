"""Portable frozen data and current-paper point estimates."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_data_checksums_and_paper_numbers():
    spec = importlib.util.spec_from_file_location(
        "check_reference", ROOT / "scripts/check_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def test_portable_files():
    manifest = __import__("json").loads((ROOT / "data/manifest.json").read_text())
    for relative in manifest["files"]:
        assert not Path(relative).is_absolute()
        assert not (ROOT / relative).is_symlink()
        assert (ROOT / relative).resolve().is_relative_to(ROOT)
