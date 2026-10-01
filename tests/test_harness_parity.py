"""b4_harness_parity.py on a synthetic fixture: re-scoring with PIXIU's code must reproduce our parses and
scores, skip incomplete files, and (with --strict) fail loudly on a mismatch."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"


def run_b4(pred_dir, out, *extra):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "b4_harness_parity.py"),
                           "--data-dir", str(FIX / "data"), "--pred-dir", str(pred_dir), "--out", str(out), *extra],
                          capture_output=True, text=True, env=env, cwd=ROOT)


def test_fixture_scoring_is_identical_to_pixiu(tmp_path):
    out = tmp_path / "parity.json"
    r = run_b4(FIX / "predictions" / "test", out, "--strict")
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text())
    assert res["identical_to_pixiu_scoring"] is True
    assert res["systems"] == ["toy_bare", "toy_tricky"]          # toy_partial (3 of 8 rows) is skipped
    # toy_bare, by hand: positive F1 2/3, neutral F1 2/3, negative F1 1 -> weighted (3*2/3 + 3*2/3 + 2*1) / 8
    assert "toy_bare                        100.0%    100.0%    0.7500   0.7500" in r.stdout


@pytest.fixture
def pred_copy(tmp_path):
    d = tmp_path / "test"
    shutil.copytree(FIX / "predictions" / "test", d)
    return d


def test_strict_fails_on_a_parse_mismatch(tmp_path, pred_copy):
    f = pred_copy / "toy_tricky.csv"
    f.write_text(f.read_text().replace("toy4,not positive,positive", "toy4,not positive,negative"))
    out = tmp_path / "parity.json"
    r = run_b4(pred_copy, out, "--strict")
    assert r.returncode == 1
    assert json.loads(out.read_text())["identical_to_pixiu_scoring"] is False


def test_mismatch_without_strict_still_exits_zero(tmp_path, pred_copy):
    f = pred_copy / "toy_tricky.csv"
    f.write_text(f.read_text().replace("toy4,not positive,positive", "toy4,not positive,negative"))
    assert run_b4(pred_copy, tmp_path / "parity.json").returncode == 0     # run_all.py keeps going; the report flags it


def test_zero_systems_is_never_reported_as_parity(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "parity.json"
    assert run_b4(empty, out).returncode == 0
    assert run_b4(empty, out, "--strict").returncode == 1
    assert not out.exists()
