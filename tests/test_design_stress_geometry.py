from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_geometry_diagnostic_recomputes_and_detects_tamper(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "results/design-open-set-stress-n60"
    analyzer = root / "scripts/analyze_design_stress_geometry.py"
    verifier = root / "scripts/verify_design_stress_geometry.py"
    output = tmp_path / "geometry"
    command = [
        sys.executable,
        str(analyzer),
        "--artifact",
        str(source),
        "--output",
        str(output),
        "--repo-root",
        str(root),
        "--code-sha",
        "1" * 40,
        "--recompute-only",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    verify = [
        sys.executable,
        str(verifier),
        str(output),
        "--source-artifact",
        str(source),
        "--repo-root",
        str(root),
    ]
    subprocess.run(verify, check=True, capture_output=True, text=True)
    geometry = output / "maximin_mixture_geometry.csv"
    geometry.write_text(
        geometry.read_text(encoding="utf-8").replace("0.8,", "0.7,", 1),
        encoding="utf-8",
    )
    failed = subprocess.run(verify, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "payload mismatch" in failed.stderr
