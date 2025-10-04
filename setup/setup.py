#!/usr/bin/env python3
"""
setup_ecmwf_env.py

Companion setup checker/installer for the ECMWF/Anemoi teaching environment.

Usage examples:
  - Check only (default):         python setup_ecmwf_env.py
  - Install from requirements:    python setup_ecmwf_env.py --install
  - Specify a requirements file:  python setup_ecmwf_env.py --requirements requirements_ECMWF.txt
  - Skip GPU tests:               python setup_ecmwf_env.py --skip-gpu-tests

What it does:
  • Optionally installs packages from the pinned requirements file (pip).
  • Verifies imports and versions for the core stack (PyTorch, xarray, cfgrib, eccodes, Cartopy, Anemoi).
  • Runs small sanity checks (PyTorch CPU, optional CUDA GPU op).
  • Writes a short pass/fail report to setup_report.txt and returns non‑zero exit code on failure.
"""

from __future__ import annotations
import argparse
import os
import sys
import subprocess
import textwrap
import platform
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# --------- Helpers ---------

@dataclass
class CheckResult:
    name: str
    dist: str
    import_name: str
    required: Optional[str]
    installed: Optional[str]
    ok_import: bool
    ok_version: bool
    notes: str = ""

def read_requirements_versions(path: str) -> Dict[str, str]:
    """Parse lines like 'package==1.2.3' into a dict {package_lower: version}."""
    versions: Dict[str, str] = {}
    if not os.path.exists(path):
        return versions
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # keep only exact pins (==); ignore others
            if "==" in line:
                pkg, ver = line.split("==", 1)
                versions[pkg.strip().lower()] = ver.strip()
    return versions

def pip(*args: str) -> int:
    """Run 'python -m pip ...' and return returncode."""
    cmd = [sys.executable, "-m", "pip", *args]
    print(">>", " ".join(cmd))
    return subprocess.run(cmd).returncode

def install_requirements(req_file: str) -> bool:
    print(f"\n[Install] Installing from requirements file: {req_file}\n")
    if not os.path.exists(req_file):
        print(f"[Install] ERROR: requirements file not found at: {req_file}")
        return False
    # Upgrade packaging essentials first (safe and often helps)
    pip("install", "--upgrade", "pip", "setuptools", "wheel")
    # Then install requirements
    rc = pip("install", "--no-cache-dir", "-r", req_file)
    ok = (rc == 0)
    print(f"[Install] Completed with return code {rc}")
    return ok

def import_and_version(dist_name: str, import_name: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Try to import import_name (or dist_name if None).
    Return (ok_import, installed_version, notes).
    """
    import importlib
    from importlib.metadata import PackageNotFoundError, version as dist_version

    if import_name is None:
        import_name = dist_name

    try:
        mod = importlib.import_module(import_name)
        # Distribution name for version lookup is the dist/package name from requirements (hyphens OK).
        ver = None
        try:
            ver = dist_version(dist_name)
        except PackageNotFoundError:
            # fallback: try import_name as a distribution
            try:
                ver = dist_version(import_name)
            except PackageNotFoundError:
                ver = getattr(mod, "__version__", None)
        return True, ver, None
    except Exception as e:
        return False, None, f"Import error: {e.__class__.__name__}: {e}"

def compare_versions(installed: Optional[str], required: Optional[str]) -> bool:
    if required is None or installed is None:
        # if we don't know either, don't fail the check purely on version
        return True
    return installed == required

def write_report(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# --------- Core Check Sets ---------

CORE_CHECKS: List[Tuple[str, str, str]] = [
    # (display name, distribution name, import name)
    ("NumPy", "numpy", "numpy"),
    ("Pandas", "pandas", "pandas"),
    ("SciPy", "scipy", "scipy"),
    ("Matplotlib", "matplotlib", "matplotlib"),
]

PYTORCH_CHECKS: List[Tuple[str, str, str]] = [
    ("PyTorch", "torch", "torch"),
    ("Torchvision", "torchvision", "torchvision"),
    ("Torch Geometric", "torch-geometric", "torch_geometric"),
]

WEATHER_IO_CHECKS: List[Tuple[str, str, str]] = [
    ("xarray", "xarray", "xarray"),
    ("netCDF4", "netCDF4", "netCDF4"),
    ("cfgrib", "cfgrib", "cfgrib"),
    ("eccodes", "eccodes", "eccodes"),
]

VIS_CHECKS: List[Tuple[str, str, str]] = [
    ("Cartopy", "Cartopy", "cartopy"),
    ("Shapely", "shapely", "shapely"),
    ("PyProj", "pyproj", "pyproj"),
]

ANEMOI_CHECKS: List[Tuple[str, str, str]] = [
    ("anemoi-datasets", "anemoi-datasets", "anemoi_datasets"),
    ("anemoi-graphs", "anemoi-graphs", "anemoi_graphs"),
    ("anemoi-models", "anemoi-models", "anemoi_models"),
    ("anemoi-training", "anemoi-training", "anemoi_training"),
    ("anemoi-inference", "anemoi-inference", "anemoi_inference"),
    ("anemoi-transform", "anemoi-transform", "anemoi_transform"),
    ("anemoi-utils", "anemoi-utils", "anemoi_utils"),
]

ALL_GROUPS = [
    ("Core", CORE_CHECKS),
    ("PyTorch", PYTORCH_CHECKS),
    ("Weather I/O", WEATHER_IO_CHECKS),
    ("Geospatial/Vis", VIS_CHECKS),
    ("Anemoi", ANEMOI_CHECKS),
]

# --------- Diagnostics ---------

def pytorch_sanity(skip_gpu_tests: bool) -> Tuple[bool, str]:
    """Run a tiny CPU (and optional GPU) sanity check for torch."""
    lines = []
    ok = True
    try:
        import torch
        lines.append(f"torch: {torch.__version__}")
        cuda_built = getattr(torch.version, 'cuda', None)
        lines.append(f"CUDA build: {cuda_built}")
        lines.append(f"CUDA available (runtime): {torch.cuda.is_available()}")
        try:
            cudnn_ver = torch.backends.cudnn.version()
        except Exception:
            cudnn_ver = None
        lines.append(f"cuDNN version: {cudnn_ver}")

        # CPU matmul
        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        c = a @ b
        lines.append(f"CPU matmul OK: {tuple(c.shape)}")

        # Optional GPU test
        if not skip_gpu_tests and torch.cuda.is_available():
            try:
                dev = torch.device("cuda:0")
                a = a.to(dev)
                b = b.to(dev)
                c = a @ b
                dev_name = torch.cuda.get_device_name(0)
                lines.append(f"GPU matmul OK on {dev_name}: {tuple(c.shape)}")
            except Exception as e:
                ok = False
                lines.append(f"GPU test failed: {e}")
        elif not skip_gpu_tests:
            lines.append("GPU test skipped: CUDA not available.")
    except Exception as e:
        ok = False
        lines.append(f"PyTorch sanity failed: {e}")
    return ok, "\n".join(lines)

# --------- Main Logic ---------

def main():
    parser = argparse.ArgumentParser(
        description="Install and verify the ECMWF/Anemoi teaching environment."
    )
    parser.add_argument(
        "--requirements", "-r",
        default="requirements_ECMWF.txt",
        help="Path to requirements file with pinned versions (default: requirements_ECMWF.txt).",
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Install packages from the requirements file using pip before verification."
    )
    parser.add_argument(
        "--skip-gpu-tests", action="store_true",
        help="Skip CUDA/GPU runtime tests."
    )
    args = parser.parse_args()

    t0 = time.time()
    # Print env header
    header = textwrap.dedent(f"""
    ===============================================
      ECMWF/Anemoi Environment Setup & Verifier
    ===============================================
    Python:   {sys.version.split()[0]}  ({platform.python_implementation()})
    Platform: {platform.system()} {platform.release()} ({platform.machine()})
    Executable: {sys.executable}
    """).strip()
    print(header)

    # Optional install
    if args.install:
        ok_install = install_requirements(args.requirements)
        if not ok_install:
            print("\n[Install] Installation step encountered errors. Continuing to verification anyway...\n")

    # Parse versions from requirements
    req_versions = read_requirements_versions(args.requirements)

    # Run checks
    from importlib import metadata as importlib_metadata
    results: List[CheckResult] = []

    def check_group(group_name: str, items: List[Tuple[str, str, str]]):
        print(f"\n--- Verifying: {group_name} ---")
        for display, dist, import_name in items:
            required = req_versions.get(dist.lower())
            ok_imp, installed, note = import_and_version(dist, import_name)
            ok_ver = compare_versions(installed, required) if ok_imp else False
            res = CheckResult(
                name=display, dist=dist, import_name=import_name,
                required=required, installed=installed,
                ok_import=ok_imp, ok_version=ok_ver, notes=note or ""
            )
            results.append(res)
            status = "OK" if (res.ok_import and res.ok_version) else ("IMPORT FAIL" if not res.ok_import else "VERSION MISMATCH")
            print(f"{display:20s}  import:{'✔' if res.ok_import else '✖'}  "
                  f"version:{(installed or 'N/A'):>12s}  "
                  f"required:{(required or '—'):>12s}  [{status}]")
            if note:
                print(f"   -> {note}")

    for gname, group in ALL_GROUPS:
        check_group(gname, group)

    # PyTorch sanity checks
    print("\n--- PyTorch Sanity ---")
    torch_ok, torch_report = pytorch_sanity(skip_gpu_tests=args.skip_gpu_tests)
    print(torch_report)

    # Summarize
    n = len(results)
    n_imp_fail = sum(1 for r in results if not r.ok_import)
    n_ver_mismatch = sum(1 for r in results if r.ok_import and not r.ok_version)
    all_ok = (n_imp_fail == 0 and n_ver_mismatch == 0 and torch_ok)

    summary = textwrap.dedent(f"""
    ===============================================
    Summary
    ===============================================
    Checks run:        {n}
    Import failures:   {n_imp_fail}
    Version mismatches:{n_ver_mismatch}
    PyTorch sanity:    {"OK" if torch_ok else "FAILED"}
    Result:            {"✅ ALL GOOD" if all_ok else "❌ ISSUES FOUND"}
    Elapsed:           {time.time()-t0:.1f}s

    Tips:
      • If Cartopy import fails, ensure wheels match your platform (GEOS/PROJ); retry with pip after upgrading pip/wheel.
      • If torch/torch-geometric mismatch, ensure versions match. You may need platform‑specific wheels.
      • If CUDA is expected but unavailable, check your NVIDIA drivers and CUDA runtime match the PyTorch build.
    """).strip()

    print("\n" + summary)

    # Write report
    report_txt = header + "\n\n" + "\n".join(
        f"{r.name:20s} import:{'OK' if r.ok_import else 'FAIL'} | installed:{r.installed or 'N/A':>12s} | "
        f"required:{r.required or '—':>12s} | version_ok:{'YES' if r.ok_version else 'NO'}"
        for r in results
    ) + "\n\n--- PyTorch Sanity ---\n" + torch_report + "\n\n" + summary + "\n"
    write_report("setup_report.txt", report_txt)

    # Exit code
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
