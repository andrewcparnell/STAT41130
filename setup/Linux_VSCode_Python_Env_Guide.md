# Linux + VS Code: Python Environment Setup (with GPU PyTorch and ECMWF tools)

This guide shows how to create a **Python 3.11** virtual environment on **Linux**, install your **requirements_ECMWF.txt**, and configure **VS Code**. It includes sanity checks and fixes for common issues (GRIB/ecCodes, CUDA, `nvidia-*` extras, Triton, etc.).

> These steps assume Debian/Ubuntu-like systems. For Fedora/Arch, adapt the package manager commands.

---

## 0) Prerequisites

- **Linux** (Ubuntu 22.04+ recommended)
- **VS Code** installed
- **Python 3.11** (preferred) or **Python 3.10**
- (Optional, for GPU) Recent **NVIDIA driver** installed. You **do not** need to install the full CUDA Toolkit; the PyTorch wheels include the needed CUDA runtime.
- Your `requirements_ECMWF.txt` in the working directory.

---

## 1) Install essential system packages

These provide ecCodes (for GRIB via `cfgrib`) and a few build tools. Run in a terminal:

```bash
sudo apt update
# ecCodes runtime and CLI tools
sudo apt install -y libeccodes0 eccodes
# Python venv & headers, basic build chain (use python3.11-venv if you target 3.11 explicitly)
sudo apt install -y python3-venv python3-dev build-essential pkg-config
# Optional but useful for geospatial wheels and SSL
sudo apt install -y libssl-dev curl git
```

> If you plan to compile geospatial libs yourself, you might also install `libproj-dev libgeos-dev`. Most users won’t need that because PyPI wheels for `pyproj`, `shapely`, and `cartopy` are commonly available.

---

## 2) Create and activate a virtual environment

Pick Python 3.11 (recommended).

```bash
# If your default "python3" is 3.11, this is fine:
python3 -m venv ~/.venvs/stat41130

# If you have multiple Pythons, be explicit:
# python3.11 -m venv ~/.venvs/stat41130

# Activate
source ~/.venvs/stat41130/bin/activate
```

Your prompt should now show `(stat41130)`.

Verify the interpreter inside the venv:
```bash
python -c "import sys; print(sys.version)"
pip --version
```

---

## 3) Upgrade pip tooling

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 4) Install CUDA-enabled PyTorch (Linux)

- **Stable CUDA 12.4 wheels (most common):**
```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
```

- **If you have a very new GPU (e.g., Blackwell sm_120)** and see a compute capability warning with cu124, try **CUDA 12.8** builds:
```bash
# Try stable cu128 if available:
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# If stable cu128 is not available for your distro/Python yet, try nightlies:
# pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision
```

> Stick to **one** CUDA channel (cu124 **or** cu128) per environment. Do **not** mix them.

---

## 5) Install your `requirements_ECMWF.txt`

From the same terminal (venv active):
```bash
pip install -r ./requirements_ECMWF.txt
```

### Notes on Linux package pins
- On **Linux**, it’s fine to keep `eccodeslib`, `eckitlib`, and `fckitlib` pins if your file includes them.
- If you also pin `nvidia-*` packages (e.g., `nvidia-nccl-cu12`, `nvidia-cufile-cu12`) or `triton` and the solver fails:
  - Consider **removing** those explicit pins and letting PyTorch’s wheels handle CUDA dependencies.
  - Or gate them to specific Python versions using environment markers (see below).

**Example of cross-platform, version-gated pins (in `requirements_ECMWF.txt`):**
```text
# Core GRIB
cfgrib==0.9.15.0
eccodeslib==2.43.0
eckitlib==1.31.4
fckitlib==0.14.0

# Anemoi (example compatible set)
anemoi-training==0.6.5
anemoi-graphs==0.6.6
anemoi-models==0.9.5
anemoi-datasets==0.5.26
anemoi-transform==0.1.17

# Optional (Linux-only) — uncomment only if you really need them and they match your Python:
# nvidia-nccl-cu12==<compatible>   ; platform_system == "Linux"
# nvidia-cufile-cu12==<compatible> ; platform_system == "Linux" and python_version < "3.11"
# triton==<compatible>             ; platform_system == "Linux"
```

If a specific pin fails with “No matching distribution”, **comment it out temporarily** and re-run `pip install -r ...` to identify the next blocker.

---

## 6) Sanity checks

### 6.1 PyTorch + CUDA
```bash
python - << 'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA build:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY
```

### 6.2 GRIB reading stack (ecCodes + cfgrib + xarray)
```bash
python - << 'PY'
import eccodes, cfgrib, xarray as xr, sys, subprocess
print("eccodes + cfgrib + xarray imports: OK")
subprocess.run([sys.executable, "-m", "cfgrib", "selfcheck"], check=False)
PY
```

> `cfgrib selfcheck` should report a found ecCodes with a version (e.g., v2.43.x).

---

## 7) Configure VS Code to use the venv

1. **Install extensions**: *Python* and *Pylance* (Microsoft).
2. **Select the interpreter**:  
   - `Ctrl+Shift+P` → **Python: Select Interpreter**  
   - Choose `~/.venvs/stat41130/bin/python`
3. **New terminal uses the venv**:  
   - In VS Code: **Terminal → New Terminal** → it should show `(stat41130)`.
4. **Verify imports**: create a file `check_env.py`:
   ```python
   import torch, xarray as xr, cfgrib, eccodes
   print("Environment OK")
   ```
   Run it: **Run → Run Without Debugging** or `python check_env.py` from the integrated terminal.

---

## 8) Common issues & fixes

- **No matching distribution** for `nvidia-*` or `triton` pins: remove those lines or switch to a compatible Python (many packages restrict to `<3.11` or exact CUDA channels). Often you don’t need them explicitly—PyTorch wheels suffice.
- **Compute capability mismatch** (e.g., sm_120 on cu124): try **cu128** builds (stable or nightly). If that still fails, temporarily use CPU or consider updating drivers.
- **ecCodes not found**: ensure `libeccodes0` and `eccodes` (CLI) are installed. Re-run the `selfcheck` snippet above.
- **Resolver backtracking / timeouts**: upgrade pip (`pip install -U pip`) and install PyTorch **before** the rest.
- **Permissions in user site**: keep using a venv; avoid `sudo pip`.

---

## 9) Deactivate / remove the environment

```bash
deactivate   # leave the venv
# Remove it entirely:
rm -rf ~/.venvs/stat41130
```

---

## 10) Optional: Python 3.10 environment

```bash
python3.10 -m venv ~/.venvs/stat41130-py310
source ~/.venvs/stat41130-py310/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
# If you need astropy on 3.10: pin astropy<7 in your requirements
pip install -r ./requirements_ECMWF.txt
```

---

### Done!

You now have a Linux environment ready for ECMWF/GRIB tooling and (optionally) GPU-accelerated PyTorch, wired up in VS Code.
