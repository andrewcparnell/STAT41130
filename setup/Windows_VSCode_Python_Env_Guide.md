# Windows + VS Code: Python Environment Setup (with GPU PyTorch and ECMWF tools)

This guide walks you through creating a **Python 3.11** (or 3.10) virtual environment on **Windows**, installing your **Windows-safe requirements**, and configuring **VS Code** to use it. It also includes quick sanity checks and fixes for common installation errors you might see on Windows.

---

## 0) Prerequisites

- **Windows 10/11**
- **VS Code** installed
- **Python 3.11** (preferred) or **Python 3.10** installed from python.org  
  - During install, tick **“Add Python to PATH”** (optional; you can also call the full path).
- **PowerShell** (use the VS Code terminal or Windows Terminal)

> If you have multiple Python versions, you can always call the exact one you want using its **full path**, or create an alias for `python3.11` (instructions below).

---

## 1) Open PowerShell (In VS Code click View > Terminal) and locate Python 3.11 (or 3.10)

```powershell
# Show where PowerShell would find these commands (may be blank if not on PATH)
Get-Command python -All
Get-Command python3.11 -All  # optional

# Common install path check (adjust if installed elsewhere)
& "$env:LocalAppData\Programs\Python\Python311\python.exe" --version
```

If that prints `Python 3.11.x`, use it. If not, check `C:\Program Files\Python311\python.exe` or re-run the Python installer.

> **Optional alias** for your PowerShell profile so `python3.11` works as a command:
> ```powershell
> notepad $PROFILE    # add the next line, adjust path if needed
> Set-Alias python3.11 "$env:LocalAppData\Programs\Python\Python311\python.exe"
> . $PROFILE
> python3.11 --version
> ```

---

## 2) Create and activate a virtual environment

Pick **3.11** (recommended) or **3.10**; both are compatible with the curated Windows requirements.

```powershell
# Create venv (3.11 shown; swap in 3.10 path/command if you prefer)
& "$env:LocalAppData\Programs\Python\Python311\python.exe" -m venv "$env:USERPROFILE\.venvs\stat41130-py311"

# Activate
& "$env:USERPROFILE\.venvs\stat41130-py311\Scripts\Activate.ps1"
```

If you see an execution policy error:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "$env:USERPROFILE\.venvs\stat41130-py311\Scripts\Activate.ps1"
```

Your prompt should now show `(stat41130-py311)` on the left. 

**Verify the interpreter inside the venv:**
```powershell
python -c "import sys; print(sys.version)"
pip --version
```

---

## 3) Upgrade pip tooling

```powershell
python -m pip install --upgrade pip setuptools wheel
```

---

## 4) Install CUDA-enabled PyTorch (Windows way)

On Windows, install the official **cu124 wheels** first (these include what you need; do *not* install `nvidia-*` packages directly).

```powershell
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

---

## 5) Install your Windows-safe requirements

Use the requirements file (e.g., `requirements_ECMWF_win.txt`). Use the `cd` command to change directory to the `setup` directory 
Install:
```powershell
pip install -r .\requirements_ECMWF_win.txt
```

---

## 6) Quick sanity checks

### 6.1 PyTorch + CUDA

In Python
```python
# PyTorch + CUDA check
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA build:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

### 6.2 GRIB reading stack (ecCodes + cfgrib + xarray)
```powershell
python - << 'PY'
import eccodes, cfgrib, xarray as xr, sys, subprocess
print("eccodes + cfgrib + xarray imports: OK")
subprocess.run([sys.executable, "-m", "cfgrib", "selfcheck"], check=False)
PY
```

---

## 7) Configure VS Code to use the venv

1. **Install extensions**: *Python* and *Pylance* (Microsoft).
2. **Select the interpreter**:  
   - `Ctrl+Shift+P` → **Python: Select Interpreter**  
   - Pick `~\.venvs\stat41130-py311\Scripts\python.exe` (or your 3.10 venv).
3. **Integrated Terminal uses the venv**:  
   - Open a new terminal in VS Code (**Terminal → New Terminal**).  
   - You should see `(stat41130-py311)` at the left of the prompt.
4. **Verify imports in VS Code**: create a new Python file and run:
   ```python
   import torch, xarray as xr, cfgrib, eccodes
   print("OK")
   ```
   Use **Run Python File** (triangle button) or **Run → Run Without Debugging**.

---

## 8) Common Windows pitfalls & fixes

- **Package not found**:  
  - `eccodeslib`, `eckitlib`, `fckitlib`, `triton`, `nvidia-nccl-cu12`, `nvidia-cufile-cu12` are **not** for Windows via PyPI. Remove those lines or gate them to Linux with environment markers.
- **Astropy 7.x on 3.10**: requires Python ≥3.11. On 3.10, pin `astropy<7`. On 3.11, 7.x is fine.
- **Execution policy blocks venv activation**: Use  
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the current session.
- **App Execution Aliases interfere with python.exe**: Disable them at  
  *Settings → Apps → App execution aliases* (turn off Python entries).
- **Multiple Python installs**: Always call the exact one you want (full path), or set a PowerShell alias in `$PROFILE`.
- **Stuck resolver / cached wheels**: Try `pip install --upgrade pip`, or clear cache with `pip cache purge` (pip ≥23.1).

---

## 9) Deactivate / remove the environment

```powershell
deactivate  # leaves the venv
# To remove it entirely:
Remove-Item -Recurse -Force "$env:USERPROFILE\.venvs\stat41130-py311"
```

---

## 10) (Optional) Create a 3.10 venv instead

```powershell
& "C:\Users\YOU\AppData\Local\Programs\Python\Python310\python.exe" -m venv "$env:USERPROFILE\.venvs\stat41130-py310"
& "$env:USERPROFILE\.venvs\stat41130-py310\Scripts\Activate.ps1"
python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
# If you need Astropy on 3.10: pin astropy<7
pip install -r .\requirements_ECMWF_win.txt
```

---

### Done!

You now have a Windows-friendly environment with GPU PyTorch and ECMWF/GRIB tooling, ready to use inside VS Code.
