# 🧠 VS Code Cheat Sheet for Python & Neural Networks

## 🚀 Getting Started

- **Install VS Code**: [https://code.visualstudio.com/](https://code.visualstudio.com/)
- **Add the Python Extension**: Search for “Python” in Extensions (left toolbar → `Ctrl+Shift+X`).
- **Optional but recommended**: Install “Jupyter” and “Pylance” extensions for notebooks and code intelligence.

---

## 🧩 Setting Up Your Environment

- **Create a project folder**: Store your Python scripts and data here.  
- **Open the folder**: `File → Open Folder...`
- **Select Python Interpreter**:  
  - Press `Ctrl+Shift+P` → type “Python: Select Interpreter”.  
  - Choose your virtual environment (e.g., `.venv` or `stat41130`).

---

## ⚙️ Virtual Environments

- Create a virtual environment in the terminal:
  ```
  python -m venv .venv
  ```
- Activate it:
  - Windows: `.venv\Scripts\activate`
  - macOS/Linux: `source .venv/bin/activate`
- Once activated, VS Code should detect it automatically.

---

## 📓 Working with Jupyter Notebooks

- Create a notebook: `File → New File → Save As example.ipynb`
- Use **Shift+Enter** to run a cell.
- Change kernels (top-right corner) if you have multiple environments.

---

## 🧰 Common Features You’ll Use

| Feature | Shortcut / Action | Description |
|----------|------------------|-------------|
| Run Script | `Ctrl+F5` | Runs the current Python file |
| Run Selected Code | `Shift+Enter` | Executes selected code in interactive window |
| Open Terminal | `` Ctrl+` `` | Opens an integrated terminal |
| Comment Line | `Ctrl+/` | Comment/uncomment selected lines |
| Find | `Ctrl+F` | Search within a file |
| Command Palette | `Ctrl+Shift+P` | Access all commands quickly |

---

## 📊 Working with Neural Network Projects

- Use the **integrated terminal** to install and manage packages (e.g. `torch`, `numpy`, `matplotlib`).
- Keep your **Python environment clean** — one virtual environment per project.
- Split your workspace into panels:  
  - Left: File explorer  
  - Right: Code editor  
  - Bottom: Terminal and output logs (useful for monitoring model training)
- Use **Run → Run Python File in Terminal** to execute training scripts and see live output.

---

## 🧠 Debugging & Productivity

- **Set breakpoints** (click beside the line number) to inspect variable values.
- **Use the Variable Explorer** (in the top-right of the Jupyter interface or “Run and Debug” sidebar).
- **Auto-formatting**: `Shift+Alt+F` (requires a formatter like `black` or `autopep8`).

---

## 📦 Keeping Track of Dependencies

- View installed packages:
  ```
  pip list
  ```
- Save dependencies:
  ```
  pip freeze > requirements.txt
  ```
- Reinstall on another computer:
  ```
  pip install -r requirements.txt
  ```

---

## ✅ Tips for Neural Network Projects

- Keep data files and notebooks organised (e.g. `data/`, `models/`, `notebooks/`).
- Use **Git integration** (sidebar) to track changes to code and notebooks.
- Use **W&B or TensorBoard** extensions for experiment tracking if desired.
