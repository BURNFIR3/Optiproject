# How to Run NeuroScan Pro

## Prerequisites

- **Windows 10/11** (64‑bit)
- **Python 3.11** (or newer) installed and added to your `PATH`. You can download it from https://www.python.org/downloads/windows/
- **Git** (optional, for cloning the repo) – https://git-scm.com/download/win

## Step‑by‑Step Setup

1. **Open PowerShell** (no admin rights required).
2. **Navigate to the project folder** where you have downloaded `NeuroScan_Pro`:
   ```powershell
   cd C:\Users\huzef\Downloads\NeuroScan_Pro
   ```
3. **Create a virtual environment** (the launch script will do this automatically, but you can do it manually if you prefer):
   ```powershell
   python -m venv venv
   ```
4. **Activate the virtual environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
5. **Install required packages** (the script also handles this):
   ```powershell
   pip install --upgrade pip
   pip install -r backend\requirements.txt
   ```
6. **Run the application** using the provided launch script:
   ```powershell
   .\run.ps1
   ```
   - The script will start a Gradio web‑server on `http://127.0.0.1:7860`.
   - Press **Ctrl +C** in the PowerShell window to stop the server.

## Alternative: Run Directly with Python
If you prefer not to use the PowerShell script, you can start the backend manually:
```powershell
python backend\app.py
```
The same URL will be shown.

## First‑time Model Download
The first run will load the TensorFlow model from `model/brain_tumor_model_finetuned.keras`. This may take a few seconds; subsequent runs are instantaneous.

## Troubleshooting
- **"pip is not recognized"** – ensure Python was added to your system `PATH` during installation.
- **Virtual‑env activation errors** – you may need to allow script execution:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```
- **TensorFlow GPU warnings** – the app runs on CPU on Windows; no extra installation required.

---
Enjoy using **NeuroScan Pro** for quick AI‑assisted brain‑tumor preview!
