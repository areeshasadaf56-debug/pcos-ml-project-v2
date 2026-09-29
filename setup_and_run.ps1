# ============================================
# PCOS ML Project - Auto Setup & Run Script
# ============================================
# Right-click this file and "Run with PowerShell",
# or run:  .\setup_and_run.ps1   from inside the project folder.

# Allow this script to run for this session (avoids the ExecutionPolicy error)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force

# Move to the folder this script lives in (so it works no matter where you double-click it from)
Set-Location -Path $PSScriptRoot

Write-Host "==> Working directory: $PSScriptRoot" -ForegroundColor Cyan

# 1. Create virtual environment if it doesn't exist yet
if (-Not (Test-Path ".\venv")) {
    Write-Host "==> Creating virtual environment (venv)..." -ForegroundColor Cyan
    python -m venv venv
} else {
    Write-Host "==> venv already exists, skipping creation." -ForegroundColor Green
}

# 2. Activate the virtual environment
Write-Host "==> Activating venv..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

# 3. Install/upgrade required packages
Write-Host "==> Installing dependencies..." -ForegroundColor Cyan
if (Test-Path ".\requirements.txt") {
    python -m pip install -r requirements.txt
} else {
    python -m pip install fastapi uvicorn pydantic
}

# 4. Start the server
Write-Host "==> Starting server on http://0.0.0.0:8000 ..." -ForegroundColor Cyan
uvicorn app_backend.main:app --host 0.0.0.0 --port 8000 --reload
