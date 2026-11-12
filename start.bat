@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project environment is missing. Run these two commands first:
  echo   python -m venv .venv
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)
echo Opening http://127.0.0.1:8000
start "" http://127.0.0.1:8000
".venv\Scripts\python.exe" backend\app.py
pause
