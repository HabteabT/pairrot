@echo off
cd /d "%~dp0"
echo Opening http://127.0.0.1:8000
start "" http://127.0.0.1:8000
".venv\Scripts\python.exe" backend\app.py
pause
