@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo The project environment is missing. Run these two commands first:
  echo   python -m venv .venv
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)
echo Training the hierarchical Tsetlin Machine on the same pairs as the vanilla model.
echo The test pairs are not scored.
".venv\Scripts\python.exe" hierarchical\train_hierarchical.py %*
if errorlevel 1 (
  echo Training stopped.
  pause
  exit /b 1
)
echo.
echo The result is in hierarchical\results\hierarchical.json
pause
