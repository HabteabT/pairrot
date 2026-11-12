@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project environment is missing. Run these two commands first:
  echo   python -m venv .venv
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)
echo Training the vanilla Tsetlin Machine. This can take several minutes.
".venv\Scripts\python.exe" code\train_vanilla.py %*
if errorlevel 1 (
  echo Training stopped with an error. The dashboard still shows the previous model.
  pause
  exit /b 1
)
echo Redrawing the automaton-state histogram for the new model.
".venv\Scripts\python.exe" code\state_histogram.py
echo.
echo Refresh http://127.0.0.1:8000 to see the validation accuracy.
pause
