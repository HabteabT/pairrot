@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project environment is missing. Run these two commands first:
  echo   python -m venv .venv
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)
echo Training the required vanilla run: cleaned posts, s = 1.0, boost off.
echo This can take a few minutes. The test pairs are not scored.
".venv\Scripts\python.exe" code\train_vanilla.py --data "dataset\prepared\s457_f1000_t10000_v2000_e2000_val0p2" --s 1.0 --boost 0 --clauses 200 --T 50 --epochs 10 %*
if errorlevel 1 (
  echo Training stopped with an error. The dashboard still shows the previous model.
  pause
  exit /b 1
)
echo Redrawing the accuracy line for the new model.
".venv\Scripts\python.exe" code\accuracy_line.py
echo.
echo Refresh http://127.0.0.1:8000 to see the validation accuracy.
pause
