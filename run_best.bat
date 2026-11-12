@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project environment is missing. Run these two commands first:
  echo   python -m venv .venv
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)
set DATA=dataset\prepared\s457_f300_t30000_v2000_e2000_val0p2_fulltext_topical

echo Step 1 of 3. Building the pairs: full posts, 300 topical words, 30000 training pairs.
if exist "%DATA%\features_train.npz" (
  echo   Already built. Delete %DATA% to build it again.
) else (
  ".venv\Scripts\python.exe" dataset\prepare_data.py --full-text --vocabulary topical --max-features 300 --train-pairs 30000 --val-pairs 2000 --test-pairs 2000
  if errorlevel 1 (
    echo Building the pairs stopped with an error.
    pause
    exit /b 1
  )
)

echo Step 2 of 3. Training the vanilla Tsetlin Machine, s = 1.0, boost on. About 10 to 20 minutes.
".venv\Scripts\python.exe" code\train_vanilla.py --data "%DATA%" --boost 1 --clauses 800 --T 50 --epochs 8
if errorlevel 1 (
  echo Training stopped with an error. The dashboard still shows the previous model.
  pause
  exit /b 1
)

echo Step 3 of 3. Drawing the automaton-state histogram.
".venv\Scripts\python.exe" code\state_histogram.py

echo.
echo Done. Double-click start.bat, or refresh http://127.0.0.1:8000, to see the validation accuracy.
pause
