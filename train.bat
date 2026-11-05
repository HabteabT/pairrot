@echo off
cd /d "%~dp0"
echo Training the vanilla Tsetlin Machine. This can take several minutes.
".venv\Scripts\python.exe" code\train_vanilla.py
echo.
echo Refresh http://127.0.0.1:8000 to see the validation accuracy.
pause
