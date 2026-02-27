@echo off
:: ============================================================
:: Trajectory Log Analyzer — LAUNCHER
:: Double-click to start the app. Runs in the background.
:: ============================================================
title Trajectory Log Analyzer

:: ── Find Python (same logic as install.bat) ───────────────
set PYTHON=
for %%P in (
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\Anaconda3\python.exe"
    "C:\ProgramData\anaconda3\python.exe"
    "C:\Anaconda3\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
) do (
    if exist %%P (
        set PYTHON=%%P
        goto :found_python
    )
)
echo [ERROR] Python not found. Run install.bat first.
pause
exit /b 1

:found_python
set APP=%~dp0app.py
set PORT=8501

:: ── Check if already running on this port ────────────────
netstat -ano | findstr ":%PORT% " | findstr LISTENING >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo App is already running on port %PORT%.
    echo Opening browser...
    start "" "http://localhost:%PORT%"
    exit /b 0
)

:: ── Start Streamlit (minimised console window) ────────────
echo Starting Trajectory Log Analyzer on port %PORT%...
start "TLog Analyzer" /MIN %PYTHON% -m streamlit run "%APP%" ^
    --server.port %PORT% ^
    --server.headless true ^
    --browser.gatherUsageStats false

:: ── Wait for server then open browser ────────────────────
timeout /t 4 /nobreak >nul
start "" "http://localhost:%PORT%"

echo.
echo App running at http://localhost:%PORT%
echo Accessible on the network at http://%COMPUTERNAME%:%PORT%
echo.
echo Close this window or press Ctrl+C in the minimised console to stop the app.
