@echo off
:: ============================================================
:: Trajectory Log Analyzer — INSTALL SCRIPT
:: Run this once on the TBox to set up the environment.
:: ============================================================
title Trajectory Log Analyzer — Install

echo.
echo =====================================================
echo   Varian Trajectory Log Analyzer  ^|  INSTALLER
echo =====================================================
echo.

:: ── Find Anaconda Python ─────────────────────────────────
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
echo [ERROR] Could not find Python. Please install Anaconda or Python first.
pause
exit /b 1

:found_python
echo [OK] Found Python: %PYTHON%
echo.

:: ── Install dependencies ──────────────────────────────────
echo Installing / updating required packages...
echo (This may take a few minutes on first run)
echo.
%PYTHON% -m pip install --upgrade ^
    streamlit ^
    pylinac ^
    numpy ^
    pandas ^
    plotly ^
    --quiet --no-warn-script-location

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Package installation failed.
    echo         Check your internet connection or proxy settings.
    pause
    exit /b 1
)

:: ── Create the launcher shortcut on Desktop ───────────────
set SCRIPT_DIR=%~dp0
set SHORTCUT=%USERPROFILE%\Desktop\Trajectory Log Analyzer.lnk

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%SCRIPT_DIR%launch.bat'; $s.WorkingDirectory = '%SCRIPT_DIR%'; $s.IconLocation = 'shell32.dll,21'; $s.Description = 'Varian Trajectory Log Analyzer'; $s.Save()"

if exist "%SHORTCUT%" (
    echo [OK] Desktop shortcut created.
) else (
    echo [WARN] Could not create desktop shortcut — you can still run launch.bat directly.
)

echo.
echo =====================================================
echo   Installation complete!
echo.
echo   To start the app, double-click:
echo     Desktop ^> "Trajectory Log Analyzer"
echo   or run launch.bat in this folder.
echo.
echo   The app opens in your browser at:
echo     http://localhost:8501
echo   Accessible from any PC on the network at:
echo     http://^<this-PC-IP^>:8501
echo =====================================================
echo.
pause
