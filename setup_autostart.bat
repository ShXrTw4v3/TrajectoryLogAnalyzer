@echo off
:: ============================================================
:: Trajectory Log Analyzer — AUTO-START SETUP
:: Registers the app as a Windows Task Scheduler job that
:: starts automatically when the TBox boots (no login needed).
::
:: Run this script as Administrator.
:: ============================================================
title TLog Analyzer — Auto-start Setup

:: Must be run as Admin
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] This script must be run as Administrator.
    echo Right-click ^> "Run as administrator"
    pause
    exit /b 1
)

set TASK_NAME=TrajectoryLogAnalyzer
set LAUNCH=%~dp0launch.bat

:: Remove existing task if present
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

:: Register new task: runs at system startup, SYSTEM account (no login needed)
schtasks /create ^
  /tn "%TASK_NAME%" ^
  /tr "\"%LAUNCH%\"" ^
  /sc ONSTART ^
  /ru SYSTEM ^
  /rl HIGHEST ^
  /delay 0001:00 ^
  /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Auto-start task registered successfully.
    echo      The app will start automatically 1 minute after the TBox boots.
    echo      It will be accessible at http://^<TBox-IP^>:8501 from any browser.
    echo.
    echo      To remove auto-start, run:
    echo        schtasks /delete /tn "%TASK_NAME%" /f
) else (
    echo.
    echo [ERROR] Failed to register task. Check you ran as Administrator.
)
pause
