@echo off
REM Run 2 windows (Teacher + Student) on one machine for testing.
REM Uses .venv if present, otherwise venv, otherwise system python.
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
start "Teacher" cmd /k "%PY% main.py"
timeout /t 2 /nobreak >nul
start "Student" cmd /k "%PY% main.py"