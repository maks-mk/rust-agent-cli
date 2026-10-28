@echo off
setlocal
cd /d "%~dp0"

if not exist "rust-agent.exe" (
    echo [start.bat] rust-agent.exe was not found next to this script.
    pause
    exit /b 1
)

rem Launch in a new Windows Terminal window and keep it open to show startup errors.
rem "%~dp0." avoids a trailing backslash directly before the closing quote.
wt -w new -d "%~dp0." cmd /k ".\rust-agent.exe" 2>nul
if not errorlevel 1 exit /b 0

echo [start.bat] Windows Terminal is unavailable, starting in this console...
".\rust-agent.exe"
if errorlevel 1 pause
exit /b
