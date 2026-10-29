@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 goto failed

where gh >nul 2>nul
if errorlevel 1 goto missing_gh

call gh auth status >nul 2>nul
if errorlevel 1 goto not_authenticated

echo [release.bat] Build and publish a release from an existing GitHub tag.
echo [release.bat] Existing release assets may be replaced.
choice /c YN /n /m "Continue [Y/N]? "
if errorlevel 2 goto cancelled
if errorlevel 1 goto dispatch
goto failed

:dispatch
if "%~1"=="" goto prompt_tag
call gh workflow run release.yml --raw-field "tag=%~1"
goto dispatch_result

:prompt_tag
call gh workflow run release.yml

:dispatch_result
if errorlevel 1 goto failed
echo.
echo [release.bat] Release workflow requested. Publication is not complete yet.
echo [release.bat] Check progress: gh run list --workflow release.yml
pause
exit /b 0

:missing_gh
echo [release.bat] GitHub CLI was not found. Install it from https://cli.github.com/
pause
exit /b 1

:not_authenticated
echo [release.bat] GitHub authentication is required. Run: gh auth login
pause
exit /b 1

:cancelled
echo [release.bat] Cancelled. No workflow was started.
pause
exit /b 0

:failed
echo [release.bat] Could not request the release workflow. See the error above.
pause
exit /b 1
