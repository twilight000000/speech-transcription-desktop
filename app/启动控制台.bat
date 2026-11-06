@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "APP_DIR=%~dp0"
for %%I in ("%APP_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "PORTABLE_RUNTIME=%PROJECT_ROOT%\portable_runtime"
set "PORTABLE_PYTHON=%PORTABLE_RUNTIME%\python"
set "PORTABLE_FFMPEG=%PORTABLE_RUNTIME%\ffmpeg\bin"

cd /d "%APP_DIR%"

if exist "%PORTABLE_PYTHON%\pythonw.exe" (
    set "PATH=%PORTABLE_FFMPEG%;%PORTABLE_PYTHON%;%PORTABLE_PYTHON%\Scripts;%PATH%"
    set "PYTHONNOUSERSITE=1"
    set "PYTHONUTF8=1"
    set "HF_HOME=%PORTABLE_RUNTIME%\hf_home"
    set "HF_HUB_CACHE=%PORTABLE_RUNTIME%\hf_cache"
    set "HUGGINGFACE_HUB_CACHE=%PORTABLE_RUNTIME%\hf_cache"
    set "TORCH_HOME=%PORTABLE_RUNTIME%\torch_cache"
    start "" "%PORTABLE_PYTHON%\pythonw.exe" "desktop_console_window.py"
    exit /b 0
)

where pythonw >nul 2>nul
if not errorlevel 1 (
    start "" pythonw "desktop_console_window.py"
    exit /b 0
)

pyw -3 "desktop_console_window.py" >nul 2>nul
if not errorlevel 1 exit /b 0

python "desktop_console_window.py"
exit /b %errorlevel%
