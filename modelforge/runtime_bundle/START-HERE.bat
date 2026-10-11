@echo off
REM ModelForge Windows 一键入口：默认打开浏览器安装向导
REM Python 查找顺序：安装包内置 runtime\python（零依赖）→ 系统 python → py launcher
cd /d "%~dp0"
if exist "%~dp0runtime\python\python.exe" (
    if "%~1"=="" ( "%~dp0runtime\python\python.exe" installer.py --web ) else ( "%~dp0runtime\python\python.exe" installer.py %* )
    goto :eof
)
where python >nul 2>nul
if %errorlevel%==0 (
    if "%~1"=="" ( python installer.py --web ) else ( python installer.py %* )
    goto :eof
)
where py >nul 2>nul
if %errorlevel%==0 (
    if "%~1"=="" ( py -3 installer.py --web ) else ( py -3 installer.py %* )
    goto :eof
)
echo [错误] 未找到 Python。请先从 https://www.python.org 安装 Python 3.9+ 并勾选 "Add to PATH"。
pause
exit /b 1
