@echo off
REM Double-click to start the TIMER GUI with the Anaconda Python.
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" 2>nul
python timer_gui.py
pause
