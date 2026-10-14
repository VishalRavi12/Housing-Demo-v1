@echo off
REM Double-click to start the dashboard and open it in the browser.
cd /d "%~dp0"
start "" http://127.0.0.1:5000
python app.py
pause
