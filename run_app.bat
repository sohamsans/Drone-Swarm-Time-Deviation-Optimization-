@echo off
cd /d "%~dp0"
echo Installing dependencies for Desktop App (first run only)...
python -m pip install -r requirements.txt

echo.
echo Starting Drone Swarm Desktop GUI...
python desktop_app.py
pause
