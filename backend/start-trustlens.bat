@echo off
cd /d "%~dp0backend"
.venv\Scripts\python.exe -m uvicorn main:app
pause