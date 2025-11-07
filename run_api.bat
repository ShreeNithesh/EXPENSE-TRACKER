@echo off
REM Run this from project root after activating your venv
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
pause
