@echo off
rem Starts the Nyaya API (new window) and the Streamlit UI using the project venv,
rem so the system / Anaconda Python is never picked up by mistake.
cd /d "%~dp0"

if not exist ".nyaya\Scripts\python.exe" (
    echo Project venv not found at backend\.nyaya
    echo Create it and install the requirements first:
    echo     python -m venv .nyaya
    echo     .nyaya\Scripts\pip install -r requirements.txt
    exit /b 1
)

start "Nyaya API" .nyaya\Scripts\python.exe -m api.main
.nyaya\Scripts\python.exe -m streamlit run streamlit_app.py
