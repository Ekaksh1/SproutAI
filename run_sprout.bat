@echo off
title SPROUT AI - Plant Care Assistant Launcher
color 0A

echo =======================================================
echo          SPROUT AI - Smart Plant Care Assistant
echo   College Design Thinking and Methodology (DTM) Project
echo =======================================================
echo.
echo [*] Checking virtual environment (.venv311)...

if exist ".venv311\Scripts\streamlit.exe" (
    echo [*] Virtual environment verified.
    echo [*] Starting Streamlit Botanical Dashboard...
    echo [*] The app will open in your default browser at http://localhost:8501
    echo.
    ".venv311\Scripts\streamlit.exe" run app.py
) else (
    echo [!] Virtual environment not detected at .venv311.
    echo [*] Attempting system streamlit launcher...
    streamlit run app.py
)

pause
