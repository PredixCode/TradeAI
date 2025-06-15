@echo off
:: This script automates the project setup by creating a virtual environment
:: and installing all required packages into it.

echo --- Starting Project Setup ---

:: 1. Check if the virtual environment folder already exists.
IF EXIST "venv" (
    echo Virtual environment 'venv' already exists. Skipping creation.
) ELSE (
    echo Creating Python virtual environment in 'venv' folder...
    python -m venv venv
)

:: 2. Upgrade pip within the virtual environment.
echo.
echo --- Upgrading pip in the virtual environment ---
call .\venv\Scripts\python.exe -m pip install --upgrade pip

:: 3. Install required packages from requirements.txt into the virtual environment.
echo.
echo --- Installing required packages from requirements.txt ---
call .\venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo --- Setup Complete! ---
echo To use the virtual environment, you must activate it in your terminal:
echo.
echo   .\venv\Scripts\activate
echo.

pause