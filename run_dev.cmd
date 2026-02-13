@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] venv yok, olusturuluyor...
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install -r requirements.txt
python live_caption_ui.py
pause
