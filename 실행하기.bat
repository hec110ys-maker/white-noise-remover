@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ============================================================
echo   화이트 노이즈 제거 프로그램을 실행합니다...
echo ============================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else (
    python main.py
)

pause
