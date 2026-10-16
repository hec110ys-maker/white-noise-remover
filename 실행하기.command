#!/bin/bash
# Move to the project root directory
cd "$(dirname "$0")"

# Ensure Homebrew and standard system paths are available
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

echo "============================================================"
echo "      동영상 화이트 노이즈 제거 프로그램을 시작합니다..."
echo "============================================================"

# Check virtual environment
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC="./.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_EXEC="python3"
else
    echo "❌ Python을 찾을 수 없습니다."
    read -p "엔터 키를 누르면 종료합니다..."
    exit 1
fi

# Run main application
$PYTHON_EXEC main.py

# Keep window open if crashed or closed
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "⚠️ 프로그램이 비정상 종료되었습니다 (종료 코드: $EXIT_CODE)."
    echo "위의 에러 메시지를 확인해 주세요."
    echo ""
    read -p "엔터 키를 누르면 창이 닫힙니다..."
fi
