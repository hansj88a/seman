@echo off
REM agent 서비스: 검색 API (public 망, 포트 8090)
REM __pycache__ 생성 방지
set PYTHONDONTWRITEBYTECODE=1
cd /d "%~dp0.."
uvicorn app.packages.agent.main:app --reload --host 0.0.0.0 --port 8090
