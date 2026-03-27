@echo off
REM manager 서비스: 동기화 API (private 망, 포트 8091)
REM __pycache__ 생성 방지
set PYTHONDONTWRITEBYTECODE=1
cd /d "%~dp0.."
uvicorn app.manager.main:app --reload --host 0.0.0.0 --port 8091
