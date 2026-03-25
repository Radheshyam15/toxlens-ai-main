@echo off
echo Starting ToxLens Backend...
start "ToxLens Backend" cmd /c "cd backend && py app.py"

echo Starting ToxLens Frontend...
start "ToxLens Frontend" cmd /k "cd "TOXLENS FRONTEND" && py -m http.server 8000"

echo Waiting for servers to start...
timeout /t 2 /nobreak >nul

echo Opening browser...
start http://localhost:8000

echo All set! You can close this window now.
