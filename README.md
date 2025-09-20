# Note Agent

## 0) 요구사항

-   Python 3.11
-   OpenAI API Key

## 1) 설치

```bash
# 가상환경(선택)
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

## 2) 환경변수

루트에 `.env` 파일 생성:

```env
OPENAI_API_KEY=sk-xxxx...
```

## 3) 서버 실행

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

-   Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
-   LangServe Playground: [http://localhost:8000/writer-agent/playground/](http://localhost:8000/writer-agent/playground/)

## 4) Trouble Shooting

-   /openapi.json 500 (Pydantic/Docs 에러)
    아래 조합으로 고정 설치:

```bash
pip install --no-cache-dir "pydantic==2.9.2" "fastapi==0.112.2" "starlette==0.38.2" "langserve==0.3.1" "sse-starlette>=2.0.0"

```

## 4) vscode launch.json

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(FastAPI) Note Agent (venv, Windows)",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "python": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
            "args": [
                "server:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--log-level",
                "debug"
            ],
            "cwd": "${workspaceFolder}",
            "envFile": "${workspaceFolder}/.env",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            },
            "justMyCode": true,
            "console": "integratedTerminal"
        },
        {
            "name": "(FastAPI) Note Agent (venv, macOS/Linux)",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "python": "${workspaceFolder}/.venv/bin/python",
            "args": [
                "server:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--log-level",
                "debug"
            ],
            "cwd": "${workspaceFolder}",
            "envFile": "${workspaceFolder}/.env",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            },
            "justMyCode": true,
            "console": "integratedTerminal"
        }
    ]
}
```
