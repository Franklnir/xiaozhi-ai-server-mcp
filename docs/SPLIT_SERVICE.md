# Split API and MCP Worker

Use two processes (or two servers):

1) API/Web
- Runs FastAPI only
- MCP supervisor disabled
- Cleanup disabled
- Deploy dari repo backend saja

2) MCP Worker
- Runs WebSocket workers
- MCP supervisor enabled
- Cleanup enabled (single node)
- Shares database yang sama dengan API

3) Mobile App
- Dibuild dari repo `mobile/`
- Tidak dipasang di VPS
- Cukup diarahkan ke backend HTTPS

## API Node .env

```
MCP_SUPERVISOR_ENABLED=false
CLEANUP_ENABLED=false
```

## Worker Node .env

```
MCP_SUPERVISOR_ENABLED=true
CLEANUP_ENABLED=true
```

## Start commands

API:
```
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --proxy-headers
```

Worker:
```
python worker.py
```
