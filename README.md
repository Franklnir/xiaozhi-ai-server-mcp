# SciG Mode MCP (FastAPI + MySQL + Xiaozhi MCP)

Tujuan: ubah mode dan introduction via dashboard, lalu Xiaozhi auto-sync lewat MCP.

## Local Run
```bash
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
venv\Scripts\python.exe start.py
```

Catatan local:
- Isi `DATABASE_URL` di `.env` dengan MySQL yang aktif.
- `COOKIE_SECURE=false` untuk akses `http://127.0.0.1:8000` / `http://localhost:8000`.

Dashboard:
- http://127.0.0.1:8000

## Deploy ke Railway
File deploy sudah disiapkan:
- `railway.json` (start command + healthcheck)
- `Procfile` (fallback web process)
- `start.py` (bind ke `0.0.0.0:$PORT`)

### Variable wajib di Railway (service app)
- `DATABASE_URL=${{MySQL.MYSQL_URL}}` (boleh pakai private/public URL)
- `SECRET_KEY=<random-panjang>`
- `APP_DEBUG=false`
- (opsional) `APP_ENV=production`

Catatan:
- Jangan isi `DATABASE_URL` dengan placeholder mentah yang belum ter-resolve.
- Healthcheck endpoint: `/healthz`
- Database schema akan auto-create saat startup.

## Deploy Docker di VPS

Untuk production VPS yang aman dan tetap terpisah dari mobile:

- Compose: `deploy/docker-compose.vps.yml`
- API env: `deploy/vps_api.env.example`
- Worker env: `deploy/vps_worker.env.example`
- HTTPS proxy: `deploy/Caddyfile.docker.example`
- Panduan: `docs/DOCKER_VPS.md`

Arsitektur Docker:
- `api` untuk FastAPI publik
- `worker` untuk MCP supervisor
- `caddy` untuk HTTPS dan reverse proxy

Jalankan dari repo backend:
```bash
cp deploy/vps_api.env.example deploy/vps_api.env
cp deploy/vps_worker.env.example deploy/vps_worker.env
docker compose -f deploy/docker-compose.vps.yml up -d --build
```
