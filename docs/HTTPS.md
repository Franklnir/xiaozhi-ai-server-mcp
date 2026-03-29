# HTTPS Setup (Backend tetap terpisah dari mobile)

Deploy backend FastAPI di VPS. Repo mobile tetap build APK secara terpisah dan cukup diarahkan ke URL HTTPS backend.

## Arsitektur singkat

1) Backend/API
- Repo: `scig_mode_mcp_fastapi`
- Jalan di VPS dengan reverse proxy HTTPS
- Bisa dipisah dari worker MCP

2) Mobile
- Repo: `mobile/`
- Build APK terpisah
- User mengisi `https://your-domain.example` di app

## Environment production minimal

Set di `.env` backend:

- `APP_ENV=production`
- `APP_DEBUG=false`
- `COOKIE_SECURE=true`
- `COOKIE_SAMESITE=lax`
- `FORCE_HTTPS=true`
- `TRUST_PROXY_HEADERS=true`
- `ALLOWED_HOSTS=your-domain.example,127.0.0.1,localhost`
- `CSRF_TRUSTED_ORIGINS=https://your-domain.example`
- `CORS_ALLOW_ORIGINS=https://your-domain.example`
- `CORS_ALLOW_CREDENTIALS=true`
- `SECRET_KEY=<random panjang>`
- `TOKEN_ENC_KEY=<fernet key>`
- `PASSWORD_MIN_LENGTH=10`

Gunakan contoh siap pakai:
- `deploy/vps_api.env.example`
- `deploy/vps_worker.env.example`

## Option A: Nginx

1) Pasang sertifikat SSL (Let's Encrypt / provider Anda).
2) Pakai `deploy/nginx.conf.example`.
3) Ganti:
   - `your-domain.example`
   - path certificate
4) Reload Nginx.

## Option B: Caddy

1) Install Caddy.
2) Pakai `deploy/Caddyfile.example`.
3) Ganti domain.
4) Jalankan Caddy sebagai service.

## Option C: Docker Compose + Caddy

Kalau ingin deploy backend di VPS dengan Docker:

1) Siapkan:
   - `deploy/docker-compose.vps.yml`
   - `deploy/Caddyfile.docker.example`
   - `deploy/vps_api.env.example`
   - `deploy/vps_worker.env.example`
2) Copy file env contoh menjadi file nyata.
3) Isi secret dan domain production.
4) Jalankan:
```bash
docker compose -f deploy/docker-compose.vps.yml up -d --build
```
5) Detail langkah ada di `docs/DOCKER_VPS.md`.

## systemd API

Gunakan `deploy/systemd-api.service.example`.

Catatan:
- service API sekarang sudah disiapkan dengan `--proxy-headers`
- worker MCP pakai `deploy/systemd-worker.service.example`
- untuk split deployment, aktifkan worker hanya di node worker

## Verifikasi setelah deploy

1) Cek HTTPS:
```bash
curl -I https://your-domain.example/healthz
```

2) Cek host policy:
```bash
curl -H 'Host: your-domain.example' https://127.0.0.1/healthz -k
```

3) Login mobile harus memakai:
```text
https://your-domain.example
```

Untuk build release Android production, hindari HTTP. Cleartext hanya cocok untuk debug/LAN testing.

