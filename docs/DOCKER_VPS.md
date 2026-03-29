# Docker VPS Deploy

Repo backend ini bisa dijalankan terpisah dari mobile memakai Docker Compose di VPS.

## Arsitektur

1. `api`
- FastAPI publik
- HTTPS lewat Caddy
- `MCP_SUPERVISOR_ENABLED=false`
- `CLEANUP_ENABLED=false`

2. `worker`
- proses MCP supervisor
- tidak diekspos ke internet
- pakai database yang sama dengan `api`

3. `caddy`
- terminasi HTTPS
- reverse proxy ke `api:8000`

## File yang dipakai

- Compose: `deploy/docker-compose.vps.yml`
- Env API: `deploy/vps_api.env`
- Env Worker: `deploy/vps_worker.env`
- Caddy: `deploy/Caddyfile.docker.example`

## Langkah deploy

1. Siapkan DNS lebih dulu.
- Buat record `A` atau `AAAA` untuk domain API, misalnya `api.domainkamu.com`, ke IP VPS.
- Tunggu propagasi sampai `ping api.domainkamu.com` mengarah ke IP VPS.

2. Siapkan VPS.
```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin git curl
sudo systemctl enable --now docker
```

3. Clone repo backend di VPS.
```bash
git clone https://github.com/Franklnir/xiaozhi-ai-server-mcp.git
cd xiaozhi-ai-server-mcp
```

4. Siapkan database MySQL / MariaDB.
- Stack Docker ini mengasumsikan database sudah ada.
- Pastikan host database bisa diakses dari container `api` dan `worker`.
- Buat database `scig_chat` dan user production sendiri, jangan pakai user development.

5. Siapkan file env nyata.
- Copy dari file contoh:
```bash
cp deploy/vps_api.env.example deploy/vps_api.env
cp deploy/vps_worker.env.example deploy/vps_worker.env
```
- Isi minimal nilai berikut di kedua file:
  - `DATABASE_URL`
  - `SECRET_KEY`
  - `ADMIN_MASTER_CODE_HASH`
  - `TOKEN_ENC_KEY`
- Khusus `deploy/vps_api.env`, isi juga:
  - `ALLOWED_HOSTS`
  - `CSRF_TRUSTED_ORIGINS`
  - `CORS_ALLOW_ORIGINS`
  - `ADMIN_ALLOWED_IPS`

6. Ganti domain pada `deploy/Caddyfile.docker.example`.
- Ubah `your-domain.example` menjadi domain API kamu, misalnya `api.domainkamu.com`.

7. Buka firewall untuk HTTP/HTTPS.
```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw reload
```

8. Jalankan stack Docker.
```bash
docker compose -f deploy/docker-compose.vps.yml up -d --build
```

9. Verifikasi health dan HTTPS.
```bash
docker compose -f deploy/docker-compose.vps.yml ps
docker compose -f deploy/docker-compose.vps.yml logs api --tail=100
docker compose -f deploy/docker-compose.vps.yml logs caddy --tail=100
curl -I https://api.domainkamu.com/healthz
```

10. Arahkan mobile dan web ke domain production.
- Mobile gunakan `https://api.domainkamu.com`
- Web/admin juga buka lewat domain yang sama
- Jangan lagi pakai `http://IP-LAN:8000` untuk production

## Generate secret

`SECRET_KEY`:
```bash
python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
```

`TOKEN_ENC_KEY`:
```bash
python3 - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
```

`ADMIN_MASTER_CODE_HASH`:
```bash
python3 - <<'PY'
import hashlib
code = "ganti_dengan_code_admin"
print(hashlib.sha256(code.encode()).hexdigest())
PY
```

## Catatan penting

- Mobile tetap repo terpisah dan hanya diarahkan ke `https://your-domain.example`
- Jangan expose port `8000` langsung ke internet kalau sudah memakai Caddy
- Untuk VPS production, gunakan `APP_ENV=production` dan `FORCE_HTTPS=true`
- Worker tidak butuh port publik
- `deploy/vps_api.env` dan `deploy/vps_worker.env` sudah di-ignore Git agar aman dari commit tidak sengaja
