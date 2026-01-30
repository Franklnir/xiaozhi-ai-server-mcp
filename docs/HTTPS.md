# HTTPS Setup (Nginx or Caddy)

This project runs behind a reverse proxy to provide HTTPS.

## Option A: Nginx

1) Put your SSL certs on the server (Let's Encrypt or your provider).
2) Use the example config in `deploy/nginx.conf.example`.
3) Replace:
   - `your-domain.example`
   - certificate paths
4) Reload Nginx.

## Option B: Caddy

1) Install Caddy (auto HTTPS).
2) Use `deploy/Caddyfile.example` and replace the domain.
3) Run `caddy run` or install as a service.

## App settings

In `.env`, set:
- `COOKIE_SECURE=true`
- `CORS_ALLOW_ORIGINS=https://your-domain.example`
- `CORS_ALLOW_CREDENTIALS=true`
- `ADMIN_ALLOWED_IPS=your.admin.ip`

Restart the app after changes.

