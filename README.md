# SciG Mode MCP (FastAPI + MySQL + Xiaozhi MCP)

Tujuan: **Ubah mode & introduction di dashboard -> Xiaozhi auto-sync via MCP dan pakai prompt terbaru.**

## 1) Setup MySQL
Buat DB `scig_chat` (atau sesuaikan), lalu update `DATABASE_URL` di `.env`.

Saat startup, server akan auto-create tabel:
- `modes`
- `active_mode`

Dan auto-seed mode default: `psikolog`, `sains`, `sejarah`, `umum`.

## 2) Install & Run
```bash
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m uvicorn main:app --reload
```

Buka dashboard:
- http://127.0.0.1:8000

## 3) Isi "Role Introduction" di Xiaozhi
Copy teks dari dashboard (panel kanan) ke kolom **Role Introduction** di Xiaozhi.

Prinsipnya:
- Xiaozhi akan memanggil tool MCP `get_active_mode`
- lalu memakai `introduction` sebagai persona/role.
