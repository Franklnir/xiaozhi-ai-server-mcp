# main.py
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import bcrypt
import jwt  # PyJWT
import uvicorn
import websockets
from fastapi import (
    FastAPI,
    Query,
    WebSocket,
    WebSocketDisconnect,
    Request,
    Form,
    Depends,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# =========================================================
# 0) Device ID Normalization + Bucketing (ONLY 4 PSID + default)
# =========================================================

_DEVICE_ID_MAXLEN = 64
_DEVICE_ID_BAD_CHARS = re.compile(r"[^a-z0-9_-]+")

PSID_LIST = ["psid_1", "psid_2", "psid_3", "psid_4"]
DEFAULT_BUCKET = "default"
ALLOWED_EFFECTIVE_IDS = set(PSID_LIST + [DEFAULT_BUCKET])


def normalize_device_id(device_id: str) -> str:
    did = (device_id or "").strip().lower()
    if not did:
        return DEFAULT_BUCKET
    did = re.sub(r"\s+", "_", did)
    did = _DEVICE_ID_BAD_CHARS.sub("_", did)
    did = re.sub(r"_+", "_", did).strip("_")
    did = did[:_DEVICE_ID_MAXLEN]
    return did or DEFAULT_BUCKET


def canonical_device_bucket(device_id: str) -> str:
    did = normalize_device_id(device_id)
    return did if did in ALLOWED_EFFECTIVE_IDS else DEFAULT_BUCKET


# =========================================================
# 1) SETTINGS
# =========================================================

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = Field(default="SciG Mode MCP", validation_alias=AliasChoices("APP_NAME"))
    app_env: str = Field(default="development", validation_alias=AliasChoices("APP_ENV"))
    app_debug: bool = Field(default=True, validation_alias=AliasChoices("APP_DEBUG"))
    app_host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("APP_HOST"))
    app_port: int = Field(default=8000, validation_alias=AliasChoices("APP_PORT"))

    # Security
    secret_key: str = Field(default="CHANGE_ME_IN_PRODUCTION", validation_alias=AliasChoices("SECRET_KEY"))
    token_expire_minutes: int = Field(default=1440, validation_alias=AliasChoices("TOKEN_EXPIRE_MINUTES"))

    # Admin master code
    admin_master_code: str = Field(default="26122003", validation_alias=AliasChoices("ADMIN_MASTER_CODE"))

    # DB
    database_url: str = Field(..., validation_alias=AliasChoices("DATABASE_URL", "MYSQL_URL"))

    # WS keepalive/reconnect
    mcp_ws_ping_interval: int = Field(default=15, validation_alias=AliasChoices("MCP_WS_PING_INTERVAL"))
    mcp_ws_ping_timeout: int = Field(default=20, validation_alias=AliasChoices("MCP_WS_PING_TIMEOUT"))
    mcp_open_timeout: int = Field(default=20, validation_alias=AliasChoices("MCP_OPEN_TIMEOUT"))
    mcp_close_timeout: int = Field(default=5, validation_alias=AliasChoices("MCP_CLOSE_TIMEOUT"))
    mcp_reconnect_delay: int = Field(default=5, validation_alias=AliasChoices("MCP_RECONNECT_DELAY"))
    mcp_max_reconnect_delay: int = Field(default=60, validation_alias=AliasChoices("MCP_MAX_RECONNECT_DELAY"))
    mcp_ws_subprotocols: str = Field(default="", validation_alias=AliasChoices("MCP_WS_SUBPROTOCOLS"))

    # Supervisor scan interval (DB -> auto connect)
    mcp_scan_interval_sec: int = Field(default=8, validation_alias=AliasChoices("MCP_SCAN_INTERVAL_SEC"))

    # Logging
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL"))
    log_tool_calls: bool = Field(default=True, validation_alias=AliasChoices("LOG_TOOL_CALLS"))
    log_system_prompt_on_get_prompt: bool = Field(default=True, validation_alias=AliasChoices("LOG_SYSTEM_PROMPT_ON_GET_PROMPT"))
    system_prompt_log_max_chars: int = Field(default=1200, validation_alias=AliasChoices("SYSTEM_PROMPT_LOG_MAX_CHARS"))

    # LLM provider for dashboard testing
    llm_provider: str = Field(default="mock", validation_alias=AliasChoices("LLM_PROVIDER"))  # mock | ollama | openai

    # Ollama
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", validation_alias=AliasChoices("OLLAMA_BASE_URL"))
    ollama_model: str = Field(default="llama3.2:1b", validation_alias=AliasChoices("OLLAMA_MODEL"))

    # OpenAI (optional)
    openai_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("OPENAI_API_KEY"))
    openai_model: str = Field(default="gpt-4o-mini", validation_alias=AliasChoices("OPENAI_MODEL"))

    # Device lock (opsional)
    force_device_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("FORCE_DEVICE_ID"))

    # Cleanup
    cleanup_enabled: bool = Field(default=True, validation_alias=AliasChoices("CLEANUP_ENABLED"))
    cleanup_days: int = Field(default=30, validation_alias=AliasChoices("CLEANUP_DAYS"))
    cleanup_interval_minutes: int = Field(default=360, validation_alias=AliasChoices("CLEANUP_INTERVAL_MINUTES"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def effective_device_id(incoming_device_id: str) -> str:
    forced = (settings.force_device_id or "").strip()
    if forced:
        return canonical_device_bucket(forced)
    return canonical_device_bucket(incoming_device_id)


def _mask_ws_url(url: str) -> str:
    return re.sub(r"(token=)([^&]+)", r"\1***", url)


logging.basicConfig(
    level=getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scig_mcp")


# =========================================================
# 2) SECURITY & AUTHENTICATION
# =========================================================

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=int(settings.token_expire_minutes)))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm="HS256")


def verify_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(request: Request) -> Dict[str, Any]:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = verify_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db_get_user_by_username(engine, username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def get_current_admin(request: Request) -> Dict[str, Any]:
    user = await get_current_user(request)
    if (user.get("role") or "") != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return user


def generate_code() -> str:
    # 8 digit
    return "".join(random.choices("0123456789", k=8))


# =========================================================
# 3) DATABASE: schema + helpers (NO DROP TABLE)
# =========================================================

LANGUAGES = [
    "Indonesia", "Inggris", "Arab", "Rusia",
    "Mandarin (Simplified)", "Mandarin (Traditional)",
    "Jepang", "Korea", "Jawa", "Sunda",
    "Spanyol", "Prancis", "Jerman", "Belanda",
    "Portugis", "Italia", "Turki", "Hindi",
    "Bengali", "Vietnam", "Thailand", "Filipina (Tagalog)",
]

DEFAULT_MODES = [
    {
        "name": "f2f_auto",
        "title": "🗣️ Face-to-Face (Auto {source} ↔ {target})",
        "introduction": (
            "PERAN: Interpreter Lisan Simultan (Dua Arah).\n\n"
            "TUGAS:\n"
            "1) Jika input dalam bahasa '{source}', terjemahkan ke '{target}'.\n"
            "2) Jika input dalam bahasa '{target}', terjemahkan ke '{source}'.\n\n"
            "ATURAN KRUSIAL:\n"
            "- HANYA keluarkan hasil terjemahan.\n"
            "- JANGAN menjawab isi chat.\n"
            "- JANGAN menambah penjelasan.\n"
            "- Jaga singkat, cepat, akurat.\n"
        ),
    },
    {
        "name": "translation_text",
        "title": "🌐 Terjemahan Teks ({source} → {target})",
        "introduction": (
            "PERAN: Penerjemah Dokumen Profesional.\n\n"
            "TUGAS: Terjemahkan teks input dari bahasa '{source}' ke bahasa '{target}'.\n\n"
            "ATURAN KRUSIAL:\n"
            "- HANYA berikan hasil terjemahan.\n"
            "- Tanpa pembuka/penutup.\n"
            "- Jangan beri komentar.\n"
        ),
    },
    {
        "name": "normal",
        "title": "🙂 Normal",
        "introduction": (
            "PERAN: Asisten AI umum.\n"
            "GAYA: Jelas, rapi, solutif.\n"
            "ATURAN:\n"
            "- Jawab sesuai pertanyaan.\n"
            "- Kalau butuh, beri langkah-langkah.\n"
            "- Kalau user kurang jelas, ajukan pertanyaan singkat yang tepat.\n"
            "- Jawaban singkat, cocok untuk dibacakan.\n"
        ),
    },
    {
        "name": "santai",
        "title": "😎 Santai",
        "introduction": (
            "PERAN: Teman ngobrol yang membantu.\n"
            "GAYA: Santai, ringan, tetap sopan, bahasa Indonesia sehari-hari.\n"
            "ATURAN:\n"
            "- Tetap jawab akurat.\n"
            "- Boleh pakai gaya casual dan emoji seperlunya.\n"
            "- Jangan bertele-tele.\n"
        ),
    },
    {
        "name": "anak_anak",
        "title": "🧸 Mode Anak-anak",
        "introduction": (
            "PERAN: Teman belajar anak.\n"
            "GAYA: Kalimat pendek, sederhana, ceria.\n"
            "ATURAN:\n"
            "- Pakai kata-kata mudah.\n"
            "- Jelasin pelan-pelan.\n"
            "- Hindari topik dewasa/keras.\n"
            "- Boleh pakai emoji lucu seperlunya.\n"
        ),
    },
    {
        "name": "coding_expert",
        "title": "💻 Senior Programmer",
        "introduction": (
            "PERAN: Senior Software Engineer.\n"
            "GAYA: Clean code, best practice, efisien.\n"
            "ATURAN:\n"
            "- Jawab langkah jelas.\n"
            "- Kalau perlu, kasih contoh kode yang siap pakai.\n"
            "- Sebutkan bagian mana yang harus di-replace/copy.\n"
        ),
    },
    {
        "name": "sains",
        "title": "🔬 Mode Sains",
        "introduction": (
            "PERAN: Profesor Sains.\n"
            "GAYA: Objektif, analitis, berbasis konsep ilmiah.\n"
            "ATURAN:\n"
            "- Jelaskan dengan logika sains.\n"
            "- Kalau ada rumus, pakai LaTeX.\n"
            "- Bedakan fakta vs asumsi.\n"
        ),
    },
    {
        "name": "sejarah",
        "title": "🏛️ Mode Sejarah",
        "introduction": (
            "PERAN: Sejarawan akademis.\n"
            "GAYA: Naratif, kronologis, konteks jelas.\n"
            "ATURAN:\n"
            "- Sebutkan waktu/era (tahun/abad) jika relevan.\n"
            "- Jelaskan sebab-akibat dan dampaknya.\n"
            "- Kalau ragu, bilang ragu.\n"
        ),
    },
    {
        "name": "matematika_dasar",
        "title": "🧮 Matematika Dasar",
        "introduction": (
            "PERAN: Tutor matematika.\n"
            "GAYA: Step-by-step, rapi.\n"
            "ATURAN:\n"
            "- Tulis langkah per langkah.\n"
            "- Tunjukkan perhitungan jelas.\n"
            "- Beri jawaban akhir yang tegas.\n"
        ),
    },
    {
        "name": "curhat",
        "title": "💬 Mode Curhat",
        "introduction": (
            "PERAN: Teman curhat yang suportif.\n"
            "GAYA: Empatik, hangat, tidak menghakimi.\n"
            "ATURAN:\n"
            "- Validasi perasaan.\n"
            "- Ajukan pertanyaan lembut untuk memahami.\n"
            "- Beri saran ringan yang praktis (CBT ringan).\n"
            "- Jika user menyebut niat menyakiti diri/krisis: sarankan cari bantuan profesional/darurat.\n"
        ),
    },
    {
        "name": "psikolog",
        "title": "❤️ Psikolog",
        "introduction": (
            "PERAN: Konselor kesehatan mental.\n"
            "GAYA: Empatik, hangat, validasi perasaan.\n"
            "ATURAN:\n"
            "- Jangan menghakimi.\n"
            "- Gunakan CBT ringan.\n"
            "- Ajak user menenangkan diri.\n"
        ),
    },
]

DDL_SCRIPT = """
CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(64) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role ENUM('user','admin') NOT NULL DEFAULT 'user',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS auth_codes (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(8) NOT NULL UNIQUE,
  token TEXT NOT NULL,
  used_by INT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (used_by) REFERENCES users(id) ON DELETE SET NULL,
  INDEX idx_auth_used_by (used_by)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS modes (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(64) NOT NULL UNIQUE,
  title VARCHAR(128) NOT NULL,
  introduction TEXT NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS active_mode (
  id TINYINT PRIMARY KEY,
  mode_id INT NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (mode_id) REFERENCES modes(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS device_active_mode (
  device_id VARCHAR(64) PRIMARY KEY,
  mode_id INT NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (mode_id) REFERENCES modes(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS chat_threads (
  id INT AUTO_INCREMENT PRIMARY KEY,
  device_id VARCHAR(64) NOT NULL,
  title VARCHAR(160) NOT NULL DEFAULT 'New Chat',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_threads_device (device_id),
  INDEX idx_threads_updated (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS chat_messages (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  thread_id INT NOT NULL,
  role VARCHAR(16) NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE,
  INDEX idx_msgs_thread (thread_id),
  INDEX idx_msgs_created (created_at),
  INDEX idx_msgs_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS device_settings (
  device_id VARCHAR(64) PRIMARY KEY,
  source_lang VARCHAR(64) NOT NULL DEFAULT 'Indonesia',
  target_lang VARCHAR(64) NOT NULL DEFAULT 'Arab',
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS app_state (
  k VARCHAR(64) PRIMARY KEY,
  v TEXT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS device_routes (
  physical_device_id VARCHAR(64) PRIMARY KEY,
  psid VARCHAR(64) NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_routes_psid (psid)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS thread_summaries (
  thread_id INT PRIMARY KEY,
  device_id VARCHAR(64) NOT NULL,
  summary TEXT NOT NULL,
  last_message_id BIGINT NOT NULL DEFAULT 0,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE,
  INDEX idx_summary_device (device_id),
  INDEX idx_summary_lastmsg (last_message_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS mcp_conn_status (
  code_id INT PRIMARY KEY,
  is_connected TINYINT NOT NULL DEFAULT 0,
  last_ok_at TIMESTAMP NULL DEFAULT NULL,
  last_err_at TIMESTAMP NULL DEFAULT NULL,
  last_error TEXT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (code_id) REFERENCES auth_codes(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        for statement in DDL_SCRIPT.split(";"):
            st = statement.strip()
            if st:
                conn.execute(text(st))

        for m in DEFAULT_MODES:
            exists_id = conn.execute(
                text("SELECT id FROM modes WHERE name=:n LIMIT 1"),
                {"n": m["name"]},
            ).scalar()
            if not exists_id:
                conn.execute(
                    text("INSERT INTO modes (name, title, introduction) VALUES (:n,:t,:i)"),
                    {"n": m["name"], "t": m["title"], "i": m["introduction"]},
                )

        active = conn.execute(text("SELECT COUNT(*) FROM active_mode WHERE id=1")).scalar() or 0
        normal_id = conn.execute(text("SELECT id FROM modes WHERE name='normal' LIMIT 1")).scalar()
        if active == 0 and normal_id:
            conn.execute(text("INSERT INTO active_mode (id, mode_id) VALUES (1,:mid)"), {"mid": int(normal_id)})

        allowed_sql = "('default','psid_1','psid_2','psid_3','psid_4')"
        conn.execute(text(f"UPDATE chat_threads SET device_id='default' WHERE device_id NOT IN {allowed_sql}"))
        conn.execute(text(f"DELETE FROM device_active_mode WHERE device_id NOT IN {allowed_sql}"))
        conn.execute(text(f"DELETE FROM device_settings WHERE device_id NOT IN {allowed_sql}"))
        conn.execute(text(f"UPDATE device_routes SET psid='default' WHERE psid NOT IN {allowed_sql}"))

        base_devices = [DEFAULT_BUCKET] + PSID_LIST
        for d in base_devices:
            ex = conn.execute(text("SELECT COUNT(*) FROM device_settings WHERE device_id=:d"), {"d": d}).scalar() or 0
            if ex == 0:
                conn.execute(
                    text("INSERT INTO device_settings (device_id, source_lang, target_lang) VALUES (:d,'Indonesia','Arab')"),
                    {"d": d},
                )

    logger.info("✅ Database initialized (NO DROP).")


def db_set_app_state(engine: Engine, key: str, value: str) -> None:
    k = (key or "").strip()
    if not k:
        return
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO app_state (k, v) VALUES (:k, :v)
                ON DUPLICATE KEY UPDATE v=:v
            """),
            {"k": k, "v": (value or "")},
        )


def db_get_app_state(engine: Engine, key: str) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT k, v, updated_at FROM app_state WHERE k=:k LIMIT 1"),
            {"k": (key or "").strip()},
        ).mappings().fetchone()
        return dict(row) if row else None


# ---------- USER MANAGEMENT ----------

def db_create_user(engine: Engine, username: str, password: str, role: str = "user") -> Dict[str, Any]:
    username = (username or "").strip()
    if not username:
        raise ValueError("Username kosong")
    if len(password or "") < 6:
        raise ValueError("Password minimal 6 karakter")

    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM users WHERE username = :username"),
            {"username": username},
        ).scalar()
        if existing:
            raise ValueError("Username already exists")

        hashed = hash_password(password)
        conn.execute(
            text("INSERT INTO users (username, password_hash, role) VALUES (:username, :password, :role)"),
            {"username": username, "password": hashed, "role": role},
        )

        user_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
        row = conn.execute(
            text("SELECT id, username, role, created_at FROM users WHERE id = :id"),
            {"id": user_id},
        ).mappings().fetchone()
        return dict(row) if row else {}


def db_get_user_by_username(engine: Engine, username: str) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, username, password_hash, role FROM users WHERE username = :username"),
            {"username": (username or "").strip()},
        ).mappings().fetchone()
        return dict(row) if row else None


def db_get_user_by_id(engine: Engine, user_id: int) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, username, role FROM users WHERE id = :id"),
            {"id": int(user_id)},
        ).mappings().fetchone()
        return dict(row) if row else None


# ---------- AUTH CODES ----------

def db_create_mcp_code(engine: Engine, token_ws_url: str) -> Dict[str, Any]:
    token_ws_url = (token_ws_url or "").strip()
    if not token_ws_url:
        raise ValueError("Token / WS URL kosong")

    with engine.begin() as conn:
        code = generate_code()
        while conn.execute(text("SELECT id FROM auth_codes WHERE code=:c"), {"c": code}).scalar():
            code = generate_code()

        conn.execute(
            text("INSERT INTO auth_codes (code, token, used_by) VALUES (:c, :t, NULL)"),
            {"c": code, "t": token_ws_url},
        )
        cid = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
        row = conn.execute(
            text("SELECT id, code, token, used_by, created_at FROM auth_codes WHERE id=:id"),
            {"id": int(cid)},
        ).mappings().fetchone()
        return dict(row) if row else {}


def db_get_mcp_code(engine: Engine, code: str) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, code, token, used_by, created_at FROM auth_codes WHERE code=:c LIMIT 1"),
            {"c": (code or "").strip()},
        ).mappings().fetchone()
        return dict(row) if row else None


def db_claim_mcp_code(engine: Engine, code: str, user_id: int) -> Dict[str, Any]:
    code = (code or "").strip()
    if not (code.isdigit() and len(code) == 8):
        raise ValueError("Kode harus 8 digit")

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, code, token, used_by FROM auth_codes WHERE code=:c LIMIT 1"),
            {"c": code},
        ).mappings().fetchone()
        if not row:
            raise ValueError("Kode tidak ditemukan")

        owner = row["used_by"]
        if owner is None:
            conn.execute(
                text("UPDATE auth_codes SET used_by=:u WHERE id=:id"),
                {"u": int(user_id), "id": int(row["id"])},
            )
            owner = int(user_id)
        elif int(owner) != int(user_id):
            raise ValueError("Kode ini sudah dimiliki user lain")

        return {"ok": True, "code_id": int(row["id"]), "code": code, "owner_user_id": int(owner), "token": row["token"]}


def db_list_mcp_codes(engine: Engine) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT a.id, a.code, a.token, a.used_by, a.created_at,
                       u.username AS used_by_username,
                       s.is_connected, s.last_ok_at, s.last_err_at, s.last_error
                FROM auth_codes a
                LEFT JOIN users u ON a.used_by = u.id
                LEFT JOIN mcp_conn_status s ON s.code_id = a.id
                ORDER BY a.created_at DESC
            """)
        ).mappings().all()
        return [dict(r) for r in rows]


def db_list_mcp_codes_for_user(engine: Engine, user_id: int) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT a.id, a.code, a.token, a.used_by, a.created_at,
                       s.is_connected, s.last_ok_at, s.last_err_at, s.last_error
                FROM auth_codes a
                LEFT JOIN mcp_conn_status s ON s.code_id = a.id
                WHERE a.used_by = :u
                ORDER BY a.created_at DESC
            """),
            {"u": int(user_id)},
        ).mappings().all()
        return [dict(r) for r in rows]


def db_list_owned_tokens(engine: Engine) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT id, code, token, used_by
                FROM auth_codes
                WHERE used_by IS NOT NULL AND token IS NOT NULL AND token <> ''
                ORDER BY id ASC
            """)
        ).mappings().all()
        return [dict(r) for r in rows]


def db_upsert_conn_status_ok(engine: Engine, code_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO mcp_conn_status (code_id, is_connected, last_ok_at, last_error, last_err_at)
                VALUES (:id, 1, NOW(), NULL, NULL)
                ON DUPLICATE KEY UPDATE is_connected=1, last_ok_at=NOW(), last_error=NULL, last_err_at=NULL
            """),
            {"id": int(code_id)},
        )


def db_upsert_conn_status_err(engine: Engine, code_id: int, err: str) -> None:
    err = (err or "")[:2000]
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO mcp_conn_status (code_id, is_connected, last_err_at, last_error)
                VALUES (:id, 0, NOW(), :e)
                ON DUPLICATE KEY UPDATE is_connected=0, last_err_at=NOW(), last_error=:e
            """),
            {"id": int(code_id), "e": err},
        )


def db_validate_register_code(engine: Engine, code: str) -> Dict[str, Any]:
    code = (code or "").strip()

    if code == settings.admin_master_code:
        return {"ok": True, "kind": "admin", "available": True, "message": "Admin master code valid"}

    if not (code.isdigit() and len(code) == 8):
        return {"ok": False, "kind": "user", "available": False, "message": "Kode harus 8 digit angka"}

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, code, used_by, token FROM auth_codes WHERE code=:c LIMIT 1"),
            {"c": code},
        ).mappings().fetchone()

    if not row:
        return {"ok": False, "kind": "user", "available": False, "message": "Kode tidak ditemukan"}

    if row.get("used_by") is not None:
        return {"ok": True, "kind": "user", "available": False, "message": "Kode sudah di-claim user lain"}

    tok = (row.get("token") or "").strip()
    if not tok:
        return {"ok": True, "kind": "user", "available": False, "message": "Kode ada, tapi token kosong (admin belum isi token/WS URL)"}

    return {"ok": True, "kind": "user", "available": True, "message": "Kode valid dan tersedia"}


def db_unclaim_mcp_code(engine: Engine, code_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(text("UPDATE auth_codes SET used_by=NULL WHERE id=:id"), {"id": int(code_id)})
        conn.execute(text("DELETE FROM mcp_conn_status WHERE code_id=:id"), {"id": int(code_id)})


# ---------- routing physical device -> PSID ----------

def db_get_route_psid(engine: Engine, physical_device_id: str) -> Optional[str]:
    pid = normalize_device_id(physical_device_id)
    with engine.begin() as conn:
        v = conn.execute(
            text("SELECT psid FROM device_routes WHERE physical_device_id=:p LIMIT 1"),
            {"p": pid},
        ).scalar()
    return canonical_device_bucket(v) if v else None


def db_set_route_psid(engine: Engine, physical_device_id: str, psid: str) -> Dict[str, Any]:
    pid = normalize_device_id(physical_device_id)
    ps = canonical_device_bucket(psid)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO device_routes (physical_device_id, psid)
                VALUES (:p, :s)
                ON DUPLICATE KEY UPDATE psid=:s
            """),
            {"p": pid, "s": ps},
        )
    return {"physical_device_id": pid, "psid": ps}


def resolve_psid_for_incoming(engine: Engine, incoming_physical_device_id: str) -> Dict[str, str]:
    forced = (settings.force_device_id or "").strip()
    physical = normalize_device_id(incoming_physical_device_id)

    if forced:
        eff = canonical_device_bucket(forced)
        return {"physical_device_id": physical, "effective_device_id": eff}

    routed = db_get_route_psid(engine, physical)
    if routed:
        return {"physical_device_id": physical, "effective_device_id": canonical_device_bucket(routed)}

    st_act = db_get_app_state(engine, "active_psid")
    active = canonical_device_bucket((st_act.get("v") if st_act else "") or "")
    effective = active or DEFAULT_BUCKET

    try:
        db_set_route_psid(engine, physical, effective)
    except Exception:
        pass

    return {"physical_device_id": physical, "effective_device_id": effective}


# ---------- modes ----------

def db_get_active_mode(engine: Engine) -> Dict[str, Any]:
    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT m.id, m.name, m.title, m.introduction "
                "FROM active_mode a JOIN modes m ON a.mode_id = m.id WHERE a.id=1"
            )
        ).mappings().fetchone()
        if not row:
            return {"id": 0, "name": "error", "title": "Error", "introduction": "No active mode set."}
        return dict(row)


def db_get_active_mode_for_device(engine: Engine, device_id: str) -> Dict[str, Any]:
    did = effective_device_id(device_id)
    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT m.id, m.name, m.title, m.introduction "
                "FROM device_active_mode dam JOIN modes m ON dam.mode_id = m.id "
                "WHERE dam.device_id=:d LIMIT 1"
            ),
            {"d": did},
        ).mappings().fetchone()
        if row:
            return dict(row)
    return db_get_active_mode(engine)


def db_set_active_mode_for_device(engine: Engine, device_id: str, *, mode_id: Optional[int] = None, name: Optional[str] = None) -> Dict[str, Any]:
    did = effective_device_id(device_id)
    with engine.begin() as conn:
        target_id = mode_id
        if name:
            target_id = conn.execute(text("SELECT id FROM modes WHERE name=:n LIMIT 1"), {"n": name}).scalar()
        if not target_id:
            raise ValueError("Mode not found")

        conn.execute(
            text(
                "INSERT INTO device_active_mode (device_id, mode_id) VALUES (:d, :mid) "
                "ON DUPLICATE KEY UPDATE mode_id=:mid"
            ),
            {"d": did, "mid": int(target_id)},
        )
    return db_get_active_mode_for_device(engine, did)


def db_get_mode_by_id(engine: Engine, mode_id: int) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, name, title, introduction FROM modes WHERE id=:id"),
            {"id": int(mode_id)},
        ).mappings().fetchone()
        return dict(row) if row else None


def db_list_modes(engine: Engine) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, name, title, introduction FROM modes ORDER BY id ASC")).mappings().all()
        return [dict(r) for r in rows]


def db_upsert_mode(engine: Engine, name: str, title: str, introduction: str) -> Dict[str, Any]:
    name = normalize_device_id(name)
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT id FROM modes WHERE name=:n"), {"n": name}).scalar()
        if exists:
            conn.execute(
                text("UPDATE modes SET title=:t, introduction=:i WHERE id=:id"),
                {"t": title, "i": introduction, "id": int(exists)},
            )
        else:
            conn.execute(
                text("INSERT INTO modes (name, title, introduction) VALUES (:n, :t, :i)"),
                {"n": name, "t": title, "i": introduction},
            )
    return {"status": "success", "name": name}


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def safe_format(template: str, vars_dict: Dict[str, Any]) -> str:
    try:
        return template.format_map(SafeDict(vars_dict))
    except Exception:
        return template


def mode_needs_lang(mode_name: str) -> bool:
    return mode_name in ("f2f_auto", "translation_text")


def build_system_prompt(mode_name: str, mode_title: str, intro_text: str) -> str:
    return f"MODE: {mode_title} ({mode_name})\n\nINSTRUCTION:\n{intro_text}"


def build_role_introduction_for_xiaozhi() -> str:
    return (
        "INSTRUKSI WAJIB UNTUK XIAOZHI (JANGAN DIABAIKAN):\n"
        "1) SETIAP KALI ADA PESAN USER BARU, SEBELUM MENJAWAB WAJIB panggil tool MCP: get_prompt.\n"
        "2) Panggil get_prompt dengan arguments:\n"
        "   - device_id: id perangkat (contoh: esp32c3_2000)\n"
        "3) Ambil isi prompt dari hasil tool (content[0].text) dan jadikan SYSTEM PROMPT untuk percakapan ini.\n"
        "4) WAJIB patuhi instruksi pada SYSTEM PROMPT tersebut.\n"
        "5) Setelah kamu memberi jawaban ke user, panggil tool MCP: log_chat (device_id,user_text,assistant_text).\n"
        "6) Jika tool gagal/timeout: jawab singkat: 'MCP belum terhubung, cek koneksi server MCP.'\n"
        "7) Jangan menyebut proses tool ke user.\n"
        "8) Setiap pesan user baru HARUS mulai dari langkah 1 (panggil get_prompt lagi).\n"
    )


# ---------- device settings ----------

def db_get_device_settings(engine: Engine, device_id: str) -> Dict[str, str]:
    did = effective_device_id(device_id)
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT source_lang, target_lang FROM device_settings WHERE device_id=:d"),
            {"d": did},
        ).mappings().fetchone()
        if row:
            return {"source": row["source_lang"], "target": row["target_lang"]}

        conn.execute(
            text("INSERT INTO device_settings (device_id, source_lang, target_lang) VALUES (:d,'Indonesia','Arab')"),
            {"d": did},
        )
        return {"source": "Indonesia", "target": "Arab"}


def db_set_device_settings(engine: Engine, device_id: str, source: str, target: str) -> Dict[str, str]:
    did = effective_device_id(device_id)
    src = (source or "").strip() or "Indonesia"
    tgt = (target or "").strip() or "Arab"
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO device_settings (device_id, source_lang, target_lang)
                VALUES (:d, :s, :t)
                ON DUPLICATE KEY UPDATE source_lang=:s, target_lang=:t
            """),
            {"d": did, "s": src, "t": tgt},
        )
    return {"device_id": did, "source": src, "target": tgt}


# =========================================================
# 4) CHAT STORAGE (threads + messages)
# =========================================================

def db_list_threads(engine: Engine, device_id: str) -> List[Dict[str, Any]]:
    did = effective_device_id(device_id)
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT id, title, created_at, updated_at "
                "FROM chat_threads WHERE device_id=:d "
                "ORDER BY updated_at DESC, id DESC"
            ),
            {"d": did},
        ).mappings().all()
        return [dict(r) for r in rows]


def db_list_threads_all_grouped(engine: Engine, limit_per_device: int = 30) -> Dict[str, Any]:
    limit_per_device = max(1, min(int(limit_per_device), 200))

    with engine.begin() as conn:
        dev_rows = conn.execute(text("""
            SELECT device_id FROM device_active_mode
            UNION
            SELECT device_id FROM device_settings
            UNION
            SELECT device_id FROM chat_threads
        """)).scalars().all()

    base_devices = [DEFAULT_BUCKET] + PSID_LIST
    device_ids: List[str] = []
    seen = set()
    for d in base_devices + [canonical_device_bucket(x) for x in dev_rows if x]:
        if d and d not in seen:
            seen.add(d)
            device_ids.append(d)

    buckets: Dict[str, List[Dict[str, Any]]] = {d: [] for d in device_ids}

    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, device_id, title, created_at, updated_at
            FROM chat_threads
            ORDER BY updated_at DESC, id DESC
        """)).mappings().all()

    for r in rows:
        did = canonical_device_bucket(r["device_id"])
        if did not in buckets:
            buckets[did] = []
            device_ids.append(did)
        if len(buckets[did]) < limit_per_device:
            buckets[did].append(dict(r))

    devices = [{"device_id": d, "threads": buckets.get(d, [])} for d in device_ids]
    return {"devices": devices, "limit_per_device": limit_per_device, "timestamp": time.time()}


def db_create_thread(engine: Engine, device_id: str) -> Dict[str, Any]:
    did = effective_device_id(device_id)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO chat_threads (device_id, title) VALUES (:d, 'New Chat')"),
            {"d": did},
        )
        tid = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
        row = conn.execute(
            text("SELECT id, device_id, title, created_at, updated_at FROM chat_threads WHERE id=:id"),
            {"id": int(tid)},
        ).mappings().fetchone()
        return dict(row) if row else {"id": int(tid), "device_id": did, "title": "New Chat"}


def db_get_or_create_latest_thread_id(engine: Engine, device_id: str) -> int:
    did = effective_device_id(device_id)
    with engine.begin() as conn:
        tid = conn.execute(
            text("SELECT id FROM chat_threads WHERE device_id=:d ORDER BY updated_at DESC, id DESC LIMIT 1"),
            {"d": did},
        ).scalar()
        if tid:
            return int(tid)

        conn.execute(text("INSERT INTO chat_threads (device_id, title) VALUES (:d,'New Chat')"), {"d": did})
        tid2 = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
        return int(tid2)


def db_delete_thread(engine: Engine, device_id: str, thread_id: int) -> Dict[str, Any]:
    did = effective_device_id(device_id)
    with engine.begin() as conn:
        owned = conn.execute(
            text("SELECT id FROM chat_threads WHERE id=:id AND device_id=:d"),
            {"id": int(thread_id), "d": did},
        ).scalar()
        if not owned:
            raise ValueError("Thread not found")

        conn.execute(text("DELETE FROM chat_threads WHERE id=:id"), {"id": int(thread_id)})
    return {"status": "deleted", "thread_id": int(thread_id)}


def db_get_thread_owner_device(engine: Engine, thread_id: int) -> Optional[str]:
    with engine.begin() as conn:
        d = conn.execute(
            text("SELECT device_id FROM chat_threads WHERE id=:id LIMIT 1"),
            {"id": int(thread_id)},
        ).scalar()
    return canonical_device_bucket(d) if d else None


def db_get_messages_page(
    engine: Engine,
    device_id: str,
    thread_id: int,
    limit: int = 200,
    before_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    did = effective_device_id(device_id)
    lim = max(1, min(int(limit), 500))

    with engine.begin() as conn:
        owned = conn.execute(
            text("SELECT id FROM chat_threads WHERE id=:id AND device_id=:d"),
            {"id": int(thread_id), "d": did},
        ).scalar()
        if not owned:
            return []

        if before_id is not None:
            q = text(
                "SELECT id, role, content, created_at "
                "FROM chat_messages "
                "WHERE thread_id=:tid AND id < :before "
                "ORDER BY id DESC "
                f"LIMIT {lim}"
            )
            rows = conn.execute(q, {"tid": int(thread_id), "before": int(before_id)}).mappings().all()
        else:
            q = text(
                "SELECT id, role, content, created_at "
                "FROM chat_messages "
                "WHERE thread_id=:tid "
                "ORDER BY id DESC "
                f"LIMIT {lim}"
            )
            rows = conn.execute(q, {"tid": int(thread_id)}).mappings().all()

    rows = list(rows)[::-1]
    return [dict(r) for r in rows]


def db_add_message(engine: Engine, thread_id: int, role: str, content: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO chat_messages (thread_id, role, content) VALUES (:t, :r, :c)"),
            {"t": int(thread_id), "r": role, "c": content},
        )
        conn.execute(text("UPDATE chat_threads SET updated_at=CURRENT_TIMESTAMP WHERE id=:id"), {"id": int(thread_id)})


def db_try_set_thread_title_from_first_user(engine: Engine, thread_id: int, user_text: str) -> None:
    title = (user_text or "").strip().replace("\n", " ")
    title = title[:48] + ("…" if len(title) > 48 else "")
    if not title:
        return
    with engine.begin() as conn:
        cur = conn.execute(text("SELECT title FROM chat_threads WHERE id=:id"), {"id": int(thread_id)}).scalar()
        if cur and cur == "New Chat":
            conn.execute(text("UPDATE chat_threads SET title=:t WHERE id=:id"), {"t": title, "id": int(thread_id)})


# =========================================================
# 5) LLM CALLS (mock | ollama | openai)
# =========================================================

def _http_post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout_s: int = 60) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read().decode("utf-8", errors="replace")
        return json.loads(data)


def _extract_mode_name_from_system_prompt(system_prompt: str) -> str:
    try:
        first = (system_prompt.splitlines()[0] if system_prompt else "").strip()
        m = re.search(r"\(([^)]+)\)\s*$", first)
        return (m.group(1).strip() if m else "")
    except Exception:
        return ""


def llm_generate(provider: str, system_prompt: str, messages: List[Dict[str, str]]) -> str:
    provider = (provider or "mock").strip().lower()

    if provider == "mock":
        last = messages[-1]["content"] if messages else ""
        mode_name = _extract_mode_name_from_system_prompt(system_prompt)
        return f"(MOCK) [{mode_name or 'unknown'}] Aku menerima: {last}"

    if provider == "ollama":
        url = settings.ollama_base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": settings.ollama_model,
            "stream": False,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
        }
        try:
            out = _http_post_json(url, payload, timeout_s=120)
            msg = out.get("message", {}) or {}
            return (msg.get("content") or "").strip() or "(Ollama) empty response"
        except Exception as e:
            return f"(Ollama error) {e}"

    if provider == "openai":
        if not settings.openai_api_key:
            return "(OpenAI error) OPENAI_API_KEY belum diisi."
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": settings.openai_model,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
        }
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        try:
            out = _http_post_json(url, payload, headers=headers, timeout_s=120)
            choices = out.get("choices") or []
            if not choices:
                return "(OpenAI) no choices"
            msg = (choices[0].get("message") or {})
            return (msg.get("content") or "").strip() or "(OpenAI) empty response"
        except Exception as e:
            return f"(OpenAI error) {e}"

    return f"(LLM error) Unknown provider: {provider}"


# =========================================================
# 6) Rolling Summary (per thread)
# =========================================================

SUMMARY_KEEP_LAST = 30
SUMMARY_BATCH_LIMIT = 80
SUMMARY_MAX_CHARS = 1400


def db_get_thread_summary(engine: Engine, thread_id: int) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT thread_id, device_id, summary, last_message_id, updated_at FROM thread_summaries WHERE thread_id=:t LIMIT 1"),
            {"t": int(thread_id)},
        ).mappings().fetchone()
        return dict(row) if row else None


def db_upsert_thread_summary(engine: Engine, thread_id: int, device_id: str, summary: str, last_message_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO thread_summaries (thread_id, device_id, summary, last_message_id) "
                "VALUES (:t, :d, :s, :lm) "
                "ON DUPLICATE KEY UPDATE summary=:s, last_message_id=:lm"
            ),
            {"t": int(thread_id), "d": effective_device_id(device_id), "s": summary, "lm": int(last_message_id)},
        )


def _summary_generate(provider: str, prev_summary: str, new_dialogue: str) -> str:
    if (provider or "").strip().lower() == "mock":
        merged = (prev_summary + "\n" + new_dialogue).strip()
        return merged[-SUMMARY_MAX_CHARS:]

    sys_prompt = (
        "You are a rolling conversation summarizer.\n"
        "Update the existing summary with the NEW dialogue.\n"
        "Keep: key facts, preferences, decisions, tasks, constraints, names, numbers.\n"
        "Omit: filler, greetings.\n"
        f"Output ONLY the updated summary, max {SUMMARY_MAX_CHARS} chars."
    )
    user_msg = (
        f"EXISTING SUMMARY:\n{prev_summary.strip()}\n\n"
        f"NEW DIALOGUE:\n{new_dialogue.strip()}\n\n"
        "UPDATED SUMMARY:"
    )
    out = llm_generate(settings.llm_provider, sys_prompt, [{"role": "user", "content": user_msg}]).strip()
    if len(out) > SUMMARY_MAX_CHARS:
        out = out[:SUMMARY_MAX_CHARS].rstrip()
    return out


def maybe_roll_summary(engine: Engine, device_id: str, thread_id: int) -> None:
    st = db_get_thread_summary(engine, thread_id)
    prev = (st["summary"] if st else "").strip()
    last_done = int(st["last_message_id"]) if st else 0

    with engine.begin() as conn:
        ids = conn.execute(
            text(f"SELECT id FROM chat_messages WHERE thread_id=:t ORDER BY id DESC LIMIT {SUMMARY_KEEP_LAST}"),
            {"t": int(thread_id)},
        ).scalars().all()

        if not ids:
            return

        cutoff_id = min(ids)
        if cutoff_id <= last_done + 1:
            return

        rows = conn.execute(
            text(
                "SELECT id, role, content FROM chat_messages "
                "WHERE thread_id=:t AND id>:last AND id<:cutoff AND role IN ('user','assistant') "
                "ORDER BY id ASC "
                f"LIMIT {SUMMARY_BATCH_LIMIT}"
            ),
            {"t": int(thread_id), "last": int(last_done), "cutoff": int(cutoff_id)},
        ).mappings().all()

        if not rows:
            return

        chunk_lines = []
        last_id = last_done
        for r in rows:
            last_id = int(r["id"])
            role = r["role"]
            content = (r["content"] or "").strip()
            if not content:
                continue
            chunk_lines.append(f"{role.upper()}: {content}")

    new_dialogue = "\n".join(chunk_lines)
    if not new_dialogue.strip():
        return

    updated = _summary_generate(settings.llm_provider, prev, new_dialogue)
    db_upsert_thread_summary(engine, thread_id, device_id, updated, last_id)


# =========================================================
# 7) Auto Cleanup
# =========================================================

def db_cleanup_old_threads(engine: Engine, *, days: int = 30) -> Dict[str, Any]:
    days = max(7, min(int(days), 365))

    with engine.begin() as conn:
        count = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM chat_threads
                WHERE updated_at < DATE_SUB(NOW(), INTERVAL {days} DAY)
                """
            )
        ).scalar() or 0

        if count:
            conn.execute(
                text(
                    f"""
                    DELETE FROM chat_threads
                    WHERE updated_at < DATE_SUB(NOW(), INTERVAL {days} DAY)
                    """
                )
            )
    return {"deleted_threads": int(count), "days": days}


async def cleanup_worker(engine: Engine) -> None:
    if not settings.cleanup_enabled:
        logger.info("[CLEANUP] Disabled.")
        return

    interval = max(15, int(settings.cleanup_interval_minutes))
    days = int(settings.cleanup_days)

    logger.info(f"[CLEANUP] Worker started. interval={interval}min, days={days}")
    while True:
        try:
            res = db_cleanup_old_threads(engine, days=days)
            if res.get("deleted_threads"):
                logger.info(f"[CLEANUP] Deleted {res['deleted_threads']} thread(s) older than {res['days']} days.")
        except Exception as e:
            logger.warning(f"[CLEANUP] Error: {e}")
        await asyncio.sleep(interval * 60)


# =========================================================
# 8) MCP Tools + JSON-RPC helpers
# =========================================================

@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]


TOOLS: List[MCPTool] = [
    MCPTool(
        "get_prompt",
        "Get current active system prompt (rendered using device_settings). Args source/target ignored unless override=true.",
        {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "source": {"type": "string"},
                "target": {"type": "string"},
                "override": {"type": "boolean"},
                "log": {"type": "boolean"},
            },
        },
    ),
    MCPTool("list_modes", "List all available modes", {"type": "object", "properties": {}}),
    MCPTool(
        "set_active_mode",
        "Change active mode (per device_id)",
        {"type": "object", "properties": {"device_id": {"type": "string"}, "mode_id": {"type": "integer"}, "name": {"type": "string"}}},
    ),
    MCPTool(
        "upsert_mode",
        "Create/Edit mode",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}, "title": {"type": "string"}, "introduction": {"type": "string"}},
            "required": ["name", "title", "introduction"],
        },
    ),
    MCPTool("get_role_introduction", "Get text for Xiaozhi role config", {"type": "object", "properties": {}}),
    MCPTool(
        "log_chat",
        "Log Xiaozhi conversation into DB (thread + messages). Optional: thread_id",
        {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "thread_id": {"type": "integer"},
                "user_text": {"type": "string"},
                "assistant_text": {"type": "string"},
            },
            "required": ["device_id", "user_text", "assistant_text"],
        },
    ),
]


def _mcp_text_result(payload: Any) -> Dict[str, Any]:
    text_val = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text_val}]}


def _jsonrpc_ok(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_err(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def compute_prompt_for_device(engine: Engine, device_id: str) -> str:
    mode = db_get_active_mode_for_device(engine, device_id)
    saved = db_get_device_settings(engine, device_id)
    source = saved["source"]
    target = saved["target"]
    vars_dict = {"source": source, "target": target}

    title_raw = mode.get("title", "")
    intro_raw = mode.get("introduction", "")

    if mode_needs_lang(mode.get("name", "")):
        rendered_title = safe_format(title_raw, vars_dict)
        rendered_intro = safe_format(intro_raw, vars_dict)
    else:
        rendered_title = title_raw
        rendered_intro = intro_raw

    return build_system_prompt(
        mode_name=mode.get("name", ""),
        mode_title=rendered_title,
        intro_text=rendered_intro,
    )


def _validate_or_create_thread_for_device(engine: Engine, device_id: str, thread_id: Optional[int]) -> int:
    """
    FIX KRUSIAL:
    - Xiaozhi kadang kirim thread_id yang:
      a) tidak ada
      b) milik device/psid lain
    Solusi:
    - validasi owner thread dulu
    - kalau invalid, bikin thread baru untuk device_id
    """
    if thread_id is not None:
        owner = db_get_thread_owner_device(engine, int(thread_id))
        if owner and owner == effective_device_id(device_id):
            return int(thread_id)
    return db_get_or_create_latest_thread_id(engine, device_id)


async def _handle_tool_call(engine: Engine, name: str, args: Dict[str, Any], manager_ref=None) -> Dict[str, Any]:
    if name == "get_prompt":
        incoming = args.get("device_id") or DEFAULT_BUCKET
        r = resolve_psid_for_incoming(engine, incoming)
        physical_id = r["physical_device_id"]
        device_id = r["effective_device_id"]

        try:
            db_set_app_state(engine, "last_physical_device_id", physical_id)
            db_set_app_state(engine, "last_device_id", device_id)
            db_set_app_state(engine, "active_psid", device_id)
        except Exception:
            pass

        mode = db_get_active_mode_for_device(engine, device_id)
        saved = db_get_device_settings(engine, device_id)
        source = saved["source"]
        target = saved["target"]

        override = bool(args.get("override", False))
        if override:
            if args.get("source"):
                source = str(args["source"]).strip() or source
            if args.get("target"):
                target = str(args["target"]).strip() or target

        vars_dict = {"source": source, "target": target}
        title_raw = mode.get("title", "")
        intro_raw = mode.get("introduction", "")

        if mode_needs_lang(mode.get("name", "")):
            rendered_title = safe_format(title_raw, vars_dict)
            rendered_intro = safe_format(intro_raw, vars_dict)
        else:
            rendered_title = title_raw
            rendered_intro = intro_raw

        prompt = build_system_prompt(mode.get("name", ""), rendered_title, rendered_intro)

        try:
            if bool(args.get("log", True)) and settings.log_system_prompt_on_get_prompt:
                snip = prompt[: settings.system_prompt_log_max_chars]
                logger.info(f"[get_prompt] physical={physical_id} effective={device_id} mode={mode.get('name')} prompt_snip={snip!r}")
        except Exception:
            pass

        return _mcp_text_result(prompt)

    if name == "list_modes":
        return _mcp_text_result(db_list_modes(engine))

    if name == "set_active_mode":
        mid = args.get("mode_id")
        mid = int(mid) if mid is not None and str(mid).strip() != "" else None
        dev = args.get("device_id") or DEFAULT_BUCKET
        res = db_set_active_mode_for_device(engine, dev, mode_id=mid, name=args.get("name"))
        return _mcp_text_result(f"Active mode for {effective_device_id(dev)} changed to: {res['title']}")

    if name == "upsert_mode":
        res = db_upsert_mode(engine, args["name"], args["title"], args["introduction"])
        return _mcp_text_result(res)

    if name == "get_role_introduction":
        return _mcp_text_result(build_role_introduction_for_xiaozhi())

    if name == "log_chat":
        incoming = args.get("device_id") or DEFAULT_BUCKET
        r = resolve_psid_for_incoming(engine, incoming)
        physical_id = r["physical_device_id"]
        device_id = r["effective_device_id"]

        try:
            db_set_app_state(engine, "last_physical_device_id", physical_id)
            db_set_app_state(engine, "last_device_id", device_id)
            db_set_app_state(engine, "active_psid", device_id)
        except Exception:
            pass

        user_text = (args.get("user_text") or "").strip()
        assistant_text = (args.get("assistant_text") or "").strip()
        thread_id_arg = args.get("thread_id")

        if not user_text and not assistant_text:
            return _mcp_text_result({"ok": False, "error": "empty"})

        # FIX: validasi thread_id (kalau invalid, bikin thread baru)
        thread_id: int
        try:
            tid_in = None
            if thread_id_arg is not None and str(thread_id_arg).strip() != "":
                tid_in = int(thread_id_arg)
            thread_id = _validate_or_create_thread_for_device(engine, device_id, tid_in)
        except Exception:
            thread_id = db_get_or_create_latest_thread_id(engine, device_id)

        # Simpan pesan dengan fallback aman jika ada FK error
        try:
            if user_text:
                db_add_message(engine, thread_id, "user", user_text)
                db_try_set_thread_title_from_first_user(engine, thread_id, user_text)

                if settings.log_tool_calls:
                    try:
                        prompt_now = compute_prompt_for_device(engine, device_id)
                        db_add_message(engine, thread_id, "tool", f"get_prompt({{'device_id':'{physical_id}'}})")
                        db_add_message(engine, thread_id, "tool", prompt_now)
                    except Exception:
                        pass

            if assistant_text:
                db_add_message(engine, thread_id, "assistant", assistant_text)

        except Exception:
            # kalau thread_id ternyata invalid di DB (race/cleanup), bikin baru dan ulang sekali
            thread_id = db_get_or_create_latest_thread_id(engine, device_id)
            if user_text:
                db_add_message(engine, thread_id, "user", user_text)
                db_try_set_thread_title_from_first_user(engine, thread_id, user_text)
            if assistant_text:
                db_add_message(engine, thread_id, "assistant", assistant_text)

        try:
            maybe_roll_summary(engine, device_id, thread_id)
        except Exception:
            pass

        try:
            if manager_ref is not None:
                await manager_ref.broadcast({
                    "type": "chat_message_sent",
                    "device_id": device_id,
                    "thread_id": thread_id,
                    "timestamp": time.time()
                })
        except Exception:
            pass

        return _mcp_text_result({"ok": True, "thread_id": thread_id})

    raise ValueError(f"Unknown tool: {name}")


def _unwrap_possible_wrapper(msg: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(msg, dict) and "payload" in msg and isinstance(msg["payload"], dict):
        inner = msg["payload"]
        if "jsonrpc" in inner and "method" in inner:
            return inner
    return msg


# =========================================================
# 9) MCP WORKER (PER TOKEN) + SUPERVISOR (DB SCAN)
# =========================================================

async def mcp_worker_for_token(
    *,
    code_id: int,
    code: str,
    ws_url: str,
    engine: Engine,
    manager_ref=None,
) -> None:
    url = (ws_url or "").strip()
    if not url:
        return

    delay = max(1, int(settings.mcp_reconnect_delay))
    max_delay = max(delay, int(settings.mcp_max_reconnect_delay))
    subprotocols = [p.strip() for p in (settings.mcp_ws_subprotocols or "").split(",") if p.strip()] or None

    logger.info(f"[MCP:{code}] Worker started. Target: {_mask_ws_url(url)}")

    while True:
        try:
            logger.info(f"[MCP:{code}] Connecting -> {_mask_ws_url(url)}")
            async with websockets.connect(
                url,
                ping_interval=settings.mcp_ws_ping_interval,
                ping_timeout=settings.mcp_ws_ping_timeout,
                open_timeout=settings.mcp_open_timeout,
                close_timeout=settings.mcp_close_timeout,
                max_size=2 * 1024 * 1024,
                max_queue=64,
                subprotocols=subprotocols,
            ) as ws:
                logger.info(f"[MCP:{code}] Connected ✅")
                db_upsert_conn_status_ok(engine, code_id)
                delay = max(1, int(settings.mcp_reconnect_delay))

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    msg = _unwrap_possible_wrapper(msg)
                    method = msg.get("method")
                    req_id = msg.get("id")
                    params = msg.get("params") or {}

                    if not method:
                        continue

                    if method == "ping":
                        if req_id is not None:
                            await ws.send(json.dumps(_jsonrpc_ok(req_id, {"pong": int(time.time())})))
                        continue

                    if method == "initialize":
                        result = {
                            "protocolVersion": "2024-11-05",
                            "serverInfo": {"name": settings.app_name, "version": "12.1-thread-safe-log_chat"},
                            "capabilities": {"tools": {"listChanged": True}},
                        }
                        if req_id is not None:
                            await ws.send(json.dumps(_jsonrpc_ok(req_id, result)))
                        continue

                    if method == "tools/list":
                        tool_list = [{"name": t.name, "description": t.description, "inputSchema": t.input_schema} for t in TOOLS]
                        if req_id is not None:
                            await ws.send(json.dumps(_jsonrpc_ok(req_id, {"tools": tool_list})))
                        continue

                    if method == "tools/call":
                        try:
                            tool_name = params.get("name")
                            tool_args = params.get("arguments") or {}
                            if not tool_name:
                                raise ValueError("Missing params.name")
                            result = await _handle_tool_call(engine, tool_name, tool_args, manager_ref=manager_ref)
                            if req_id is not None:
                                await ws.send(json.dumps(_jsonrpc_ok(req_id, result)))
                        except Exception as e:
                            if req_id is not None:
                                await ws.send(json.dumps(_jsonrpc_err(req_id, -32000, str(e))))
                        continue

                    if req_id is not None:
                        await ws.send(json.dumps(_jsonrpc_err(req_id, -32601, f"Method not found: {method}")))

        except websockets.exceptions.ConnectionClosed as e:
            code_close = getattr(e, "code", None)
            reason = getattr(e, "reason", "")
            msg = f"Disconnected (code={code_close}, reason={reason!r})"
            logger.warning(f"[MCP:{code}] {msg}. Reconnect in {delay}s ...")
            db_upsert_conn_status_err(engine, code_id, msg)
        except Exception as e:
            msg = f"Disconnected/error: {e}"
            logger.warning(f"[MCP:{code}] {msg}. Reconnect in {delay}s ...")
            db_upsert_conn_status_err(engine, code_id, msg)

        await asyncio.sleep(delay + random.uniform(0, 0.5))
        delay = min(max_delay, max(delay * 2, 1))


class MCPSupervisor:
    def __init__(self):
        self.tasks: Dict[int, asyncio.Task] = {}
        self.running = True

    async def run(self, engine: Engine, manager_ref=None) -> None:
        interval = max(3, int(settings.mcp_scan_interval_sec))
        logger.info(f"[MCP_SUP] started. scan_interval={interval}s")

        while self.running:
            try:
                rows = db_list_owned_tokens(engine)
                active_ids = set()
                for r in rows:
                    cid = int(r["id"])
                    active_ids.add(cid)

                    if cid not in self.tasks or self.tasks[cid].done():
                        ws_url = (r.get("token") or "").strip()
                        code = (r.get("code") or str(cid)).strip()
                        if not ws_url:
                            continue
                        t = asyncio.create_task(
                            mcp_worker_for_token(
                                code_id=cid,
                                code=code,
                                ws_url=ws_url,
                                engine=engine,
                                manager_ref=manager_ref,
                            )
                        )
                        self.tasks[cid] = t
                        logger.info(f"[MCP_SUP] worker spawned for code={code} id={cid}")

                for cid in list(self.tasks.keys()):
                    if cid not in active_ids:
                        task = self.tasks.pop(cid, None)
                        if task:
                            task.cancel()
                            logger.info(f"[MCP_SUP] worker cancelled for code_id={cid}")

            except Exception as e:
                logger.warning(f"[MCP_SUP] scan error: {e}")

            await asyncio.sleep(interval)

    async def shutdown(self):
        self.running = False
        for _, t in list(self.tasks.items()):
            t.cancel()
        for _, t in list(self.tasks.items()):
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self.tasks.clear()


mcp_supervisor = MCPSupervisor()


# =========================================================
# 10) FASTAPI APP + TEMPLATE ENGINE
# =========================================================

engine = create_engine(settings.database_url, pool_pre_ping=True, pool_recycle=1800)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

jinja_env: Environment = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)

def render_template(name: str, **ctx) -> str:
    return jinja_env.get_template(name).render(**ctx)


class ModeUpsertIn(BaseModel):
    name: str
    title: str
    introduction: str


class SetActiveIn(BaseModel):
    device_id: str = DEFAULT_BUCKET
    mode_id: int


class RenderPromptIn(BaseModel):
    mode_id: Optional[int] = None
    mode_name: Optional[str] = None
    vars: Dict[str, Any] = Field(default_factory=dict)


class ChatNewIn(BaseModel):
    device_id: str


class ChatSendIn(BaseModel):
    device_id: str
    thread_id: int
    message: str
    mode_id: Optional[int] = None
    vars: Dict[str, Any] = Field(default_factory=dict)


class DeviceSettingsIn(BaseModel):
    device_id: str
    source: str
    target: str


class ClaimCodeIn(BaseModel):
    code: str


class AdminUnclaimIn(BaseModel):
    code_id: int


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    task_cleanup = asyncio.create_task(cleanup_worker(engine))
    task_mcp_sup = asyncio.create_task(mcp_supervisor.run(engine, manager_ref=manager))
    try:
        yield
    finally:
        for t in (task_cleanup, task_mcp_sup):
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        try:
            await mcp_supervisor.shutdown()
        except Exception:
            pass


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# =========================================================
# 11) AUTH PAGES
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return RedirectResponse(url="/login")


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(render_template("login.html", error=None))


@app.post("/login")
async def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    user = db_get_user_by_username(engine, username)
    if (not user) or (not verify_password(password, user["password_hash"])):
        return HTMLResponse(render_template("login.html", error="Username atau password salah"))

    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})

    redirect_url = "/admin" if user["role"] == "admin" else "/dashboard"
    response = RedirectResponse(url=redirect_url, status_code=303)

    secure_cookie = request.url.scheme == "https"
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=secure_cookie,
        max_age=int(settings.token_expire_minutes) * 60,
    )
    return response


@app.get("/register", response_class=HTMLResponse)
async def register_page():
    return HTMLResponse(render_template("register.html", error=None, admin_code=settings.admin_master_code))


@app.post("/register")
async def register_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    code: str = Form(...),
):
    errors = []

    if password != confirm_password:
        errors.append("Password dan konfirmasi password tidak sama")

    code = (code or "").strip()
    role = "user"

    v = db_validate_register_code(engine, code)
    if not v.get("ok"):
        errors.append(v.get("message") or "Kode tidak valid")
    else:
        if v.get("kind") == "admin":
            role = "admin"
        else:
            if not v.get("available"):
                errors.append(v.get("message") or "Kode sudah dipakai")

    if errors:
        return HTMLResponse(render_template("register.html", error="; ".join(errors), admin_code=settings.admin_master_code))

    try:
        user = db_create_user(engine, username, password, role)

        if role == "user":
            try:
                db_claim_mcp_code(engine, code, int(user["id"]))
            except Exception as e:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM users WHERE id=:id"), {"id": int(user["id"])})
                return HTMLResponse(render_template("register.html", error=f"Gagal claim kode: {e}", admin_code=settings.admin_master_code))

        access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
        redirect_url = "/admin" if role == "admin" else "/dashboard"
        response = RedirectResponse(url=redirect_url, status_code=303)

        secure_cookie = request.url.scheme == "https"
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            samesite="lax",
            secure=secure_cookie,
            max_age=int(settings.token_expire_minutes) * 60,
        )
        return response

    except ValueError as e:
        return HTMLResponse(render_template("register.html", error=str(e), admin_code=settings.admin_master_code))


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "user":
        return RedirectResponse(url="/admin")
    html = render_template("dashboard/index.html", username=current_user["username"])
    return HTMLResponse(html)


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, current_admin: dict = Depends(get_current_admin)):
    codes = db_list_mcp_codes(engine)
    for c in codes:
        ca = c.get("created_at")
        if isinstance(ca, datetime):
            c["created_at"] = ca.strftime("%Y-%m-%d %H:%M:%S")
        else:
            c["created_at"] = str(ca)
    html = render_template("admin.html", username=current_admin["username"], codes=codes)
    return HTMLResponse(html)


@app.post("/admin/codes")
async def admin_create_code(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
    token: str = Form(...),
):
    try:
        _ = db_create_mcp_code(engine, token)
    except Exception as e:
        logger.warning(f"[ADMIN] create code error: {e}")
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/codes/unclaim")
async def admin_unclaim_code(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
    code_id: int = Form(...),
):
    try:
        db_unclaim_mcp_code(engine, int(code_id))
    except Exception as e:
        logger.warning(f"[ADMIN] unclaim error: {e}")
    return RedirectResponse(url="/admin", status_code=303)


# =========================================================
# 12) PUBLIC + PROTECTED API ENDPOINTS
# =========================================================

@app.get("/healthz")
async def healthz():
    return {"ok": True, "llm_provider": settings.llm_provider, "cleanup_enabled": settings.cleanup_enabled}


@app.get("/api/public/validate-code")
async def api_public_validate_code(code: str = Query(...)):
    try:
        return db_validate_register_code(engine, code)
    except Exception as e:
        return {"ok": False, "available": False, "message": str(e)}


@app.get("/api/config")
async def api_config(current_user: dict = Depends(get_current_user)):
    forced = (settings.force_device_id or "").strip()
    return {
        "device_id_locked": bool(forced),
        "force_device_id": canonical_device_bucket(forced) if forced else "",
        "llm_provider": settings.llm_provider,
        "cleanup_enabled": bool(settings.cleanup_enabled),
        "cleanup_days": int(settings.cleanup_days),
        "cleanup_interval_minutes": int(settings.cleanup_interval_minutes),
        "db_drop_disabled": True,
        "psids": [DEFAULT_BUCKET] + PSID_LIST,
        "languages": LANGUAGES,
    }


@app.get("/api/last-device")
async def api_last_device(current_user: dict = Depends(get_current_user)):
    st_eff = db_get_app_state(engine, "last_device_id")
    st_phy = db_get_app_state(engine, "last_physical_device_id")
    st_act = db_get_app_state(engine, "active_psid")
    return {
        "last_physical_device_id": (st_phy.get("v") if st_phy else "") or "",
        "last_effective_device_id": canonical_device_bucket((st_eff.get("v") if st_eff else "") or ""),
        "active_psid": canonical_device_bucket((st_act.get("v") if st_act else "") or ""),
        "updated_at": (st_eff.get("updated_at") if st_eff else None),
    }


@app.get("/api/modes")
async def api_modes(current_user: dict = Depends(get_current_user)):
    return db_list_modes(engine)


@app.post("/api/modes")
async def api_save_mode(data: ModeUpsertIn, current_user: dict = Depends(get_current_user)):
    result = db_upsert_mode(engine, data.name, data.title, data.introduction)
    await manager.broadcast({"type": "modes_updated", "timestamp": time.time()})
    return result


@app.get("/api/mode")
async def api_get_active(device_id: str = Query(DEFAULT_BUCKET), current_user: dict = Depends(get_current_user)):
    return db_get_active_mode_for_device(engine, device_id)


@app.post("/api/mode")
async def api_set_active(data: SetActiveIn, current_user: dict = Depends(get_current_user)):
    psid = effective_device_id(data.device_id)
    result = db_set_active_mode_for_device(engine, psid, mode_id=data.mode_id)

    try:
        db_set_app_state(engine, "active_psid", psid)
        db_set_app_state(engine, "last_device_id", psid)
    except Exception:
        pass

    last_phys = None
    try:
        st = db_get_app_state(engine, "last_physical_device_id")
        last_phys = (st.get("v") if st else None) or None
    except Exception:
        last_phys = None

    if last_phys:
        try:
            db_set_route_psid(engine, last_phys, psid)
            await manager.broadcast({
                "type": "route_changed",
                "physical_device_id": normalize_device_id(last_phys),
                "psid": psid,
                "timestamp": time.time()
            })
        except Exception:
            pass

    await manager.broadcast({"type": "active_mode_changed", "device_id": psid, "mode_id": data.mode_id, "timestamp": time.time()})
    return result


@app.get("/api/role-introduction")
async def api_role_intro(current_user: dict = Depends(get_current_user)):
    return _mcp_text_result(build_role_introduction_for_xiaozhi())


@app.post("/api/render-prompt")
async def api_render_prompt(data: RenderPromptIn, current_user: dict = Depends(get_current_user)):
    mode: Optional[Dict[str, Any]] = None
    if data.mode_id is not None:
        mode = db_get_mode_by_id(engine, int(data.mode_id))
    if not mode and data.mode_name:
        modes = db_list_modes(engine)
        mode = next((m for m in modes if m["name"] == data.mode_name), None)
    if not mode:
        mode = db_get_active_mode(engine)

    intro = mode.get("introduction", "")
    title = mode.get("title", "")
    rendered_intro = safe_format(intro, data.vars or {})
    rendered_title = safe_format(title, data.vars or {})

    return {
        "prompt": build_system_prompt(
            mode_name=mode.get("name", ""),
            mode_title=rendered_title,
            intro_text=rendered_intro,
        )
    }


@app.get("/api/device/settings")
async def api_get_device_settings(device_id: str = Query(...), current_user: dict = Depends(get_current_user)):
    did = effective_device_id(device_id)
    st = db_get_device_settings(engine, did)
    return {"device_id": did, "source": st["source"], "target": st["target"]}


@app.post("/api/device/settings")
async def api_set_device_settings(data: DeviceSettingsIn, current_user: dict = Depends(get_current_user)):
    result = db_set_device_settings(engine, data.device_id, data.source, data.target)
    await manager.broadcast({"type": "device_settings_changed", "device_id": result["device_id"], "timestamp": time.time()})
    return result


@app.get("/api/chats")
async def api_list_chats(device_id: str = Query(...), current_user: dict = Depends(get_current_user)):
    return db_list_threads(engine, device_id)


@app.get("/api/chats/all")
async def api_chats_all(limit_per_device: int = Query(30), current_user: dict = Depends(get_current_user)):
    return db_list_threads_all_grouped(engine, limit_per_device=limit_per_device)


@app.post("/api/chats/new")
async def api_new_chat(data: ChatNewIn, current_user: dict = Depends(get_current_user)):
    result = db_create_thread(engine, data.device_id)
    await manager.broadcast({"type": "chat_created", "device_id": effective_device_id(data.device_id), "thread_id": result["id"], "timestamp": time.time()})
    return result


@app.get("/api/chats/{thread_id}/owner")
async def api_thread_owner(thread_id: int, current_user: dict = Depends(get_current_user)):
    owner = db_get_thread_owner_device(engine, thread_id)
    return {"thread_id": int(thread_id), "owner_device_id": owner}


@app.get("/api/chats/{thread_id}/messages")
async def api_messages(
    thread_id: int,
    device_id: str = Query(...),
    limit: int = Query(200),
    before_id: Optional[int] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    try:
        return db_get_messages_page(engine, device_id, thread_id, limit=limit, before_id=before_id)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.delete("/api/chats/{thread_id}")
async def api_delete_chat(thread_id: int, device_id: str = Query(...), current_user: dict = Depends(get_current_user)):
    try:
        result = db_delete_thread(engine, device_id, thread_id)
        await manager.broadcast({"type": "chat_deleted", "device_id": effective_device_id(device_id), "thread_id": thread_id, "timestamp": time.time()})
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/chats/send")
async def api_send_chat(data: ChatSendIn, current_user: dict = Depends(get_current_user)):
    try:
        user_text = (data.message or "").strip()
        if not user_text:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        device_id = effective_device_id(data.device_id)

        owner = db_get_thread_owner_device(engine, data.thread_id)
        if owner != device_id:
            return JSONResponse({"error": f"Thread not owned by {device_id} (owner={owner})"}, status_code=403)

        active_mode = db_get_active_mode_for_device(engine, device_id)
        if (active_mode.get("name") or "") != "normal":
            return JSONResponse({"error": "Send chat hanya diizinkan saat mode NORMAL aktif."}, status_code=403)

        db_add_message(engine, data.thread_id, "user", user_text)
        db_try_set_thread_title_from_first_user(engine, data.thread_id, user_text)

        mode = active_mode
        vars_dict = dict(data.vars or {})
        if mode_needs_lang(mode.get("name", "")):
            saved = db_get_device_settings(engine, device_id)
            vars_dict.setdefault("source", saved["source"])
            vars_dict.setdefault("target", saved["target"])

        rendered_title = safe_format((mode.get("title") or ""), vars_dict)
        rendered_intro = safe_format((mode.get("introduction") or ""), vars_dict)
        system_prompt = build_system_prompt(mode.get("name", ""), rendered_title, rendered_intro)

        summary_row = db_get_thread_summary(engine, data.thread_id)
        if summary_row and (summary_row.get("summary") or "").strip():
            system_prompt += "\n\nCONTEXT SUMMARY (rolling):\n" + summary_row["summary"].strip()

        full_msgs = db_get_messages_page(engine, device_id, data.thread_id, limit=200)
        llm_msgs: List[Dict[str, str]] = []
        for m in full_msgs[-30:]:
            if m["role"] in ("user", "assistant"):
                llm_msgs.append({"role": m["role"], "content": m["content"]})

        assistant_text = llm_generate(settings.llm_provider, system_prompt, llm_msgs)
        db_add_message(engine, data.thread_id, "assistant", assistant_text)

        try:
            maybe_roll_summary(engine, device_id, data.thread_id)
        except Exception:
            pass

        await manager.broadcast({"type": "chat_message_sent", "device_id": device_id, "thread_id": data.thread_id, "timestamp": time.time()})
        return {"ok": True}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# CLAIM DIMATIKAN: hanya lewat REGISTER
@app.post("/api/mcp/claim")
async def api_claim_mcp_code(data: ClaimCodeIn, current_user: dict = Depends(get_current_user)):
    return JSONResponse({"error": "Claim code hanya lewat halaman Register."}, status_code=403)


@app.get("/api/mcp/my-codes")
async def api_my_codes(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "user":
        return []
    return db_list_mcp_codes_for_user(engine, int(current_user["id"]))


@app.get("/api/mcp/all-codes")
async def api_all_codes(current_admin: dict = Depends(get_current_admin)):
    return db_list_mcp_codes(engine)


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":
    logger.info(
        f"Starting {settings.app_name} on {settings.app_host}:{settings.app_port} | "
        f"ENV={settings.app_env} | "
        f"LLM={settings.llm_provider} | FORCE_DEVICE_ID={(settings.force_device_id or '')!r} | "
        f"ADMIN_MASTER_CODE={settings.admin_master_code!r}"
    )
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=bool(settings.app_debug),
        log_level=(settings.log_level or "info").lower(),
    )
