# db.py
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import sys
import time
import urllib.request
import urllib.parse
import hashlib
import string
import os
import threading
from collections import deque
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
from cryptography.fernet import Fernet, InvalidToken
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

CODE_LENGTH = 10
CODE_ALPHABET = string.ascii_letters + string.digits
CODE_REGEX = re.compile(r"^[A-Za-z0-9]{10}$")

# Lightweight in-process metrics (per worker)
_METRICS_LOCK = threading.Lock()
_METRICS_HTTP_TS = deque()
_METRICS_HTTP_ERR_TS = deque()
_METRICS_MSG_TS = deque()
_METRICS_START = time.time()
_METRICS_HTTP_TOTAL = 0
_METRICS_HTTP_ERR = 0
_METRICS_MSG_TOTAL = 0


def _metrics_trim(dq: deque, now: float, window_sec: int = 300) -> None:
    while dq and (now - dq[0]) > window_sec:
        dq.popleft()


def _metrics_add_http(status_code: int) -> None:
    global _METRICS_HTTP_TOTAL, _METRICS_HTTP_ERR
    now = time.time()
    with _METRICS_LOCK:
        _METRICS_HTTP_TOTAL += 1
        if int(status_code) >= 400:
            _METRICS_HTTP_ERR += 1
            _METRICS_HTTP_ERR_TS.append(now)
        _METRICS_HTTP_TS.append(now)
        _metrics_trim(_METRICS_HTTP_TS, now, 300)
        _metrics_trim(_METRICS_HTTP_ERR_TS, now, 300)


def _metrics_add_message() -> None:
    global _METRICS_MSG_TOTAL
    now = time.time()
    with _METRICS_LOCK:
        _METRICS_MSG_TOTAL += 1
        _METRICS_MSG_TS.append(now)
        _metrics_trim(_METRICS_MSG_TS, now, 300)


def _metrics_snapshot() -> Dict[str, Any]:
    now = time.time()
    with _METRICS_LOCK:
        _metrics_trim(_METRICS_HTTP_TS, now, 300)
        _metrics_trim(_METRICS_HTTP_ERR_TS, now, 300)
        _metrics_trim(_METRICS_MSG_TS, now, 300)
        http_1m = sum(1 for t in _METRICS_HTTP_TS if now - t <= 60)
        http_err_1m = sum(1 for t in _METRICS_HTTP_ERR_TS if now - t <= 60)
        msg_1m = sum(1 for t in _METRICS_MSG_TS if now - t <= 60)
        snap = {
            "uptime_sec": int(now - _METRICS_START),
            "http_total": int(_METRICS_HTTP_TOTAL),
            "http_errors": int(_METRICS_HTTP_ERR),
            "http_last_1m": int(http_1m),
            "http_last_5m": int(len(_METRICS_HTTP_TS)),
            "http_errors_last_1m": int(http_err_1m),
            "http_errors_last_5m": int(len(_METRICS_HTTP_ERR_TS)),
            "msg_total": int(_METRICS_MSG_TOTAL),
            "msg_last_1m": int(msg_1m),
            "msg_last_5m": int(len(_METRICS_MSG_TS)),
        }
    return snap


def _process_mem_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    if os.name != "nt":
        try:
            import resource  # type: ignore
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On Linux, ru_maxrss is KB. On macOS, it is bytes.
            if rss > 10_000_000:
                return rss / (1024 * 1024)
            return rss / 1024
        except Exception:
            return None

    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        ):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        return None

    return None


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


def _is_valid_user_code(code: str) -> bool:
    return bool(CODE_REGEX.fullmatch((code or "").strip()))


def parse_mode_psid_map(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for part in re.split(r"[;,]", raw or ""):
        if not part.strip() or "=" not in part:
            continue
        k, v = part.split("=", 1)
        mode_key = normalize_device_id(k.strip())
        ps_raw = normalize_device_id(v.strip())
        if mode_key and ps_raw in ALLOWED_EFFECTIVE_IDS:
            mapping[mode_key] = ps_raw
    return mapping


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
    admin_master_code: str = Field(default="", validation_alias=AliasChoices("ADMIN_MASTER_CODE"))
    admin_master_code_hash: Optional[str] = Field(default=None, validation_alias=AliasChoices("ADMIN_MASTER_CODE_HASH"))

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
    # Allow disabling via legacy env name SCIG_MCP_ENABLED for convenience
    mcp_supervisor_enabled: bool = Field(default=True, validation_alias=AliasChoices("MCP_SUPERVISOR_ENABLED", "SCIG_MCP_ENABLED"))

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

    # Translation hints (optional)
    translation_glossary: str = Field(default="", validation_alias=AliasChoices("TRANSLATION_GLOSSARY"))
    translation_do_not_translate: str = Field(default="", validation_alias=AliasChoices("TRANSLATION_DO_NOT_TRANSLATE"))

    # Mode -> PSID map (comma separated, e.g. "curhat=psid_3,normal=default")
    mode_psid_map: str = Field(default="psikolog=psid_1,coding_expert=psid_2,curhat=psid_3,matematika_dasar=psid_4", validation_alias=AliasChoices("MODE_PSID_MAP"))

    # Token encryption (Fernet key, urlsafe base64 32 bytes)
    token_enc_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("TOKEN_ENC_KEY"))

    # Security hardening
    cookie_secure: Optional[bool] = Field(default=None, validation_alias=AliasChoices("COOKIE_SECURE"))
    cookie_samesite: str = Field(default="lax", validation_alias=AliasChoices("COOKIE_SAMESITE"))
    cors_allow_origins: str = Field(default="*", validation_alias=AliasChoices("CORS_ALLOW_ORIGINS"))
    cors_allow_credentials: bool = Field(default=False, validation_alias=AliasChoices("CORS_ALLOW_CREDENTIALS"))
    admin_allowed_ips: str = Field(default="", validation_alias=AliasChoices("ADMIN_ALLOWED_IPS"))
    trust_proxy_headers: bool = Field(default=False, validation_alias=AliasChoices("TRUST_PROXY_HEADERS"))
    security_headers_enabled: bool = Field(default=True, validation_alias=AliasChoices("SECURITY_HEADERS_ENABLED"))

    # Rate limits (per IP)
    rl_login_per_minute: int = Field(default=12, validation_alias=AliasChoices("RL_LOGIN_PER_MINUTE"))
    rl_register_per_minute: int = Field(default=6, validation_alias=AliasChoices("RL_REGISTER_PER_MINUTE"))
    rl_admin_per_minute: int = Field(default=60, validation_alias=AliasChoices("RL_ADMIN_PER_MINUTE"))

    # Password policy
    password_min_length: int = Field(default=6, validation_alias=AliasChoices("PASSWORD_MIN_LENGTH"))

    # Device lock (opsional)
    force_device_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("FORCE_DEVICE_ID"))

    # Cleanup
    cleanup_enabled: bool = Field(default=True, validation_alias=AliasChoices("CLEANUP_ENABLED"))
    cleanup_days: int = Field(default=30, validation_alias=AliasChoices("CLEANUP_DAYS"))
    cleanup_interval_minutes: int = Field(default=360, validation_alias=AliasChoices("CLEANUP_INTERVAL_MINUTES"))

    # Monitoring alerts
    monitor_mem_alert_mb: int = Field(default=600, validation_alias=AliasChoices("MONITOR_MEM_ALERT_MB"))
    monitor_http_error_rate_alert_pct: int = Field(default=20, validation_alias=AliasChoices("MONITOR_HTTP_ERROR_RATE_ALERT_PCT"))
    monitor_mcp_workers_min: int = Field(default=1, validation_alias=AliasChoices("MONITOR_MCP_WORKERS_MIN"))
    monitor_mcp_disconnected_alert_pct: int = Field(default=50, validation_alias=AliasChoices("MONITOR_MCP_DISCONNECTED_ALERT_PCT"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


RATE_LIMIT_BUCKETS: Dict[str, List[float]] = {}


def _parse_csv(value: str) -> List[str]:
    parts = [v.strip() for v in (value or "").split(",")]
    return [p for p in parts if p]


def _get_client_ip(request: Request) -> str:
    if settings.trust_proxy_headers:
        xf = request.headers.get("x-forwarded-for") or ""
        if xf:
            return xf.split(",")[0].strip()
        xr = request.headers.get("x-real-ip")
        if xr:
            return xr.strip()
    return getattr(request.client, "host", "") or ""


def _is_admin_ip_allowed(ip: str) -> bool:
    allow = _parse_csv(settings.admin_allowed_ips)
    if not allow:
        return True
    return ip in allow


def _rate_limit(key: str, limit: int, window_sec: int) -> None:
    now = time.time()
    bucket = RATE_LIMIT_BUCKETS.setdefault(key, [])
    # prune
    cutoff = now - window_sec
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
    bucket.append(now)


def _guard_admin(request: Request) -> None:
    ip = _get_client_ip(request)
    if not _is_admin_ip_allowed(ip):
        raise HTTPException(status_code=403, detail="Admin access denied from this IP")
    _rate_limit(f"admin:{ip}", max(1, int(settings.rl_admin_per_minute)), 60)


def _guard_login(request: Request) -> None:
    ip = _get_client_ip(request)
    _rate_limit(f"login:{ip}", max(1, int(settings.rl_login_per_minute)), 60)


def _guard_register(request: Request) -> None:
    ip = _get_client_ip(request)
    _rate_limit(f"register:{ip}", max(1, int(settings.rl_register_per_minute)), 60)


def _should_secure_cookie(request: Request) -> bool:
    if settings.cookie_secure is not None:
        return bool(settings.cookie_secure)
    if settings.trust_proxy_headers:
        proto = (request.headers.get("x-forwarded-proto") or "").lower().strip()
        if proto in ("https", "wss"):
            return True
    return request.url.scheme == "https"


def _cookie_samesite() -> str:
    val = (settings.cookie_samesite or "lax").strip().lower()
    if val not in ("lax", "strict", "none"):
        return "lax"
    return val


def _validate_password(password: str) -> None:
    if len(password or "") < int(settings.password_min_length):
        raise ValueError(f"Password minimal {int(settings.password_min_length)} karakter")


def _admin_code_match(code: str) -> bool:
    val = (code or "").strip()
    if not val:
        return False
    h = (settings.admin_master_code_hash or "").strip()
    if h:
        try:
            return hashlib.sha256(val.encode("utf-8")).hexdigest() == h.lower()
        except Exception:
            return False
    return val == (settings.admin_master_code or "")


def _security_startup_warnings() -> None:
    if (settings.secret_key or "") == "CHANGE_ME_IN_PRODUCTION":
        logger.warning("[SECURITY] SECRET_KEY masih default. Harap ganti di .env.")
    if settings.admin_master_code_hash:
        pass
    else:
        if (settings.admin_master_code or "").strip() == "":
            logger.warning("[SECURITY] ADMIN_MASTER_CODE/ADMIN_MASTER_CODE_HASH belum diisi. Admin register dinonaktifkan.")
        elif (settings.admin_master_code or "") == "26122003":
            logger.warning("[SECURITY] ADMIN_MASTER_CODE masih default. Harap ganti di .env.")
    if settings.cors_allow_origins.strip() == "*":
        logger.warning("[SECURITY] CORS_ALLOW_ORIGINS masih wildcard. Batasi di production.")
    if not settings.token_enc_key:
        logger.warning("[SECURITY] TOKEN_ENC_KEY belum diisi. Token tidak akan dienkripsi.")


def effective_device_id(incoming_device_id: str) -> str:
    forced = (settings.force_device_id or "").strip()
    if forced:
        return canonical_device_bucket(forced)
    return canonical_device_bucket(incoming_device_id)


def _mask_ws_url(url: str) -> str:
    return re.sub(r"(token=)([^&]+)", r"\1***", url)


_TOKEN_WS_CLEAN_RE = re.compile(r"[\s\u200b\u200c\u200d\ufeff]+")
_TOKEN_ENC_PREFIX = "enc:"


def normalize_ws_url(token_ws_url: str) -> str:
    raw = (token_ws_url or "").strip()
    if not raw:
        return ""
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        raw = raw[1:-1].strip()
    cleaned = _TOKEN_WS_CLEAN_RE.sub("", raw)
    return cleaned


def token_hash(raw_token: str) -> str:
    raw = normalize_ws_url(raw_token)
    if not raw:
        return ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@lru_cache
def _get_token_fernet() -> Optional[Fernet]:
    key = (settings.token_enc_key or "").strip()
    if not key:
        return None
    try:
        return Fernet(key)
    except Exception:
        logger.warning("[SECURITY] TOKEN_ENC_KEY invalid. Tokens will stay plaintext.")
        return None


def encrypt_token(token_ws_url: str) -> str:
    raw = normalize_ws_url(token_ws_url)
    if not raw:
        return ""
    if raw.startswith(_TOKEN_ENC_PREFIX):
        return raw
    f = _get_token_fernet()
    if not f:
        return raw
    enc = f.encrypt(raw.encode("utf-8")).decode("utf-8")
    return _TOKEN_ENC_PREFIX + enc


def decrypt_token(token_ws_url: str) -> str:
    raw = (token_ws_url or "").strip()
    if not raw:
        return ""
    if raw.startswith(_TOKEN_ENC_PREFIX):
        f = _get_token_fernet()
        if not f:
            raise ValueError("TOKEN_ENC_KEY not set for encrypted tokens")
        blob = raw[len(_TOKEN_ENC_PREFIX):]
        try:
            return f.decrypt(blob.encode("utf-8")).decode("utf-8")
        except InvalidToken as e:
            raise ValueError("Encrypted token cannot be decrypted") from e
    return raw


def _retention_days() -> int:
    try:
        days = int(settings.cleanup_days)
    except Exception:
        days = 30
    return max(7, min(days, 365))


logging.basicConfig(
    level=getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scig_mcp")

engine = create_engine(settings.database_url, pool_pre_ping=True, pool_recycle=1800)


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


def _try_get_user_from_request(request: Request) -> Optional[Dict[str, Any]]:
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        payload = verify_token(token)
    except Exception:
        return None
    username = payload.get("sub")
    if not username:
        return None
    return db_get_user_by_username(engine, username)


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
    return "".join(random.choices(CODE_ALPHABET, k=CODE_LENGTH))


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
            "TUGAS UTAMA:\n"
            "1) Jika input dalam bahasa '{source}', terjemahkan ke '{target}'.\n"
            "2) Jika input dalam bahasa '{target}', terjemahkan ke '{source}'.\n\n"
            "KECANGGIHAN & KUALITAS:\n"
            "- Deteksi bahasa input secara otomatis (source vs target).\n"
            "- Pertahankan makna, maksud, nuansa emosi, dan konteks.\n"
            "- Koreksi tata bahasa/penulisan tanpa mengubah maksud.\n"
            "- Istilah teknis/khusus: gunakan GLOSSARY jika tersedia.\n"
            "- Nama orang/brand/produk: jangan diterjemahkan.\n"
            "- DO_NOT_TRANSLATE: jangan ubah kata/frasanya bila ada daftar.\n"
            "- Angka & mata uang: terjemahkan nama mata uang bila muncul (misal USD -> Dollar AS) tanpa mengubah nilai.\n\n"
            "GLOSSARY (jika ada, prioritas):\n"
            "{glossary}\n\n"
            "DO_NOT_TRANSLATE (jika ada, pertahankan persis):\n"
            "{dont_translate}\n\n"
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
            "KUALITAS TERJEMAHAN:\n"
            "- Pertahankan makna, maksud, register, dan konteks.\n"
            "- Koreksi tata bahasa/penulisan tanpa mengubah maksud.\n"
            "- Istilah teknis/khusus: gunakan GLOSSARY jika tersedia.\n"
            "- Nama orang/brand/produk: jangan diterjemahkan.\n"
            "- DO_NOT_TRANSLATE: jangan ubah kata/frasanya bila ada daftar.\n"
            "- Angka & mata uang: terjemahkan nama mata uang bila muncul (misal USD -> Dollar AS) tanpa mengubah nilai.\n\n"
            "GLOSSARY (jika ada, prioritas):\n"
            "{glossary}\n\n"
            "DO_NOT_TRANSLATE (jika ada, pertahankan persis):\n"
            "{dont_translate}\n\n"
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
DEFAULT_MODE_NAME_SET = {m["name"] for m in DEFAULT_MODES}

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
  code VARCHAR(16) NOT NULL UNIQUE,
  token LONGTEXT NOT NULL,
  token_hash CHAR(64) NULL,
  used_by INT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (used_by) REFERENCES users(id) ON DELETE SET NULL,
  INDEX idx_auth_used_by (used_by),
  INDEX idx_auth_token_hash (token_hash)
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
  mode_id INT NULL,
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

CREATE TABLE IF NOT EXISTS user_devices (
  user_id INT NOT NULL,
  device_id VARCHAR(64) NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (user_id, device_id),
  INDEX idx_user_devices_user (user_id),
  INDEX idx_user_devices_device (device_id),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS admin_audit_logs (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  admin_user_id INT NULL,
  admin_username VARCHAR(64) NULL,
  action VARCHAR(64) NOT NULL,
  target_type VARCHAR(64) NULL,
  target_id VARCHAR(64) NULL,
  ip VARCHAR(64) NULL,
  user_agent VARCHAR(255) NULL,
  details TEXT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_audit_admin (admin_user_id),
  INDEX idx_audit_action (action),
  INDEX idx_audit_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        for statement in DDL_SCRIPT.split(";"):
            st = statement.strip()
            if st:
                conn.execute(text(st))
        try:
            conn.execute(text("ALTER TABLE auth_codes MODIFY token LONGTEXT NOT NULL"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE auth_codes MODIFY code VARCHAR(16) NOT NULL"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE auth_codes ADD COLUMN token_hash CHAR(64) NULL"))
        except Exception:
            pass
        try:
            conn.execute(text("CREATE INDEX idx_auth_token_hash ON auth_codes (token_hash)"))
        except Exception:
            pass
        try:
            conn.execute(text("CREATE TABLE IF NOT EXISTS user_devices (user_id INT NOT NULL, device_id VARCHAR(64) NOT NULL, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, PRIMARY KEY (user_id, device_id), INDEX idx_user_devices_user (user_id), INDEX idx_user_devices_device (device_id), FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE)"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE chat_threads ADD COLUMN mode_id INT NULL"))
        except Exception:
            pass

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

        normal_id = conn.execute(text("SELECT id FROM modes WHERE name='normal' LIMIT 1")).scalar()
        allowed_mode_names = [m["name"] for m in DEFAULT_MODES]
        allowed_params = {f"mode_name_{i}": n for i, n in enumerate(allowed_mode_names)}
        allowed_sql = ", ".join(f":mode_name_{i}" for i in range(len(allowed_mode_names)))
        disallowed_rows = conn.execute(
            text(f"SELECT id FROM modes WHERE name NOT IN ({allowed_sql})"),
            allowed_params,
        ).mappings().all()

        disallowed_ids = [int(r["id"]) for r in disallowed_rows if r.get("id") is not None]
        if disallowed_ids:
            id_params = {f"mid_{i}": v for i, v in enumerate(disallowed_ids)}
            id_sql = ", ".join(f":mid_{i}" for i in range(len(disallowed_ids)))

            if normal_id:
                conn.execute(
                    text(f"UPDATE active_mode SET mode_id=:normal_id WHERE mode_id IN ({id_sql})"),
                    {"normal_id": int(normal_id), **id_params},
                )
                conn.execute(
                    text(f"UPDATE device_active_mode SET mode_id=:normal_id WHERE mode_id IN ({id_sql})"),
                    {"normal_id": int(normal_id), **id_params},
                )
                conn.execute(
                    text(f"UPDATE chat_threads SET mode_id=:normal_id WHERE mode_id IN ({id_sql})"),
                    {"normal_id": int(normal_id), **id_params},
                )
            else:
                conn.execute(
                    text(f"DELETE FROM active_mode WHERE mode_id IN ({id_sql})"),
                    id_params,
                )
                conn.execute(
                    text(f"DELETE FROM device_active_mode WHERE mode_id IN ({id_sql})"),
                    id_params,
                )
                conn.execute(
                    text(f"UPDATE chat_threads SET mode_id=NULL WHERE mode_id IN ({id_sql})"),
                    id_params,
                )

            conn.execute(
                text(f"DELETE FROM modes WHERE id IN ({id_sql})"),
                id_params,
            )

        active = conn.execute(text("SELECT COUNT(*) FROM active_mode WHERE id=1")).scalar() or 0
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

    logger.info("Database initialized (NO DROP).")


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
    _validate_password(password)
    role = (role or "user").strip().lower()
    if role not in ("user", "admin"):
        raise ValueError("Role harus 'user' atau 'admin'")

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


def db_list_users(engine: Engine) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT u.id, u.username, u.role, u.created_at, u.updated_at,
                       COUNT(a.id) AS code_count,
                       SUM(CASE WHEN s.is_connected = 1 THEN 1 ELSE 0 END) AS connected_codes,
                       MAX(s.last_ok_at) AS last_ok_at,
                       MAX(s.last_err_at) AS last_err_at
                FROM users u
                LEFT JOIN auth_codes a ON a.used_by = u.id
                LEFT JOIN mcp_conn_status s ON s.code_id = a.id
                GROUP BY u.id, u.username, u.role, u.created_at, u.updated_at
                ORDER BY u.created_at DESC, u.id DESC
                """
            )
        ).mappings().all()
        return [dict(r) for r in rows]


def db_update_user(
    engine: Engine,
    user_id: int,
    *,
    password: Optional[str] = None,
    role: Optional[str] = None,
) -> Dict[str, Any]:
    user = db_get_user_by_id(engine, user_id)
    if not user:
        raise ValueError("User not found")

    updates: List[str] = []
    params: Dict[str, Any] = {"id": int(user_id)}

    if role is not None:
        role = (role or "").strip().lower()
        if role not in ("user", "admin"):
            raise ValueError("Role harus 'user' atau 'admin'")
        if user.get("role") == "admin" and role != "admin":
            with engine.begin() as conn:
                admin_count = conn.execute(
                    text("SELECT COUNT(*) FROM users WHERE role='admin'")
                ).scalar() or 0
            if admin_count <= 1:
                raise ValueError("Tidak bisa menurunkan role admin terakhir")
        updates.append("role=:role")
        params["role"] = role

    if password is not None and (password or "").strip():
        _validate_password(password)
        updates.append("password_hash=:ph")
        params["ph"] = hash_password(password)

    if not updates:
        return user

    with engine.begin() as conn:
        conn.execute(
            text(f"UPDATE users SET {', '.join(updates)} WHERE id=:id"),
            params,
        )
        row = conn.execute(
            text("SELECT id, username, role, created_at FROM users WHERE id=:id"),
            {"id": int(user_id)},
        ).mappings().fetchone()
        return dict(row) if row else user


def _try_cleanup_user_legacy(engine: Engine, user_id: int) -> None:
    # Best-effort cleanup for legacy tables that may still reference users.
    statements = [
        "DELETE FROM chat_sessions WHERE user_id=:u",
    ]
    for st in statements:
        try:
            with engine.begin() as conn:
                conn.execute(text(st), {"u": int(user_id)})
        except Exception:
            pass


def db_delete_user(engine: Engine, user_id: int, *, delete_codes: bool = False) -> Dict[str, Any]:
    # Read + validation first
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, username, role FROM users WHERE id=:id"),
            {"id": int(user_id)},
        ).mappings().fetchone()
        if not row:
            raise ValueError("User not found")

        if row.get("role") == "admin":
            admin_count = conn.execute(
                text("SELECT COUNT(*) FROM users WHERE role='admin'")
            ).scalar() or 0
            if admin_count <= 1:
                raise ValueError("Tidak bisa menghapus admin terakhir")

        code_ids = conn.execute(
            text("SELECT id FROM auth_codes WHERE used_by=:u"),
            {"u": int(user_id)},
        ).scalars().all()

    # Cleanup legacy FK tables (if any)
    _try_cleanup_user_legacy(engine, int(user_id))

    # Actual delete
    with engine.begin() as conn:
        if delete_codes and code_ids:
            params = {f"id{i}": int(cid) for i, cid in enumerate(code_ids)}
            placeholders = ", ".join(f":id{i}" for i in range(len(code_ids)))
            conn.execute(
                text(f"DELETE FROM auth_codes WHERE id IN ({placeholders})"),
                params,
            )
        else:
            conn.execute(
                text(
                    """
                    DELETE s FROM mcp_conn_status s
                    JOIN auth_codes a ON s.code_id = a.id
                    WHERE a.used_by = :u
                    """
                ),
                {"u": int(user_id)},
            )
            conn.execute(
                text("UPDATE auth_codes SET used_by=NULL WHERE used_by=:u"),
                {"u": int(user_id)},
            )
        conn.execute(
            text("DELETE FROM users WHERE id=:id"),
            {"id": int(user_id)},
        )

    return {
        "status": "deleted",
        "user_id": int(user_id),
        "username": row.get("username"),
        "code_ids": [int(x) for x in (code_ids or [])],
        "codes_deleted": bool(delete_codes),
    }


# ---------- AUTH CODES ----------

def db_create_mcp_code(engine: Engine, token_ws_url: str) -> Dict[str, Any]:
    raw = normalize_ws_url(token_ws_url)
    if not raw:
        raise ValueError("Token / WS URL kosong")
    th = token_hash(raw)

    with engine.begin() as conn:
        if th:
            existing = conn.execute(
                text("SELECT id, code, used_by FROM auth_codes WHERE token_hash=:h LIMIT 1"),
                {"h": th},
            ).mappings().fetchone()
            if existing:
                if existing.get("used_by") is not None:
                    raise ValueError("Token sudah dipakai user lain")
                raise ValueError("Token sudah ada, tidak boleh buat code baru")

        code = generate_code()
        while conn.execute(text("SELECT id FROM auth_codes WHERE code=:c"), {"c": code}).scalar():
            code = generate_code()

        conn.execute(
            text("INSERT INTO auth_codes (code, token, token_hash, used_by) VALUES (:c, :t, :h, NULL)"),
            {"c": code, "t": encrypt_token(raw), "h": th or None},
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


def db_update_mcp_code(engine: Engine, code_id: int, token_ws_url: str) -> Dict[str, Any]:
    raw = normalize_ws_url(token_ws_url)
    if not raw:
        raise ValueError("Token / WS URL kosong")
    th = token_hash(raw)

    with engine.begin() as conn:
        if th:
            existing = conn.execute(
                text("SELECT id FROM auth_codes WHERE token_hash=:h AND id<>:id LIMIT 1"),
                {"h": th, "id": int(code_id)},
            ).scalar()
            if existing:
                raise ValueError("Token sudah dipakai code lain")
        conn.execute(
            text("UPDATE auth_codes SET token=:t, token_hash=:h WHERE id=:id"),
            {"t": encrypt_token(raw), "h": th or None, "id": int(code_id)},
        )
        conn.execute(
            text("DELETE FROM mcp_conn_status WHERE code_id=:id"),
            {"id": int(code_id)},
        )
        row = conn.execute(
            text("SELECT id, code, token, used_by, created_at FROM auth_codes WHERE id=:id"),
            {"id": int(code_id)},
        ).mappings().fetchone()
        return dict(row) if row else {}


def db_update_mcp_code_value(engine: Engine, code_id: int, new_code: str) -> Dict[str, Any]:
    new_code = (new_code or "").strip()
    if not _is_valid_user_code(new_code):
        raise ValueError("Kode harus 10 karakter alfanumerik")
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM auth_codes WHERE code=:c AND id<>:id LIMIT 1"),
            {"c": new_code, "id": int(code_id)},
        ).scalar()
        if existing:
            raise ValueError("Code sudah ada")

        conn.execute(
            text("UPDATE auth_codes SET code=:c WHERE id=:id"),
            {"c": new_code, "id": int(code_id)},
        )
        row = conn.execute(
            text("SELECT id, code, token, used_by, created_at FROM auth_codes WHERE id=:id"),
            {"id": int(code_id)},
        ).mappings().fetchone()
        return dict(row) if row else {}


def db_delete_mcp_code(engine: Engine, code_id: int) -> Dict[str, Any]:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, code, used_by FROM auth_codes WHERE id=:id"),
            {"id": int(code_id)},
        ).mappings().fetchone()
        if not row:
            raise ValueError("Code tidak ditemukan")
        conn.execute(
            text("DELETE FROM auth_codes WHERE id=:id"),
            {"id": int(code_id)},
        )
        return {"status": "deleted", "code_id": int(code_id), "code": row.get("code"), "used_by": row.get("used_by")}


def db_claim_mcp_code(engine: Engine, code: str, user_id: int) -> Dict[str, Any]:
    code = (code or "").strip()
    if not _is_valid_user_code(code):
        raise ValueError("Kode harus 10 karakter alfanumerik")

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
                SELECT a.id, a.code, a.used_by, a.created_at,
                       CASE WHEN a.token IS NOT NULL AND a.token <> '' THEN 1 ELSE 0 END AS has_token,
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
                SELECT a.id, a.code, a.used_by, a.created_at,
                       CASE WHEN a.token IS NOT NULL AND a.token <> '' THEN 1 ELSE 0 END AS has_token,
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
                SELECT id, code, token, used_by, token_hash
                FROM auth_codes
                WHERE used_by IS NOT NULL AND token IS NOT NULL AND token <> ''
                ORDER BY id ASC
            """)
        ).mappings().all()
        out: List[Dict[str, Any]] = []
        for r in rows:
            tok_raw = (r.get("token") or "")
            try:
                tok = decrypt_token(tok_raw)
            except Exception as e:
                logger.warning(f"[SECURITY] Token decrypt failed for code_id={r.get('id')}: {e}")
                continue
            d = dict(r)
            d["token"] = tok
            out.append(d)
        return out


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

    if _admin_code_match(code):
        return {"ok": True, "kind": "admin", "available": True, "message": "Admin master code valid"}

    if not _is_valid_user_code(code):
        return {"ok": False, "kind": "user", "available": False, "message": "Kode harus 10 karakter alfanumerik"}

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


def db_unclaim_mcp_code(engine: Engine, code_id: int, *, delete_user: bool = False) -> Dict[str, Any]:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, code, used_by FROM auth_codes WHERE id=:id"),
            {"id": int(code_id)},
        ).mappings().fetchone()
        if not row:
            raise ValueError("Code tidak ditemukan")

        conn.execute(
            text("UPDATE auth_codes SET used_by=NULL WHERE id=:id"),
            {"id": int(code_id)},
        )
        conn.execute(
            text("DELETE FROM mcp_conn_status WHERE code_id=:id"),
            {"id": int(code_id)},
        )

    deleted_user = None
    if delete_user and row.get("used_by") is not None:
        try:
            deleted_user = db_delete_user(engine, int(row["used_by"]))
        except Exception as e:
            logger.warning(f"[ADMIN] delete user after unclaim failed: {e}")

    return {
        "ok": True,
        "code_id": int(code_id),
        "code": row.get("code"),
        "user_deleted": deleted_user,
    }


def db_encrypt_tokens_if_needed(engine: Engine) -> int:
    if not _get_token_fernet():
        logger.warning("[SECURITY] TOKEN_ENC_KEY not set. Tokens stored plaintext.")
        return 0

    updated = 0
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, token FROM auth_codes WHERE token IS NOT NULL AND token <> ''")
        ).mappings().all()
        for r in rows:
            tok = (r.get("token") or "").strip()
            if not tok or tok.startswith(_TOKEN_ENC_PREFIX):
                continue
            enc = encrypt_token(tok)
            if enc != tok:
                conn.execute(
                    text("UPDATE auth_codes SET token=:t WHERE id=:id"),
                    {"t": enc, "id": int(r["id"])},
                )
                updated += 1
    if updated:
        logger.info(f"[SECURITY] Encrypted {updated} token(s) in database.")
    return updated


def db_backfill_token_hash(engine: Engine) -> int:
    updated = 0
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, token, token_hash FROM auth_codes WHERE token IS NOT NULL AND token <> ''")
        ).mappings().all()
        for r in rows:
            if (r.get("token_hash") or "").strip():
                continue
            tok_raw = (r.get("token") or "").strip()
            if not tok_raw:
                continue
            try:
                raw = decrypt_token(tok_raw)
            except Exception:
                raw = tok_raw if not tok_raw.startswith(_TOKEN_ENC_PREFIX) else ""
            if not raw:
                continue
            th = token_hash(raw)
            if not th:
                continue
            conn.execute(
                text("UPDATE auth_codes SET token_hash=:h WHERE id=:id"),
                {"h": th, "id": int(r["id"])},
            )
            updated += 1
    if updated:
        logger.info(f"[SECURITY] Backfilled token_hash for {updated} token(s).")
    return updated


def _safe_details(details: Optional[Dict[str, Any]]) -> str:
    if not details:
        return ""
    try:
        text_val = json.dumps(details, ensure_ascii=False)
        return text_val[:2000]
    except Exception:
        return ""


def db_add_admin_audit_log(
    engine: Engine,
    *,
    admin_user: Optional[Dict[str, Any]],
    action: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    action = (action or "").strip()[:64] or "unknown"
    ttype = (target_type or "").strip()[:64] or None
    tid = (target_id or "").strip()[:64] or None
    ip = (ip or "").strip()[:64] or None
    ua = (user_agent or "").strip()[:255] or None
    admin_id = admin_user.get("id") if admin_user else None
    admin_name = (admin_user.get("username") if admin_user else None) or None
    det = _safe_details(details)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO admin_audit_logs
                  (admin_user_id, admin_username, action, target_type, target_id, ip, user_agent, details)
                VALUES
                  (:uid, :uname, :action, :ttype, :tid, :ip, :ua, :det)
                """
            ),
            {
                "uid": int(admin_id) if admin_id is not None else None,
                "uname": admin_name,
                "action": action,
                "ttype": ttype,
                "tid": tid,
                "ip": ip,
                "ua": ua,
                "det": det or None,
            },
        )


def db_list_admin_audit_logs(engine: Engine, limit: int = 200) -> List[Dict[str, Any]]:
    lim = max(1, min(int(limit), 500))
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, admin_user_id, admin_username, action, target_type, target_id,
                       ip, user_agent, details, created_at
                FROM admin_audit_logs
                ORDER BY id DESC
                LIMIT :lim
                """
            ),
            {"lim": lim},
        ).mappings().all()
        return [dict(r) for r in rows]


def db_upsert_user_device(engine: Engine, user_id: int, device_id: str) -> None:
    did = effective_device_id(device_id)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO user_devices (user_id, device_id)
                VALUES (:u, :d)
                ON DUPLICATE KEY UPDATE device_id=:d
                """
            ),
            {"u": int(user_id), "d": did},
        )


def db_list_user_devices(engine: Engine, user_id: int) -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT device_id FROM user_devices WHERE user_id=:u ORDER BY updated_at DESC"),
            {"u": int(user_id)},
        ).scalars().all()
        return [canonical_device_bucket(r) for r in rows if r]


def db_list_threads_for_device_any(engine: Engine, device_id: str) -> List[Dict[str, Any]]:
    did = canonical_device_bucket(device_id)
    days = _retention_days()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT t.id, t.device_id, t.title, t.created_at, t.updated_at, t.mode_id, "
                "m.name AS mode_name, m.title AS mode_title "
                "FROM chat_threads t "
                "LEFT JOIN modes m ON t.mode_id = m.id "
                f"WHERE t.device_id=:d AND t.updated_at >= DATE_SUB(NOW(), INTERVAL {days} DAY) "
                "ORDER BY t.updated_at DESC, t.id DESC"
            ),
            {"d": did},
        ).mappings().all()
        return [dict(r) for r in rows]


def db_get_messages_page_admin(
    engine: Engine,
    thread_id: int,
    limit: int = 200,
    before_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    lim = max(1, min(int(limit), 10000))
    days = _retention_days()
    with engine.begin() as conn:
        if before_id is not None:
            q = text(
                "SELECT id, role, content, created_at "
                "FROM chat_messages "
                f"WHERE thread_id=:tid AND id < :before AND created_at >= DATE_SUB(NOW(), INTERVAL {days} DAY) "
                "ORDER BY id DESC "
                f"LIMIT {lim}"
            )
            rows = conn.execute(q, {"tid": int(thread_id), "before": int(before_id)}).mappings().all()
        else:
            q = text(
                "SELECT id, role, content, created_at "
                "FROM chat_messages "
                f"WHERE thread_id=:tid AND created_at >= DATE_SUB(NOW(), INTERVAL {days} DAY) "
                "ORDER BY id DESC "
                f"LIMIT {lim}"
            )
            rows = conn.execute(q, {"tid": int(thread_id)}).mappings().all()
    rows = list(rows)[::-1]
    return [dict(r) for r in rows]


def db_resolve_partner_device_by_code(engine: Engine, code: str) -> Dict[str, Any]:
    """
    Given a 10-char auth code, return the claimed user's latest device_id (fallback to DEFAULT_BUCKET).
    Raises ValueError if code invalid or not claimed.
    """
    code_norm = (code or "").strip()
    if not CODE_REGEX.fullmatch(code_norm):
        raise ValueError("Kode harus 10 karakter alfanumerik")

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT used_by FROM auth_codes WHERE code=:c LIMIT 1"),
            {"c": code_norm},
        ).mappings().fetchone()
        if not row or row.get("used_by") is None:
            raise ValueError("Kode belum di-claim")
        partner_user_id = int(row["used_by"])

    devices = db_list_user_devices(engine, partner_user_id) or []
    partner_device_id = devices[0] if devices else DEFAULT_BUCKET
    return {"user_id": partner_user_id, "device_id": canonical_device_bucket(partner_device_id)}


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

def _db_get_normal_mode_row(conn) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        text("SELECT id, name, title, introduction FROM modes WHERE name='normal' LIMIT 1")
    ).mappings().fetchone()
    return dict(row) if row else None


def _db_should_force_normal_mode(conn) -> bool:
    """
    Force fallback to normal when MCP is effectively down:
    - there are claimed auth codes (used_by is set), and
    - none of the MCP connections is currently connected.
    """
    claimed_total = int(
        conn.execute(text("SELECT COUNT(*) FROM auth_codes WHERE used_by IS NOT NULL")).scalar() or 0
    )
    if claimed_total <= 0:
        return False
    any_connected = conn.execute(
        text("SELECT 1 FROM mcp_conn_status WHERE is_connected=1 LIMIT 1")
    ).scalar()
    return not bool(any_connected)


def _db_apply_normal_fallback_if_needed(conn, mode_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if mode_row is None:
        mode_row = {"id": 0, "name": "error", "title": "Error", "introduction": "No active mode set."}
    if _db_should_force_normal_mode(conn):
        normal = _db_get_normal_mode_row(conn)
        if normal:
            return normal
    return mode_row


def db_get_active_mode(engine: Engine) -> Dict[str, Any]:
    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT m.id, m.name, m.title, m.introduction "
                "FROM active_mode a JOIN modes m ON a.mode_id = m.id WHERE a.id=1"
            )
        ).mappings().fetchone()
        return _db_apply_normal_fallback_if_needed(conn, dict(row) if row else None)


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
        if not row:
            row = conn.execute(
                text(
                    "SELECT m.id, m.name, m.title, m.introduction "
                    "FROM active_mode a JOIN modes m ON a.mode_id = m.id WHERE a.id=1"
                )
            ).mappings().fetchone()
        return _db_apply_normal_fallback_if_needed(conn, dict(row) if row else None)


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
            text("SELECT id, name, title, introduction FROM modes WHERE id=:id AND name<>'walkie_talkie'"),
            {"id": int(mode_id)},
        ).mappings().fetchone()
        return dict(row) if row else None


def db_list_modes(engine: Engine) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, name, title, introduction FROM modes WHERE name<>'walkie_talkie' ORDER BY id ASC")
        ).mappings().all()
        return [dict(r) for r in rows]


def db_upsert_mode(engine: Engine, name: str, title: str, introduction: str) -> Dict[str, Any]:
    mode_name = normalize_device_id(name)
    if not mode_name:
        raise ValueError("Mode name wajib diisi.")
    if not (title or "").strip():
        raise ValueError("Mode title wajib diisi.")
    if not (introduction or "").strip():
        raise ValueError("Mode introduction wajib diisi.")

    with engine.begin() as conn:
        exists_id = conn.execute(
            text("SELECT id FROM modes WHERE name=:n LIMIT 1"),
            {"n": mode_name},
        ).scalar()
        if exists_id:
            conn.execute(
                text("UPDATE modes SET title=:t, introduction=:i WHERE id=:id"),
                {"t": title, "i": introduction, "id": int(exists_id)},
            )
            return {"status": "updated", "id": int(exists_id), "name": mode_name}

        conn.execute(
            text("INSERT INTO modes (name, title, introduction) VALUES (:n, :t, :i)"),
            {"n": mode_name, "t": title, "i": introduction},
        )
        new_id = conn.execute(
            text("SELECT id FROM modes WHERE name=:n LIMIT 1"),
            {"n": mode_name},
        ).scalar()
        return {"status": "created", "id": int(new_id), "name": mode_name}


def db_delete_mode(
    engine: Engine,
    *,
    mode_id: Optional[int] = None,
    mode_name: Optional[str] = None,
) -> Dict[str, Any]:
    if mode_id is None and not (mode_name or "").strip():
        raise ValueError("Pilih mode_id atau mode_name untuk delete.")

    with engine.begin() as conn:
        row = None
        if mode_id is not None:
            row = conn.execute(
                text("SELECT id, name FROM modes WHERE id=:id LIMIT 1"),
                {"id": int(mode_id)},
            ).mappings().fetchone()
        else:
            row = conn.execute(
                text("SELECT id, name FROM modes WHERE name=:n LIMIT 1"),
                {"n": normalize_device_id(mode_name or "")},
            ).mappings().fetchone()

        if not row:
            raise ValueError("Mode tidak ditemukan.")

        target_id = int(row["id"])
        target_name = str(row["name"] or "")

        if target_name in DEFAULT_MODE_NAME_SET:
            raise ValueError(f"Mode default '{target_name}' tidak boleh dihapus.")

        normal_id = conn.execute(text("SELECT id FROM modes WHERE name='normal' LIMIT 1")).scalar()
        if normal_id:
            conn.execute(
                text("UPDATE active_mode SET mode_id=:normal_id WHERE mode_id=:target_id"),
                {"normal_id": int(normal_id), "target_id": target_id},
            )
            conn.execute(
                text("UPDATE device_active_mode SET mode_id=:normal_id WHERE mode_id=:target_id"),
                {"normal_id": int(normal_id), "target_id": target_id},
            )
            conn.execute(
                text("UPDATE chat_threads SET mode_id=:normal_id WHERE mode_id=:target_id"),
                {"normal_id": int(normal_id), "target_id": target_id},
            )
        else:
            conn.execute(text("DELETE FROM active_mode WHERE mode_id=:target_id"), {"target_id": target_id})
            conn.execute(text("DELETE FROM device_active_mode WHERE mode_id=:target_id"), {"target_id": target_id})
            conn.execute(text("UPDATE chat_threads SET mode_id=NULL WHERE mode_id=:target_id"), {"target_id": target_id})

        conn.execute(text("DELETE FROM modes WHERE id=:id"), {"id": target_id})
        return {"status": "deleted", "id": target_id, "name": target_name}


def db_list_admin_audit_logs_page(
    engine: Engine,
    *,
    page: int = 1,
    page_size: int = 50,
    admin_username: str = "",
    action: str = "",
    date_from: str = "",
    date_to: str = "",
) -> Dict[str, Any]:
    p = max(1, int(page or 1))
    ps = max(10, min(int(page_size or 50), 200))

    where_parts: List[str] = []
    params: Dict[str, Any] = {}

    admin_username = (admin_username or "").strip()
    action = (action or "").strip()
    date_from = (date_from or "").strip()
    date_to = (date_to or "").strip()

    if admin_username:
        where_parts.append("admin_username LIKE :admin_username")
        params["admin_username"] = f"%{admin_username}%"

    if action:
        where_parts.append("action LIKE :action")
        params["action"] = f"%{action}%"

    try:
        if date_from:
            datetime.strptime(date_from, "%Y-%m-%d")
            where_parts.append("created_at >= :date_from")
            params["date_from"] = f"{date_from} 00:00:00"
    except Exception:
        date_from = ""

    try:
        if date_to:
            datetime.strptime(date_to, "%Y-%m-%d")
            where_parts.append("created_at <= :date_to")
            params["date_to"] = f"{date_to} 23:59:59"
    except Exception:
        date_to = ""

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    with engine.begin() as conn:
        total = int(
            conn.execute(
                text(f"SELECT COUNT(*) FROM admin_audit_logs {where_sql}"),
                params,
            ).scalar()
            or 0
        )

        total_pages = max(1, (total + ps - 1) // ps)
        p = min(p, total_pages)
        offset = (p - 1) * ps

        rows = conn.execute(
            text(
                f"""
                SELECT id, admin_user_id, admin_username, action, target_type, target_id,
                       ip, user_agent, details, created_at
                FROM admin_audit_logs
                {where_sql}
                ORDER BY id DESC
                LIMIT :lim OFFSET :off
                """
            ),
            {**params, "lim": ps, "off": offset},
        ).mappings().all()

    return {
        "logs": [dict(r) for r in rows],
        "page": int(p),
        "page_size": int(ps),
        "total": int(total),
        "total_pages": int(total_pages),
        "filters": {
            "admin_username": admin_username,
            "action": action,
            "date_from": date_from,
            "date_to": date_to,
        },
    }


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def safe_format(template: str, vars_dict: Dict[str, Any]) -> str:
    try:
        return template.format_map(SafeDict(vars_dict))
    except Exception:
        return template


XIAOZHI_STRICT_RULES = (
    "1) Setiap pesan user WAJIB diawali get_prompt sebelum respons apa pun.\n"
    "2) Argumen minimal get_prompt: {\"device_id\":\"<device_fisik>\"}.\n"
    "3) Gunakan hanya result.content[0].text sebagai system prompt aktif untuk giliran tersebut.\n"
    "4) Jika get_prompt gagal/timeout/hasil kosong, aktifkan fallback mode normal: jawab langsung sesuai pertanyaan user dengan akurat dan ringkas, tanpa menampilkan error MCP.\n"
    "5) Dilarang membuat data fiktif: angka, fakta, kutipan, referensi, status koneksi, atau hasil proses internal.\n"
    "6) Jika informasi kurang atau tidak yakin, nyatakan ketidakpastian secara eksplisit dan minta 1 klarifikasi paling penting.\n"
    "7) Setelah jawaban final, WAJIB panggil log_chat dengan: "
    "{\"device_id\":\"<device_fisik>\",\"user_text\":\"<teks_user_asli>\",\"assistant_text\":\"<jawaban_final>\"}. "
    "thread_id opsional jika valid.\n"
    "8) Jika log_chat gagal, retry 1x tanpa mengubah isi jawaban. Jika masih gagal, lanjut tanpa bocorkan error internal.\n"
    "9) Jangan ungkap detail internal tool/JSON/token/endpoint/backend ke user.\n"
    "10) Ulangi aturan ini dari langkah 1 untuk setiap pesan user baru.\n"
)


def mode_needs_lang(mode_name: str) -> bool:
    normalized = normalize_device_id(mode_name or "")
    return normalized in ("f2f_auto", "translation_text")


def build_system_prompt(mode_name: str, mode_title: str, intro_text: str) -> str:
    normalized_name = normalize_device_id(mode_name or "")
    label = (mode_title or "").strip() or (normalized_name or "custom_mode")
    intro = (intro_text or "").strip()

    sections = [
        f"MODE: {label} ({normalized_name or 'custom_mode'})",
        "INSTRUKSI MODE (DARI DATABASE):\n" + (intro or "- Tidak ada instruksi tambahan."),
    ]
    return "\n\n".join(sections)


def build_role_introduction_for_xiaozhi(style: str = "strict") -> str:
    s = (style or "strict").strip().lower()
    if s in ("natural", "friendly", "ramah"):
        return (
            "INSTRUKSI XIAOZHI (NATURAL, TETAP PATUH MCP):\n"
            "Tambahan gaya:\n"
            "- Saat menyusun jawaban final, gunakan bahasa ramah, jelas, dan langsung ke inti.\n"
            "- Hindari basa-basi panjang; tetap singkat dan relevan.\n\n"
            "PROTOKOL WAJIB:\n"
            + XIAOZHI_STRICT_RULES
        )

    return (
        "INSTRUKSI XIAOZHI (STRICT, ANTI-HALUSINASI):\n"
        + XIAOZHI_STRICT_RULES
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
    days = _retention_days()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT t.id, t.title, t.created_at, t.updated_at, t.mode_id, "
                "m.name AS mode_name, m.title AS mode_title "
                "FROM chat_threads t "
                "LEFT JOIN modes m ON t.mode_id = m.id "
                f"WHERE t.device_id=:d AND t.updated_at >= DATE_SUB(NOW(), INTERVAL {days} DAY) "
                "ORDER BY t.updated_at DESC, t.id DESC"
            ),
            {"d": did},
        ).mappings().all()
        return [dict(r) for r in rows]


def db_list_threads_all_grouped(engine: Engine, limit_per_device: int = 30) -> Dict[str, Any]:
    limit_per_device = max(1, min(int(limit_per_device), 200))
    days = _retention_days()

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
            SELECT t.id, t.device_id, t.title, t.created_at, t.updated_at, t.mode_id,
                   m.name AS mode_name, m.title AS mode_title
            FROM chat_threads t
            LEFT JOIN modes m ON t.mode_id = m.id
            WHERE t.updated_at >= DATE_SUB(NOW(), INTERVAL {days} DAY)
            ORDER BY t.updated_at DESC, t.id DESC
        """.format(days=days))).mappings().all()

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
    mode = db_get_active_mode_for_device(engine, did)
    mode_id = int(mode.get("id") or 0) or None
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO chat_threads (device_id, mode_id, title) VALUES (:d, :m, 'New Chat')"),
            {"d": did, "m": mode_id},
        )
        tid = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
        row = conn.execute(
            text("SELECT id, device_id, mode_id, title, created_at, updated_at FROM chat_threads WHERE id=:id"),
            {"id": int(tid)},
        ).mappings().fetchone()
        return dict(row) if row else {"id": int(tid), "device_id": did, "mode_id": mode_id, "title": "New Chat"}


def db_get_latest_thread_id(engine: Engine, device_id: str) -> Optional[int]:
    did = effective_device_id(device_id)
    with engine.begin() as conn:
        tid = conn.execute(
            text("SELECT id FROM chat_threads WHERE device_id=:d ORDER BY updated_at DESC, id DESC LIMIT 1"),
            {"d": did},
        ).scalar()
        return int(tid) if tid else None


def db_get_or_create_latest_thread_id(engine: Engine, device_id: str) -> int:
    did = effective_device_id(device_id)
    with engine.begin() as conn:
        tid = conn.execute(
            text("SELECT id FROM chat_threads WHERE device_id=:d ORDER BY updated_at DESC, id DESC LIMIT 1"),
            {"d": did},
        ).scalar()
        if tid:
            return int(tid)

        mode = db_get_active_mode_for_device(engine, did)
        mode_id = int(mode.get("id") or 0) or None
        conn.execute(
            text("INSERT INTO chat_threads (device_id, mode_id, title) VALUES (:d, :m, 'New Chat')"),
            {"d": did, "m": mode_id},
        )
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
    after_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    did = effective_device_id(device_id)
    lim = max(1, min(int(limit), 10000))
    days = _retention_days()

    with engine.begin() as conn:
        owned = conn.execute(
            text("SELECT id FROM chat_threads WHERE id=:id AND device_id=:d"),
            {"id": int(thread_id), "d": did},
        ).scalar()
        if not owned:
            return []

        if after_id is not None:
            q = text(
                "SELECT id, role, content, created_at "
                "FROM chat_messages "
                f"WHERE thread_id=:tid AND id > :after AND created_at >= DATE_SUB(NOW(), INTERVAL {days} DAY) "
                "ORDER BY id ASC "
                f"LIMIT {lim}"
            )
            rows = conn.execute(q, {"tid": int(thread_id), "after": int(after_id)}).mappings().all()
            return [dict(r) for r in rows]

        if before_id is not None:
            q = text(
                "SELECT id, role, content, created_at "
                "FROM chat_messages "
                f"WHERE thread_id=:tid AND id < :before AND created_at >= DATE_SUB(NOW(), INTERVAL {days} DAY) "
                "ORDER BY id DESC "
                f"LIMIT {lim}"
            )
            rows = conn.execute(q, {"tid": int(thread_id), "before": int(before_id)}).mappings().all()
        else:
            q = text(
                "SELECT id, role, content, created_at "
                "FROM chat_messages "
                f"WHERE thread_id=:tid AND created_at >= DATE_SUB(NOW(), INTERVAL {days} DAY) "
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
    if role in ("user", "assistant"):
        _metrics_add_message()


def db_try_set_thread_title_from_first_user(engine: Engine, thread_id: int, user_text: str) -> None:
    title = (user_text or "").strip().replace("\n", " ")
    title = title[:48] + ("..." if len(title) > 48 else "")
    if not title:
        return
    with engine.begin() as conn:
        cur = conn.execute(text("SELECT title FROM chat_threads WHERE id=:id"), {"id": int(thread_id)}).scalar()
        if cur and cur == "New Chat":
            conn.execute(text("UPDATE chat_threads SET title=:t WHERE id=:id"), {"t": title, "id": int(thread_id)})


def build_recent_chat_context(
    engine: Engine,
    device_id: str,
    *,
    days: int = 30,
    limit_messages: int = 80,
    max_chars: int = 4000,
) -> str:
    days = max(1, min(int(days), 365))
    lim = max(1, min(int(limit_messages), 200))
    max_chars = max(400, min(int(max_chars), 12000))

    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT m.id, m.role, m.content, m.created_at "
                "FROM chat_messages m "
                "JOIN chat_threads t ON m.thread_id = t.id "
                "WHERE t.device_id=:d "
                "AND m.role IN ('user','assistant') "
                f"AND m.created_at >= DATE_SUB(NOW(), INTERVAL {days} DAY) "
                "ORDER BY m.created_at DESC, m.id DESC "
                f"LIMIT {lim}"
            ),
            {"d": effective_device_id(device_id)},
        ).mappings().all()

    if not rows:
        return ""

    rows = list(rows)[::-1]
    lines_rev: List[str] = []
    total = 0
    for r in reversed(rows):
        role = (r.get("role") or "").upper()
        content = (r.get("content") or "").replace("\n", " ").strip()
        if not content:
            continue
        if len(content) > 500:
            content = content[:500].rstrip() + "..."
        line = f"{role}: {content}"
        if total + len(line) + 1 > max_chars:
            break
        lines_rev.append(line)
        total += len(line) + 1

    if not lines_rev:
        return ""

    return "\n".join(reversed(lines_rev))


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


def db_cleanup_old_messages(engine: Engine, *, days: int = 30) -> Dict[str, Any]:
    days = max(7, min(int(days), 365))

    with engine.begin() as conn:
        count = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM chat_messages
                WHERE created_at < DATE_SUB(NOW(), INTERVAL {days} DAY)
                """
            )
        ).scalar() or 0

        if count:
            conn.execute(
                text(
                    f"""
                    DELETE FROM chat_messages
                    WHERE created_at < DATE_SUB(NOW(), INTERVAL {days} DAY)
                    """
                )
            )
    return {"deleted_messages": int(count), "days": days}


def db_cleanup_old_summaries(engine: Engine, *, days: int = 30) -> Dict[str, Any]:
    days = max(7, min(int(days), 365))

    with engine.begin() as conn:
        count = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM thread_summaries
                WHERE updated_at < DATE_SUB(NOW(), INTERVAL {days} DAY)
                """
            )
        ).scalar() or 0

        if count:
            conn.execute(
                text(
                    f"""
                    DELETE FROM thread_summaries
                    WHERE updated_at < DATE_SUB(NOW(), INTERVAL {days} DAY)
                    """
                )
            )
    return {"deleted_summaries": int(count), "days": days}


async def cleanup_worker(engine: Engine) -> None:
    if not settings.cleanup_enabled:
        logger.info("[CLEANUP] Disabled.")
        return

    interval = max(15, int(settings.cleanup_interval_minutes))
    days = int(settings.cleanup_days)

    logger.info(f"[CLEANUP] Worker started. interval={interval}min, days={days}")
    while True:
        try:
            resm = db_cleanup_old_messages(engine, days=days)
            res = db_cleanup_old_threads(engine, days=days)
            ress = db_cleanup_old_summaries(engine, days=days)
            if resm.get("deleted_messages"):
                logger.info(f"[CLEANUP] Deleted {resm['deleted_messages']} message(s) older than {resm['days']} days.")
            if res.get("deleted_threads"):
                logger.info(f"[CLEANUP] Deleted {res['deleted_threads']} thread(s) older than {res['days']} days.")
            if ress.get("deleted_summaries"):
                logger.info(f"[CLEANUP] Deleted {ress['deleted_summaries']} summary row(s) older than {ress['days']} days.")
        except Exception as e:
            logger.warning(f"[CLEANUP] Error: {e}")
        await asyncio.sleep(interval * 60)


