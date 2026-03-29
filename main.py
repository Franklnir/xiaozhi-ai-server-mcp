# main.py
from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.parse
import os  # PENTING: Untuk membaca variabel environment Railway
import secrets
import requests
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field
from sqlalchemy import text
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Import module lokal
from auth import (
    create_access_token,
    get_current_user,
    get_current_admin,
    _guard_admin,
    _guard_login,
    _guard_register,
    _should_secure_cookie,
    _cookie_samesite,
    _try_get_user_from_request,
    _get_client_ip,
    _parse_csv,
)
from db import (
    settings,
    logger,
    engine,
    DEFAULT_BUCKET,
    PSID_LIST,
    LANGUAGES,
    parse_mode_psid_map,
    normalize_ws_url,
    normalize_device_id,
    canonical_device_bucket,
    effective_device_id,
    _hash_token,
    init_db,
    cleanup_worker,
    _security_startup_warnings,
    db_encrypt_tokens_if_needed,
    db_backfill_token_hash,
    _metrics_add_http,
    _metrics_snapshot,
    _metrics_add_message,
    _process_mem_mb,
    verify_password,
    db_get_user_by_username,
    db_validate_register_code,
    db_create_user,
    db_claim_mcp_code,
    db_list_mcp_codes,
    db_list_users,
    db_list_admin_audit_logs,
    db_create_mcp_code,
    db_update_mcp_code,
    db_clear_mcp_code_token,
    db_update_mcp_code_value,
    db_delete_mcp_code,
    db_unclaim_mcp_code,
    db_add_admin_audit_log,
    db_update_user,
    db_delete_user,
    db_list_mcp_codes_for_user,
    db_list_modes,
    db_upsert_mode,
    db_get_active_mode_for_device,
    db_set_active_mode_for_device,
    db_get_active_mode,
    db_get_mode_by_id,
    db_get_device_settings,
    db_set_device_settings,
    db_list_threads,
    db_list_threads_all_grouped,
    db_create_thread,
    db_get_thread_owner_device,
    db_get_messages_page,
    db_delete_thread,
    db_add_message,
    db_try_set_thread_title_from_first_user,
    db_get_thread_summary,
    maybe_roll_summary,
    llm_generate,
    safe_format,
    mode_needs_lang,
    build_system_prompt,
    build_role_introduction_for_xiaozhi,
    db_get_app_state,
    db_set_app_state,
    db_set_route_psid,
    db_upsert_user_device,
    db_list_user_devices,
    db_link_tracked_device,
    db_get_user_device_alias,
    db_upsert_device_registry,
    db_get_device_registry,
    db_get_device_by_token_hash,
    db_update_device_last_seen,
    db_upsert_device_status,
    db_get_device_status,
    db_add_device_location_history,
    db_list_device_location_history,
    db_list_devices_for_user,
    db_set_user_device_alias,
    db_unlink_tracked_device,
    db_create_pair_token,
    db_claim_pair_token,
    db_get_geocode_cache,
    db_set_geocode_cache,
    db_list_devices_admin,
    db_list_threads_for_device_any,
    db_get_messages_page_admin,
)
from mcp import mcp_supervisor, stop_mcp_worker, _mcp_text_result

# ==================================================# 10) FASTAPI APP + TEMPLATE ENGINE
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
LOCAL_RELEASES_DIR = BASE_DIR / "releases"
MOBILE_APK_DIR = BASE_DIR / "mobile" / "android" / "app" / "build" / "outputs" / "apk"
MOBILE_RELEASE_CONFIG_PATHS = (
    BASE_DIR / "mobile_release.json",
    BASE_DIR / "mobile" / "mobile_release.json",
)


def _load_mobile_release_config() -> Dict[str, str]:
    config = {
        "appName": "xiaozhiscig",
        "appVersion": "0.7.1",
        "defaultServerUrl": "https://xiaozhiscig.biz.id",
        "apkFallbackUrl": "https://github.com/Franklnir/xiaozhi-ai-mobile-mcp/releases",
    }
    for config_path in MOBILE_RELEASE_CONFIG_PATHS:
        if not config_path.is_file():
            continue
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        for key in config:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                config[key] = value.strip()
        break
    return config


MOBILE_RELEASE_CONFIG = _load_mobile_release_config()
MOBILE_APP_NAME = os.environ.get("MOBILE_APP_NAME", MOBILE_RELEASE_CONFIG["appName"])
MOBILE_APP_VERSION = os.environ.get("MOBILE_APP_VERSION", MOBILE_RELEASE_CONFIG["appVersion"])
MOBILE_DEFAULT_SERVER_URL = os.environ.get(
    "MOBILE_DEFAULT_SERVER_URL",
    MOBILE_RELEASE_CONFIG["defaultServerUrl"],
)

# APK download URL (GitHub releases)
APK_DOWNLOAD_URL = os.environ.get(
    "APK_DOWNLOAD_URL",
    MOBILE_RELEASE_CONFIG["apkFallbackUrl"],
)

jinja_env: Environment = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)

def render_template(name: str, **ctx) -> str:
    return jinja_env.get_template(name).render(**ctx)


def _find_local_apk() -> Optional[Path]:
    candidates: List[Path] = []
    search_roots = [
        LOCAL_RELEASES_DIR,
        MOBILE_APK_DIR / "release",
        MOBILE_APK_DIR / "debug",
    ]
    for root in search_roots:
        if root.exists():
            candidates.extend(path for path in root.glob("*.apk") if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)

def _state_bool(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    val = str(value).strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    if val in ("1", "true", "yes", "on"):
        return True
    return default


def _get_register_requires_code() -> bool:
    st = db_get_app_state(engine, "register_requires_code")
    return _state_bool(st.get("v") if st else None, default=True)


def _clean_social_links(raw: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        url = str(item.get("url") or "").strip()
        if not label or not url:
            continue
        out.append({"label": label[:32], "url": url[:2000]})
        if len(out) >= 20:
            break
    return out


_UNSAFE_HTTP_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_CSRF_EXEMPT_PATHS = {"/login", "/register", "/logout", "/healthz"}
_CSRF_EXEMPT_PREFIXES = ("/api/public/",)


def _is_mobile_request(request: Request) -> bool:
    return (request.headers.get("x-client") or "").strip().lower() == "mobile"


def _require_mobile_request(request: Request) -> None:
    if _is_mobile_request(request):
        return
    raise HTTPException(status_code=403, detail="Data perangkat HP hanya tersedia lewat aplikasi mobile.")


def _request_scheme(request: Request) -> str:
    if settings.trust_proxy_headers:
        proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
        if proto in ("https", "wss"):
            return "https"
        if proto in ("http", "ws"):
            return "http"
    return request.url.scheme


def _normalize_origin_value(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    parsed = urllib.parse.urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return ""
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return ""
    return f"{scheme}://{parsed.netloc.lower()}"


def _request_origin(request: Request) -> str:
    host = (request.headers.get("host") or request.url.netloc or "").strip().lower()
    if not host:
        return ""
    return _normalize_origin_value(f"{_request_scheme(request)}://{host}")


def _trusted_request_origins(request: Request) -> set[str]:
    trusted: set[str] = set()
    current_origin = _request_origin(request)
    if current_origin:
        trusted.add(current_origin)

    for raw in _parse_csv(settings.csrf_trusted_origins):
        normalized = _normalize_origin_value(raw)
        if normalized:
            trusted.add(normalized)

    for raw in _parse_csv(settings.cors_allow_origins):
        if raw == "*":
            continue
        normalized = _normalize_origin_value(raw)
        if normalized:
            trusted.add(normalized)

    return trusted


def _requires_origin_check(request: Request) -> bool:
    if request.method.upper() not in _UNSAFE_HTTP_METHODS:
        return False
    if _is_mobile_request(request):
        return False
    if request.url.path in _CSRF_EXEMPT_PATHS:
        return False
    if any(request.url.path.startswith(prefix) for prefix in _CSRF_EXEMPT_PREFIXES):
        return False
    return bool(request.cookies.get("access_token"))


def _csrf_error_response(request: Request):
    payload = {"error": "Invalid request origin"}
    if request.url.path.startswith("/api/") or _wants_json(request):
        return JSONResponse(payload, status_code=403)
    return HTMLResponse("Invalid request origin", status_code=403)


def _location_changed(existing: Dict[str, Any], lat: Optional[float], lon: Optional[float]) -> bool:
    if lat is None or lon is None:
        return False
    prev_lat = existing.get("latitude")
    prev_lon = existing.get("longitude")
    if prev_lat is None or prev_lon is None:
        return True
    try:
        return round(float(prev_lat), 6) != round(float(lat), 6) or round(float(prev_lon), 6) != round(float(lon), 6)
    except Exception:
        return True


def _get_social_links() -> List[Dict[str, str]]:
    st = db_get_app_state(engine, "social_links")
    if not st or not st.get("v"):
        return []
    try:
        raw = json.loads(st.get("v") or "[]")
    except Exception:
        return []
    return _clean_social_links(raw)


def _set_social_links(links: List[Dict[str, str]]) -> None:
    db_set_app_state(engine, "social_links", json.dumps(links, ensure_ascii=False))


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
    device_id: Optional[str] = None
    title: Optional[str] = None
    introduction: Optional[str] = None
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


class DeviceRegisterIn(BaseModel):
    device_id: str
    device_name: Optional[str] = None
    device_token: Optional[str] = None
    platform: Optional[str] = None
    model: Optional[str] = None
    os_version: Optional[str] = None


class DeviceHeartbeatIn(BaseModel):
    device_id: str
    device_token: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    battery_level: Optional[float] = None
    battery_status: Optional[str] = None
    charging_type: Optional[str] = None
    battery_temp: Optional[float] = None
    network_type: Optional[str] = None
    signal_strength: Optional[int] = None
    carrier: Optional[str] = None
    ram_used: Optional[int] = None
    ram_total: Optional[int] = None
    storage_used: Optional[int] = None
    storage_total: Optional[int] = None


class DeviceAliasIn(BaseModel):
    device_id: str
    alias: str


class DeviceUnpairIn(BaseModel):
    device_id: str


class DevicePairTokenIn(BaseModel):
    device_id: str
    device_token: str


class DevicePairClaimIn(BaseModel):
    pair_token: str
    alias: Optional[str] = None


class ClaimCodeIn(BaseModel):
    code: str


class AdminUnclaimIn(BaseModel):
    code_id: int


class McpTokenIn(BaseModel):
    token: str


class McpTokenUpdateIn(BaseModel):
    code_id: int
    token: str


class McpTokenClearIn(BaseModel):
    code_id: int


class McpTokenTestIn(BaseModel):
    token: str


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

GEO_CACHE_TTL_DAYS = 7


def _geo_key(lat: float, lon: float) -> tuple[str, str]:
    return f"{lat:.5f}", f"{lon:.5f}"


async def reverse_geocode(lat: float, lon: float) -> Dict[str, Any]:
    lat_key, lon_key = _geo_key(lat, lon)
    cached = db_get_geocode_cache(engine, lat_key, lon_key)
    if cached:
        updated_at = cached.get("updated_at")
        if not updated_at or updated_at >= datetime.utcnow() - timedelta(days=GEO_CACHE_TTL_DAYS):
            return cached.get("data") or {}

    def _fetch() -> Dict[str, Any]:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"format": "jsonv2", "lat": lat, "lon": lon}
        headers = {"User-Agent": "SciG-Mode-MCP/1.0 (contact: admin@example.com)"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    try:
        payload = await asyncio.to_thread(_fetch)
    except Exception:
        return {}

    address = payload.get("address") or {}
    street = address.get("road") or address.get("pedestrian") or address.get("footway") or ""
    area = (
        address.get("suburb")
        or address.get("neighbourhood")
        or address.get("village")
        or address.get("hamlet")
        or ""
    )
    city = (
        address.get("city")
        or address.get("town")
        or address.get("county")
        or address.get("municipality")
        or address.get("state_district")
        or ""
    )
    full = payload.get("display_name") or ""

    result = {
        "street": street,
        "area": area,
        "city": city,
        "full": full,
    }
    try:
        db_set_geocode_cache(engine, lat_key, lon_key, result)
    except Exception:
        pass
    return result


def _format_device_address(device: Dict[str, Any]) -> str:
    addr = device.get("address_full") or ""
    if addr:
        return addr
    parts = [device.get("address_street"), device.get("address_area"), device.get("address_city")]
    return ", ".join([p for p in parts if p]) or "Alamat belum tersedia"


def _format_device_snapshot(device: Dict[str, Any], heading: str) -> List[str]:
    label = device.get("alias") or device.get("device_name") or device.get("device_id") or "-"
    lines = [heading]
    lines.append(f"- Nama: {label}")
    if device.get("device_id"):
        lines.append(f"- Device ID: {device.get('device_id')}")
    lines.append(f"- Status: {'Online' if device.get('is_online') else 'Offline'}")
    lines.append(f"- Alamat: {_format_device_address(device)}")

    bat = device.get("battery_level")
    if bat is not None:
        bat_pct = f"{int(bat * 100)}%" if bat <= 1 else f"{bat:.0f}%"
        bat_status = device.get("battery_status") or "-"
        chg = device.get("charging_type") or "-"
        lines.append(f"- Baterai: {bat_pct} ({bat_status}, {chg})")
    if device.get("battery_temp") is not None:
        lines.append(f"- Suhu baterai: {device.get('battery_temp'):.1f}°C")

    net = device.get("network_type") or "-"
    carrier = device.get("carrier") or "-"
    lines.append(f"- Jaringan: {net} / {carrier}")

    if device.get("ram_used") and device.get("ram_total"):
        lines.append(f"- RAM: {device.get('ram_used')} / {device.get('ram_total')}")
    if device.get("storage_used") and device.get("storage_total"):
        lines.append(f"- Storage: {device.get('storage_used')} / {device.get('storage_total')}")
    return lines


def build_device_status_text(device_id: str, user_id: Optional[int] = None) -> str:
    reg = db_get_device_registry(engine, device_id) or {}
    status = db_get_device_status(engine, device_id) or {}
    devices: List[Dict[str, Any]] = []
    if user_id is not None:
        try:
            devices = db_list_devices_for_user(engine, int(user_id))
        except Exception:
            devices = []

    primary = next((item for item in devices if item.get("device_id") == device_id), None)
    if primary is None and (reg or status):
        primary = {
            "device_id": device_id,
            "device_name": reg.get("device_name"),
            "alias": reg.get("device_name") or device_id,
            "is_online": status.get("is_online"),
            **status,
        }

    if primary is None:
        return "Data perangkat belum tersedia."

    lines: List[str] = []
    lines.extend(_format_device_snapshot(primary, "PERANGKAT UTAMA USER SAAT INI:"))

    paired_devices = [item for item in devices if item.get("device_id") != device_id]
    if paired_devices:
        lines.append("")
        lines.append("PERANGKAT LAIN YANG SUDAH DIPAIRKAN:")
        for index, paired in enumerate(paired_devices[:6], start=1):
            pair_label = paired.get("alias") or paired.get("device_name") or paired.get("device_id") or "-"
            lines.extend(_format_device_snapshot(paired, f"{index}. DEVICE PAIR: {pair_label}"))
            if index < min(len(paired_devices), 6):
                lines.append("")

    return "\n".join(lines)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(engine)
    _security_startup_warnings()
    try:
        db_encrypt_tokens_if_needed(engine)
        db_backfill_token_hash(engine)
    except Exception as e:
        logger.warning(f"[SECURITY] token encryption migration failed: {e}")
    task_cleanup = asyncio.create_task(cleanup_worker(engine))
    task_mcp_sup = None
    if settings.mcp_supervisor_enabled:
        task_mcp_sup = asyncio.create_task(mcp_supervisor.run(engine, manager_ref=manager))
    try:
        yield
    finally:
        for t in (task_cleanup, task_mcp_sup):
            if not t:
                continue
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if settings.mcp_supervisor_enabled:
            try:
                await mcp_supervisor.shutdown()
            except Exception:
                pass


app = FastAPI(title=settings.app_name, lifespan=lifespan)

allowed_hosts = _parse_csv(settings.allowed_hosts)
if allowed_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)


@app.middleware("http")
async def enforce_https(request: Request, call_next):
    if settings.force_https and _request_scheme(request) != "https":
        secure_url = str(request.url.replace(scheme="https"))
        return RedirectResponse(url=secure_url, status_code=307)
    return await call_next(request)


@app.middleware("http")
async def csrf_origin_guard(request: Request, call_next):
    if _requires_origin_check(request):
        trusted_origins = _trusted_request_origins(request)
        request_origin = _normalize_origin_value(request.headers.get("origin") or "")
        referer_origin = _normalize_origin_value(request.headers.get("referer") or "")
        sec_fetch_site = (request.headers.get("sec-fetch-site") or "").strip().lower()

        if request_origin:
            if request_origin not in trusted_origins:
                return _csrf_error_response(request)
        elif referer_origin:
            if referer_origin not in trusted_origins:
                return _csrf_error_response(request)
        elif sec_fetch_site:
            if sec_fetch_site not in ("same-origin", "same-site", "none"):
                return _csrf_error_response(request)
        else:
            return _csrf_error_response(request)

    return await call_next(request)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception:
        _metrics_add_http(500)
        raise
    _metrics_add_http(response.status_code)
    return response

cors_origins = _parse_csv(settings.cors_allow_origins)
if not cors_origins:
    cors_origins = ["*"]
if settings.cors_allow_credentials and "*" in cors_origins:
    logger.warning("[SECURITY] CORS_ALLOW_CREDENTIALS true but CORS_ALLOW_ORIGINS is '*'. Remove wildcard.")
    cors_origins = [o for o in cors_origins if o != "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=bool(settings.cors_allow_credentials),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    if settings.security_headers_enabled:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if _should_secure_cookie(request):
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


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


# ==================================================# 11) AUTH PAGES
# ==================================================
@app.get("/download", response_class=HTMLResponse)
async def download_page():
    local_apk = _find_local_apk()
    apk_url = "/download/apk" if local_apk else APK_DOWNLOAD_URL
    return HTMLResponse(
        render_template(
            "download.html",
            apk_url=apk_url,
            app_name=MOBILE_APP_NAME,
            app_version=MOBILE_APP_VERSION,
            backend_url=MOBILE_DEFAULT_SERVER_URL,
            is_local_apk=bool(local_apk),
        )
    )


@app.get("/download/apk")
async def download_apk():
    local_apk = _find_local_apk()
    if local_apk is None:
        return RedirectResponse(url=APK_DOWNLOAD_URL, status_code=307)
    return FileResponse(
        path=local_apk,
        media_type="application/vnd.android.package-archive",
        filename=local_apk.name,
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = _try_get_user_from_request(request)
    if user:
        return RedirectResponse(url="/admin" if user.get("role") == "admin" else "/dashboard")
    return RedirectResponse(url="/login")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = _try_get_user_from_request(request)
    if user:
        return RedirectResponse(url="/admin" if user.get("role") == "admin" else "/dashboard")
    return HTMLResponse(render_template("login.html", error=None))

def _wants_json(request: Request) -> bool:
    accept = (request.headers.get("accept") or "").lower()
    if "application/json" in accept:
        return True
    if (request.headers.get("x-client") or "").lower() == "mobile":
        return True
    if (request.query_params.get("format") or "").lower() == "json":
        return True
    return False


@app.post("/login")
async def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    _guard_login(request)
    user = db_get_user_by_username(engine, username)
    if (not user) or (not verify_password(password, user["password_hash"])):
        if _wants_json(request):
            return JSONResponse({"ok": False, "error": "Username atau password salah"}, status_code=401)
        return HTMLResponse(render_template("login.html", error="Username atau password salah"))

    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})

    if _wants_json(request):
        response = JSONResponse(
            {
                "ok": True,
                "access_token": access_token,
                "username": user["username"],
                "role": user["role"],
            }
        )
        secure_cookie = _should_secure_cookie(request)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            samesite=_cookie_samesite(),
            secure=secure_cookie,
            max_age=int(settings.token_expire_minutes) * 60,
        )
        return response

    redirect_url = "/admin" if user["role"] == "admin" else "/dashboard"
    response = RedirectResponse(url=redirect_url, status_code=303)

    secure_cookie = _should_secure_cookie(request)
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite=_cookie_samesite(),
        secure=secure_cookie,
        max_age=int(settings.token_expire_minutes) * 60,
    )
    return response


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    user = _try_get_user_from_request(request)
    if user:
        return RedirectResponse(url="/admin" if user.get("role") == "admin" else "/dashboard")
    return HTMLResponse(render_template("register.html", error=None, register_requires_code=_get_register_requires_code()))


@app.post("/register")
async def register_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    code: Optional[str] = Form(None),
):
    _guard_register(request)
    errors = []

    if password != confirm_password:
        errors.append("Password dan konfirmasi password tidak sama")

    code = (code or "").strip()
    role = "user"
    requires_code = _get_register_requires_code()
    claim_code: Optional[str] = None

    if requires_code or code:
        v = db_validate_register_code(engine, code)
        if not v.get("ok"):
            errors.append(v.get("message") or "Kode tidak valid")
        else:
            if v.get("kind") == "admin":
                role = "admin"
            else:
                if not v.get("available"):
                    errors.append(v.get("message") or "Kode sudah dipakai")
                else:
                    claim_code = code

    if errors:
        if _wants_json(request):
            return JSONResponse({"ok": False, "error": "; ".join(errors)}, status_code=400)
        return HTMLResponse(render_template("register.html", error="; ".join(errors), register_requires_code=_get_register_requires_code()))

    try:
        user = db_create_user(engine, username, password, role)

        if role == "user" and claim_code:
            try:
                db_claim_mcp_code(engine, claim_code, int(user["id"]))
            except Exception as e:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM users WHERE id=:id"), {"id": int(user["id"])})
                if _wants_json(request):
                    return JSONResponse({"ok": False, "error": f"Gagal claim kode: {e}"}, status_code=400)
                return HTMLResponse(render_template("register.html", error=f"Gagal claim kode: {e}", register_requires_code=_get_register_requires_code()))

        access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
        if _wants_json(request):
            response = JSONResponse(
                {
                    "ok": True,
                    "access_token": access_token,
                    "username": user["username"],
                    "role": user["role"],
                }
            )
            secure_cookie = _should_secure_cookie(request)
            response.set_cookie(
                key="access_token",
                value=access_token,
                httponly=True,
                samesite=_cookie_samesite(),
                secure=secure_cookie,
                max_age=int(settings.token_expire_minutes) * 60,
            )
            return response

        redirect_url = "/admin" if role == "admin" else "/dashboard"
        response = RedirectResponse(url=redirect_url, status_code=303)

        secure_cookie = _should_secure_cookie(request)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            samesite=_cookie_samesite(),
            secure=secure_cookie,
            max_age=int(settings.token_expire_minutes) * 60,
        )
        return response

    except ValueError as e:
        if _wants_json(request):
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        return HTMLResponse(render_template("register.html", error=str(e), register_requires_code=_get_register_requires_code()))


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
    _guard_admin(request)
    err = request.query_params.get("err")
    ok = request.query_params.get("ok")
    codes = db_list_mcp_codes(engine)
    for c in codes:
        ca = c.get("created_at")
        if isinstance(ca, datetime):
            c["created_at"] = ca.strftime("%Y-%m-%d %H:%M:%S")
        else:
            c["created_at"] = str(ca)
        lo = c.get("last_ok_at")
        le = c.get("last_err_at")
        if isinstance(lo, datetime):
            c["last_ok_at"] = lo.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(le, datetime):
            c["last_err_at"] = le.strftime("%Y-%m-%d %H:%M:%S")

    # FIX: Fungsi ini sekarang sudah di-import dan ada di db.py
    users = db_list_users(engine)
    
    users = db_list_users(engine)
    for u in users:
        uc = u.get("created_at")
        uu = u.get("updated_at")
        if isinstance(uc, datetime):
            u["created_at"] = uc.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(uu, datetime):
            u["updated_at"] = uu.strftime("%Y-%m-%d %H:%M:%S")

        lok = u.get("last_ok_at")
        ler = u.get("last_err_at")
        if isinstance(lok, datetime):
            u["last_ok_at"] = lok.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(ler, datetime):
            u["last_err_at"] = ler.strftime("%Y-%m-%d %H:%M:%S")

    logs = db_list_admin_audit_logs(engine, limit=200)
    for l in logs:
        ca = l.get("created_at")
        if isinstance(ca, datetime):
            l["created_at"] = ca.strftime("%Y-%m-%d %H:%M:%S")

    register_requires_code = _get_register_requires_code()
    social_links = _get_social_links()
    social_links_json = json.dumps(social_links, ensure_ascii=False)

    html = render_template(
        "admin.html",
        username=current_admin["username"],
        codes=codes,
        users=users,
        logs=logs,
        register_requires_code=register_requires_code,
        social_links=social_links,
        social_links_json=social_links_json,
        err=err,
        ok=ok,
    )
    return HTMLResponse(html)


@app.post("/admin/codes")
async def admin_create_code(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
    token: str = Form(...),
):
    _guard_admin(request)
    try:
        created = db_create_mcp_code(engine, token)
        db_add_admin_audit_log(
            engine,
            admin_user=current_admin,
            action="code_create",
            target_type="auth_code",
            target_id=str(created.get("id") or ""),
            ip=_get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            details={"code": created.get("code"), "has_token": True},
        )
    except Exception as e:
        logger.warning(f"[ADMIN] create code error: {e}")
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/codes/update")
async def admin_update_code(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
    code_id: int = Form(...),
    token: str = Form(...),
):
    _guard_admin(request)
    try:
        _ = db_update_mcp_code(engine, int(code_id), token)
        db_add_admin_audit_log(
            engine,
            admin_user=current_admin,
            action="code_update",
            target_type="auth_code",
            target_id=str(code_id),
            ip=_get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            details={"token_updated": True},
        )
        stop_mcp_worker(int(code_id))
    except Exception as e:
        logger.warning(f"[ADMIN] update code error: {e}")
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/codes/update-code")
async def admin_update_code_value(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
    code_id: int = Form(...),
    code: str = Form(...),
):
    _guard_admin(request)
    try:
        updated = db_update_mcp_code_value(engine, int(code_id), code)
        db_add_admin_audit_log(
            engine,
            admin_user=current_admin,
            action="code_value_update",
            target_type="auth_code",
            target_id=str(code_id),
            ip=_get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            details={"new_code": updated.get("code")},
        )
    except Exception as e:
        logger.warning(f"[ADMIN] update code value error: {e}")
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/codes/delete")
async def admin_delete_code(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
    code_id: int = Form(...),
):
    _guard_admin(request)
    try:
        stop_mcp_worker(int(code_id))
        deleted = db_delete_mcp_code(engine, int(code_id))
        db_add_admin_audit_log(
            engine,
            admin_user=current_admin,
            action="code_delete",
            target_type="auth_code",
            target_id=str(code_id),
            ip=_get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            details={"code": deleted.get("code"), "used_by": deleted.get("used_by")},
        )
    except Exception as e:
        logger.warning(f"[ADMIN] delete code error: {e}")
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/codes/unclaim")
async def admin_unclaim_code(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
    code_id: int = Form(...),
):
    _guard_admin(request)
    try:
        stop_mcp_worker(int(code_id))
        res = db_unclaim_mcp_code(engine, int(code_id), delete_user=True)
        db_add_admin_audit_log(
            engine,
            admin_user=current_admin,
            action="code_unclaim",
            target_type="auth_code",
            target_id=str(code_id),
            ip=_get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            details={
                "code": res.get("code"),
                "user_deleted": bool(res.get("user_deleted")),
            },
        )
        user_deleted = res.get("user_deleted") if isinstance(res, dict) else None
        if isinstance(user_deleted, dict):
            for cid in user_deleted.get("code_ids", []) or []:
                stop_mcp_worker(int(cid))
    except Exception as e:
        logger.warning(f"[ADMIN] unclaim error: {e}")
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/users")
async def admin_create_user(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form("user"),
):
    _guard_admin(request)
    try:
        created = db_create_user(engine, username, password, role=role)
        db_add_admin_audit_log(
            engine,
            admin_user=current_admin,
            action="user_create",
            target_type="user",
            target_id=str(created.get("id") or ""),
            ip=_get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            details={"username": created.get("username"), "role": created.get("role")},
        )
    except Exception as e:
        logger.warning(f"[ADMIN] create user error: {e}")
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/users/update")
async def admin_update_user(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
    user_id: int = Form(...),
    role: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
):
    _guard_admin(request)
    try:
        updated = db_update_user(engine, int(user_id), password=password, role=role)
        db_add_admin_audit_log(
            engine,
            admin_user=current_admin,
            action="user_update",
            target_type="user",
            target_id=str(user_id),
            ip=_get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            details={"role": updated.get("role"), "password_changed": bool(password)},
        )
    except Exception as e:
        logger.warning(f"[ADMIN] update user error: {e}")
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/users/delete")
async def admin_delete_user(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
    user_id: int = Form(...),
):
    _guard_admin(request)
    try:
        res = db_delete_user(engine, int(user_id), delete_codes=True)
        for cid in res.get("code_ids", []) or []:
            stop_mcp_worker(int(cid))
        db_add_admin_audit_log(
            engine,
            admin_user=current_admin,
            action="user_delete",
            target_type="user",
            target_id=str(user_id),
            ip=_get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            details={
                "username": res.get("username"),
                "code_ids": res.get("code_ids"),
                "codes_deleted": res.get("codes_deleted"),
            },
        )
    except Exception as e:
        logger.warning(f"[ADMIN] delete user error: {e}")
        msg = urllib.parse.quote(str(e))
        return RedirectResponse(url=f"/admin?err={msg}", status_code=303)
    return RedirectResponse(url="/admin?ok=user_deleted", status_code=303)


# --- Admin Settings ---
@app.post("/admin/settings/register-code")
async def admin_set_register_code(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
    enabled: str = Form(...),
):
    _guard_admin(request)
    flag = "1" if str(enabled).strip().lower() in ("1", "true", "yes", "on") else "0"
    db_set_app_state(engine, "register_requires_code", flag)
    db_add_admin_audit_log(
        engine,
        admin_user=current_admin,
        action="register_code_toggle",
        target_type="settings",
        target_id="register_requires_code",
        ip=_get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
        details={"enabled": flag},
    )
    return RedirectResponse(url="/admin?ok=register_code_updated", status_code=303)


@app.post("/admin/settings/social-links")
async def admin_set_social_links(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
):
    _guard_admin(request)
    links: List[Dict[str, str]] = []

    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            payload = None
        raw_links = payload.get("links") if isinstance(payload, dict) else payload
        links = _clean_social_links(raw_links)
    else:
        form = await request.form()
        raw_json = (form.get("links_json") or "").strip()
        if raw_json:
            try:
                raw_links = json.loads(raw_json)
            except Exception:
                raw_links = []
            links = _clean_social_links(raw_links)

    _set_social_links(links)
    db_add_admin_audit_log(
        engine,
        admin_user=current_admin,
        action="social_links_update",
        target_type="settings",
        target_id="social_links",
        ip=_get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
        details={"count": len(links)},
    )

    if _wants_json(request):
        return {"ok": True, "count": len(links), "links": links}
    return RedirectResponse(url="/admin?ok=social_links_updated", status_code=303)

# ==================================================# 12) PUBLIC + PROTECTED API ENDPOINTS
# ==================================================
@app.get("/healthz")
async def healthz():
    return {"ok": True, "llm_provider": settings.llm_provider, "cleanup_enabled": settings.cleanup_enabled}


@app.get("/api/public/validate-code")
async def api_public_validate_code(code: str = Query(...)):
    try:
        return db_validate_register_code(engine, code)
    except Exception as e:
        return {"ok": False, "available": False, "message": str(e)}


@app.get("/api/public/settings")
async def api_public_settings():
    return {
        "register_requires_code": _get_register_requires_code(),
        "social_links": _get_social_links(),
    }


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
        "mode_psid_map": parse_mode_psid_map(settings.mode_psid_map),
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
    try:
        db_upsert_user_device(engine, int(current_user["id"]), device_id)
    except Exception:
        pass
    return db_get_active_mode_for_device(engine, device_id)


@app.post("/api/mode")
async def api_set_active(data: SetActiveIn, current_user: dict = Depends(get_current_user)):
    try:
        db_upsert_user_device(engine, int(current_user["id"]), data.device_id)
    except Exception:
        pass
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

    intro = data.introduction if data.introduction is not None else mode.get("introduction", "")
    title = data.title if data.title is not None else mode.get("title", "")
    if mode.get("name") == "device_live":
        intro_lower = intro.lower()
        if (
            (not intro.strip())
            or ("update 30 detik" in intro_lower)
            or ("snapshot maksimal 5 menit" in intro_lower)
            or ("real-time" in intro_lower)
        ):
            intro = (
                "PERAN: Asisten monitoring perangkat yang fokus pada data HP terbaru.\n"
                "KONTEKS PERANGKAT (snapshot maksimal 5 detik sekali):\n"
                "{device_status}\n\n"
                "ATURAN:\n"
                "- Jika user tanya 'lokasi saya', utamakan jawab lokasi perangkat utama milik user ini.\n"
                "- Jika user tanya lokasi device lain yang sudah dipairkan, sebut nama alias device lalu alamat detail terbarunya.\n"
                "- Jika ada beberapa device, bedakan dengan jelas pakai nama alias masing-masing.\n"
                "- Jika user tanya lokasi, jawab pakai alamat detail terbaru, bukan koordinat mentah.\n"
                "- Jika user tanya status HP (baterai/jaringan), jawab pakai data terbaru.\n"
                "- Jika data kosong/unknown, jelaskan belum ada data terbaru.\n"
                "- Jawab singkat, jelas, dan bisa dibacakan.\n"
            )
    merged_vars = dict(data.vars or {})
    merged_vars.setdefault("glossary", settings.translation_glossary or "")
    merged_vars.setdefault("dont_translate", settings.translation_do_not_translate or "")
    if data.device_id:
        status = db_get_device_status(engine, data.device_id) or {}
        merged_vars.setdefault("device_status", build_device_status_text(data.device_id, int(current_user["id"])))
        merged_vars.setdefault("device_latitude", status.get("latitude"))
        merged_vars.setdefault("device_longitude", status.get("longitude"))
        merged_vars.setdefault("device_address", status.get("address_full") or "")
        merged_vars.setdefault("device_location_detail", status.get("address_full") or "")
    rendered_intro = safe_format(intro, merged_vars)
    rendered_title = safe_format(title, merged_vars)

    return {
        "prompt": build_system_prompt(
            mode_name=mode.get("name", ""),
            mode_title=rendered_title,
            intro_text=rendered_intro,
        )
    }


@app.get("/api/device/settings")
async def api_get_device_settings(device_id: str = Query(...), current_user: dict = Depends(get_current_user)):
    try:
        db_upsert_user_device(engine, int(current_user["id"]), device_id)
    except Exception:
        pass
    did = effective_device_id(device_id)
    st = db_get_device_settings(engine, did)
    return {"device_id": did, "source": st["source"], "target": st["target"]}


@app.post("/api/device/settings")
async def api_set_device_settings(data: DeviceSettingsIn, current_user: dict = Depends(get_current_user)):
    try:
        db_upsert_user_device(engine, int(current_user["id"]), data.device_id)
    except Exception:
        pass
    result = db_set_device_settings(engine, data.device_id, data.source, data.target)
    await manager.broadcast({"type": "device_settings_changed", "device_id": result["device_id"], "timestamp": time.time()})
    return result


# ==================================================# Device Tracking API
# ==================================================
@app.post("/api/devices/register")
async def api_devices_register(
    request: Request,
    data: DeviceRegisterIn,
    current_user: dict = Depends(get_current_user),
):
    _require_mobile_request(request)
    device_id = normalize_device_id(data.device_id)
    if not device_id:
        return JSONResponse({"error": "device_id required"}, status_code=400)

    registry = db_get_device_registry(engine, device_id)
    token_plain = (data.device_token or "").strip() or None

    if token_plain and registry:
        if _hash_token(token_plain) != (registry.get("token_hash") or ""):
            token_plain = None

    if not token_plain:
        token_plain = secrets.token_urlsafe(32)

    token_hash = _hash_token(token_plain)
    token_prefix = token_plain[:10]
    db_upsert_device_registry(
        engine,
        device_id=device_id,
        owner_user_id=int(current_user["id"]),
        token_hash=token_hash,
        token_prefix=token_prefix,
        device_name=data.device_name,
        platform=data.platform,
        model=data.model,
        os_version=data.os_version,
    )

    try:
        existing_alias = db_get_user_device_alias(engine, int(current_user["id"]), device_id)
        if existing_alias:
            db_link_tracked_device(engine, int(current_user["id"]), device_id)
        else:
            db_link_tracked_device(engine, int(current_user["id"]), device_id, alias=data.device_name)
    except Exception:
        pass

    return {
        "device_id": device_id,
        "device_token": token_plain,
        "device_name": data.device_name or registry.get("device_name") if registry else data.device_name,
    }


@app.post("/api/devices/heartbeat")
async def api_devices_heartbeat(
    request: Request,
    data: DeviceHeartbeatIn,
    current_user: Optional[dict] = Depends(_try_get_user_from_request),
):
    _require_mobile_request(request)
    device_id = normalize_device_id(data.device_id)
    if not device_id or not data.device_token:
        return JSONResponse({"error": "device_id and device_token required"}, status_code=400)

    registry = db_get_device_registry(engine, device_id)
    if not registry:
        return JSONResponse({"error": "device not registered"}, status_code=404)
    if _hash_token(data.device_token) != (registry.get("token_hash") or ""):
        return JSONResponse({"error": "invalid device token"}, status_code=403)

    existing = db_get_device_status(engine, device_id) or {}
    lat = data.latitude if data.latitude is not None else existing.get("latitude")
    lon = data.longitude if data.longitude is not None else existing.get("longitude")

    address = {}
    if lat is not None and lon is not None:
        address = await reverse_geocode(lat, lon)

    status_payload = {
        "latitude": lat,
        "longitude": lon,
        "address_street": address.get("street") or existing.get("address_street"),
        "address_area": address.get("area") or existing.get("address_area"),
        "address_city": address.get("city") or existing.get("address_city"),
        "address_full": address.get("full") or existing.get("address_full"),
        "battery_level": data.battery_level if data.battery_level is not None else existing.get("battery_level"),
        "battery_status": data.battery_status or existing.get("battery_status"),
        "charging_type": data.charging_type or existing.get("charging_type"),
        "battery_temp": data.battery_temp if data.battery_temp is not None else existing.get("battery_temp"),
        "network_type": data.network_type or existing.get("network_type"),
        "signal_strength": data.signal_strength if data.signal_strength is not None else existing.get("signal_strength"),
        "carrier": data.carrier or existing.get("carrier"),
        "ram_used": data.ram_used if data.ram_used is not None else existing.get("ram_used"),
        "ram_total": data.ram_total if data.ram_total is not None else existing.get("ram_total"),
        "storage_used": data.storage_used if data.storage_used is not None else existing.get("storage_used"),
        "storage_total": data.storage_total if data.storage_total is not None else existing.get("storage_total"),
        "is_online": 1,
    }

    location_changed = _location_changed(existing, lat, lon)
    db_upsert_device_status(engine, device_id, status_payload)
    if location_changed:
        db_add_device_location_history(engine, device_id, status_payload)
    db_update_device_last_seen(engine, device_id)

    if current_user:
        try:
            db_link_tracked_device(engine, int(current_user["id"]), device_id)
        except Exception:
            pass

    await manager.broadcast({"type": "device_status_updated", "device_id": device_id, "timestamp": time.time()})
    return {
        "ok": True,
        "device_id": device_id,
        "address": {
            "street": status_payload.get("address_street") or "",
            "area": status_payload.get("address_area") or "",
            "city": status_payload.get("address_city") or "",
            "full": status_payload.get("address_full") or "",
        },
    }


@app.get("/api/devices")
async def api_devices_list(request: Request, current_user: dict = Depends(get_current_user)):
    _require_mobile_request(request)
    return db_list_devices_for_user(engine, int(current_user["id"]))


@app.get("/api/devices/{device_id}")
async def api_devices_detail(
    request: Request,
    device_id: str,
    current_user: dict = Depends(get_current_user),
):
    _require_mobile_request(request)
    did = normalize_device_id(device_id)
    devices = db_list_devices_for_user(engine, int(current_user["id"]))
    device = next((d for d in devices if d.get("device_id") == did), None)
    if not device:
        return JSONResponse({"error": "device not found"}, status_code=404)
    return device


@app.get("/api/devices/{device_id}/locations")
async def api_devices_location_history(
    request: Request,
    device_id: str,
    limit: int = Query(200, ge=1, le=1000),
    current_user: dict = Depends(get_current_user),
):
    _require_mobile_request(request)
    did = normalize_device_id(device_id)
    devices = db_list_devices_for_user(engine, int(current_user["id"]))
    device = next((d for d in devices if d.get("device_id") == did), None)
    if not device:
        return JSONResponse({"error": "device not found"}, status_code=404)
    return db_list_device_location_history(engine, did, limit)


@app.post("/api/devices/alias")
async def api_devices_alias(
    request: Request,
    data: DeviceAliasIn,
    current_user: dict = Depends(get_current_user),
):
    _require_mobile_request(request)
    device_id = normalize_device_id(data.device_id)
    reg = db_get_device_registry(engine, device_id) or {}
    alias = (data.alias or "").strip() or reg.get("device_name") or device_id
    db_set_user_device_alias(engine, int(current_user["id"]), device_id, alias)
    return {"ok": True, "alias": alias}


@app.post("/api/devices/unpair")
async def api_devices_unpair(
    request: Request,
    data: DeviceUnpairIn,
    current_user: dict = Depends(get_current_user),
):
    _require_mobile_request(request)
    device_id = normalize_device_id(data.device_id)
    owned_devices = set(db_list_user_devices(engine, int(current_user["id"])))
    if device_id in owned_devices:
        return JSONResponse(
            {"error": "device utama milik Anda tidak bisa dihapus dari monitor"},
            status_code=400,
        )
    db_unlink_tracked_device(engine, int(current_user["id"]), device_id)
    return {"ok": True, "device_id": device_id}


@app.post("/api/devices/pair-token")
async def api_devices_pair_token(
    request: Request,
    data: DevicePairTokenIn,
    current_user: dict = Depends(get_current_user),
):
    _require_mobile_request(request)
    device_id = normalize_device_id(data.device_id)
    registry = db_get_device_registry(engine, device_id)
    if not registry:
        return JSONResponse({"error": "device not registered"}, status_code=404)
    if _hash_token(data.device_token) != (registry.get("token_hash") or ""):
        return JSONResponse({"error": "invalid device token"}, status_code=403)

    pair_token = secrets.token_urlsafe(16)
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    db_create_pair_token(engine, device_id, _hash_token(pair_token), expires_at)
    return {"pair_token": pair_token, "expires_at": expires_at.isoformat() + "Z"}


@app.post("/api/devices/pair-claim")
async def api_devices_pair_claim(
    request: Request,
    data: DevicePairClaimIn,
    current_user: dict = Depends(get_current_user),
):
    _require_mobile_request(request)
    token = (data.pair_token or "").strip()
    if not token:
        return JSONResponse({"error": "pair_token required"}, status_code=400)
    device_id = db_claim_pair_token(engine, _hash_token(token))
    if not device_id:
        return JSONResponse({"error": "pair token invalid or expired"}, status_code=400)

    reg = db_get_device_registry(engine, device_id) or {}
    alias = db_get_user_device_alias(engine, int(current_user["id"]), device_id) or reg.get("device_name") or device_id
    db_link_tracked_device(engine, int(current_user["id"]), device_id, alias=alias)
    status = db_get_device_status(engine, device_id) or {}
    return {"ok": True, "device_id": device_id, "device": {**reg, **status, "alias": alias}}


@app.get("/api/chats")
async def api_list_chats(device_id: str = Query(...), current_user: dict = Depends(get_current_user)):
    try:
        db_upsert_user_device(engine, int(current_user["id"]), device_id)
    except Exception:
        pass
    return db_list_threads(engine, device_id)


@app.get("/api/chats/all")
async def api_chats_all(limit_per_device: int = Query(30), current_user: dict = Depends(get_current_user)):
    return db_list_threads_all_grouped(engine, limit_per_device=limit_per_device)


@app.post("/api/chats/new")
async def api_new_chat(data: ChatNewIn, current_user: dict = Depends(get_current_user)):
    try:
        db_upsert_user_device(engine, int(current_user["id"]), data.device_id)
    except Exception:
        pass
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
        try:
            db_upsert_user_device(engine, int(current_user["id"]), device_id)
        except Exception:
            pass
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

        try:
            db_upsert_user_device(engine, int(current_user["id"]), data.device_id)
        except Exception:
            pass
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

        full_msgs = db_get_messages_page(engine, device_id, data.thread_id, limit=200)
        llm_msgs: List[Dict[str, str]] = []
        for m in full_msgs[-60:]:
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
    codes = db_list_mcp_codes_for_user(engine, int(current_user["id"]))
    for c in codes:
        c.pop("token", None)
    return codes


@app.post("/api/mcp/test-token")
async def api_mcp_test_token(data: McpTokenTestIn, current_user: dict = Depends(get_current_user)):
    raw = normalize_ws_url(data.token)
    if not raw:
        return JSONResponse({"ok": False, "error": "Token/URL kosong"}, status_code=400)
    if not raw.startswith("ws"):
        return JSONResponse({"ok": False, "error": "Token/URL harus ws:// atau wss://"}, status_code=400)
    try:
        async with websockets.connect(raw, open_timeout=6, close_timeout=3, ping_interval=None) as ws:
            # attempt a simple RPC call (optional)
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}))
            await asyncio.wait_for(ws.recv(), timeout=6)
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/mcp/my-codes/create")
async def api_create_my_code(data: McpTokenIn, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "user":
        return JSONResponse({"error": "Only user allowed"}, status_code=403)
    try:
        created = db_create_mcp_code(engine, data.token)
        try:
            db_claim_mcp_code(engine, created.get("code") or "", int(current_user["id"]))
        except Exception as e:
            try:
                db_delete_mcp_code(engine, int(created.get("id") or 0))
            except Exception:
                pass
            raise e
        return {"ok": True, "code_id": created.get("id"), "code": created.get("code")}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/mcp/my-codes/update")
async def api_update_my_code(data: McpTokenUpdateIn, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "user":
        return JSONResponse({"error": "Only user allowed"}, status_code=403)
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, used_by FROM auth_codes WHERE id=:id"),
            {"id": int(data.code_id)},
        ).mappings().fetchone()
    if not row:
        return JSONResponse({"ok": False, "error": "Code tidak ditemukan"}, status_code=404)
    if int(row.get("used_by") or 0) != int(current_user["id"]):
        return JSONResponse({"ok": False, "error": "Tidak punya akses ke code ini"}, status_code=403)
    try:
        updated = db_update_mcp_code(engine, int(data.code_id), data.token)
        stop_mcp_worker(int(data.code_id))
        return {"ok": True, "code_id": updated.get("id"), "code": updated.get("code")}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/mcp/my-codes/clear")
async def api_clear_my_code_token(data: McpTokenClearIn, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "user":
        return JSONResponse({"error": "Only user allowed"}, status_code=403)
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, used_by FROM auth_codes WHERE id=:id"),
            {"id": int(data.code_id)},
        ).mappings().fetchone()
    if not row:
        return JSONResponse({"ok": False, "error": "Code tidak ditemukan"}, status_code=404)
    if int(row.get("used_by") or 0) != int(current_user["id"]):
        return JSONResponse({"ok": False, "error": "Tidak punya akses ke code ini"}, status_code=403)
    try:
        res = db_clear_mcp_code_token(engine, int(data.code_id))
        stop_mcp_worker(int(data.code_id))
        return {"ok": True, "code_id": res.get("code_id")}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/api/mcp/all-codes")
async def api_all_codes(request: Request, current_admin: dict = Depends(get_current_admin)):
    _guard_admin(request)
    return db_list_mcp_codes(engine)


@app.get("/api/admin/user-devices")
async def api_admin_user_devices(
    request: Request,
    user_id: int = Query(...),
    current_admin: dict = Depends(get_current_admin),
):
    _guard_admin(request)
    return {"user_id": int(user_id), "devices": db_list_user_devices(engine, int(user_id))}


@app.get("/api/admin/devices")
async def api_admin_devices(
    request: Request,
    limit: int = Query(200),
    current_admin: dict = Depends(get_current_admin),
):
    _guard_admin(request)
    return JSONResponse(
        {"error": "Monitoring device di website dimatikan. Gunakan aplikasi mobile."},
        status_code=403,
    )


@app.get("/api/admin/threads")
async def api_admin_threads(
    request: Request,
    device_id: str = Query(...),
    current_admin: dict = Depends(get_current_admin),
):
    _guard_admin(request)
    return db_list_threads_for_device_any(engine, device_id)


@app.get("/api/admin/messages")
async def api_admin_messages(
    request: Request,
    thread_id: int = Query(...),
    limit: int = Query(200),
    before_id: Optional[int] = Query(None),
    current_admin: dict = Depends(get_current_admin),
):
    _guard_admin(request)
    return db_get_messages_page_admin(engine, thread_id, limit=limit, before_id=before_id)


@app.get("/api/admin/metrics")
async def api_admin_metrics(
    request: Request,
    current_admin: dict = Depends(get_current_admin),
):
    _guard_admin(request)
    snap = _metrics_snapshot()
    snap["mcp_supervisor_enabled"] = bool(settings.mcp_supervisor_enabled)
    snap["mcp_workers"] = len(mcp_supervisor.tasks) if settings.mcp_supervisor_enabled else 0
    snap["ws_clients"] = len(manager.active_connections)
    snap["process_mem_mb"] = _process_mem_mb()
    snap["threads"] = threading.active_count()
    try:
        with engine.begin() as conn:
            snap["codes_total"] = int(conn.execute(text("SELECT COUNT(*) FROM auth_codes")).scalar() or 0)
            snap["codes_connected"] = int(conn.execute(text("SELECT COUNT(*) FROM mcp_conn_status WHERE is_connected=1")).scalar() or 0)
    except Exception:
        snap["codes_total"] = None
        snap["codes_connected"] = None
    return snap


# ==================================================# Entry (UPDATED FOR RAILWAY)
# ==================================================
if __name__ == "__main__":
    # FIX: Pastikan aplikasi berjalan pada port yang diberikan oleh Railway
    # dan menggunakan host 0.0.0.0 agar bisa diakses dari luar container
    port_str = os.environ.get("PORT")
    if port_str:
        port = int(port_str)
    else:
        port = int(settings.app_port) if settings.app_port else 8000
    
    host = "0.0.0.0"

    logger.info(
        f"Starting {settings.app_name} on {host}:{port} | "
        f"ENV={settings.app_env} | "
        f"LLM={settings.llm_provider} | FORCE_DEVICE_ID={(settings.force_device_id or '')!r} | "
        f"ADMIN_CODE_SET={bool((settings.admin_master_code or '').strip() or (settings.admin_master_code_hash or '').strip())}"
    )
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=bool(settings.app_debug),
        log_level=(settings.log_level or "info").lower(),
    )
