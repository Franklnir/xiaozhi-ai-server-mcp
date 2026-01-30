# main.py
from __future__ import annotations

import asyncio
import threading
import time
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import (
    FastAPI,
    Query,
    WebSocket,
    WebSocketDisconnect,
    Request,
    Form,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field
from sqlalchemy import text

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
    normalize_device_id,
    canonical_device_bucket,
    effective_device_id,
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
    db_list_threads_for_device_any,
    db_get_messages_page_admin,
)
from mcp import mcp_supervisor, stop_mcp_worker, _mcp_text_result

# =========================================================
# 10) FASTAPI APP + TEMPLATE ENGINE
# =========================================================


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


# =========================================================
# 11) AUTH PAGES
# =========================================================

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


@app.post("/login")
async def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    _guard_login(request)
    user = db_get_user_by_username(engine, username)
    if (not user) or (not verify_password(password, user["password_hash"])):
        return HTMLResponse(render_template("login.html", error="Username atau password salah"))

    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})

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
    return HTMLResponse(render_template("register.html", error=None))


@app.post("/register")
async def register_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    code: str = Form(...),
):
    _guard_register(request)
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
        return HTMLResponse(render_template("register.html", error="; ".join(errors)))

    try:
        user = db_create_user(engine, username, password, role)

        if role == "user":
            try:
                db_claim_mcp_code(engine, code, int(user["id"]))
            except Exception as e:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM users WHERE id=:id"), {"id": int(user["id"])})
                return HTMLResponse(render_template("register.html", error=f"Gagal claim kode: {e}"))

        access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
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
        return HTMLResponse(render_template("register.html", error=str(e)))


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

    html = render_template(
        "admin.html",
        username=current_admin["username"],
        codes=codes,
        users=users,
        logs=logs,
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

    intro = mode.get("introduction", "")
    title = mode.get("title", "")
    merged_vars = dict(data.vars or {})
    merged_vars.setdefault("glossary", settings.translation_glossary or "")
    merged_vars.setdefault("dont_translate", settings.translation_do_not_translate or "")
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


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":
    logger.info(
        f"Starting {settings.app_name} on {settings.app_host}:{settings.app_port} | "
        f"ENV={settings.app_env} | "
        f"LLM={settings.llm_provider} | FORCE_DEVICE_ID={(settings.force_device_id or '')!r} | "
        f"ADMIN_CODE_SET={bool((settings.admin_master_code or '').strip() or (settings.admin_master_code_hash or '').strip())}"
    )
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=bool(settings.app_debug),
        log_level=(settings.log_level or "info").lower(),
    )
