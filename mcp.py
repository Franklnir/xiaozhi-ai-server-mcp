# mcp.py
from __future__ import annotations

import asyncio
import json
import random
import time
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import websockets
from sqlalchemy.engine import Engine

from db import (
    settings,
    logger,
    DEFAULT_BUCKET,
    _mask_ws_url,
    normalize_ws_url,
    decrypt_token,
    _retention_days,
    build_recent_chat_context,
    effective_device_id,
    normalize_device_id,
    resolve_psid_for_incoming,
    db_set_app_state,
    db_get_app_state,
    db_set_route_psid,
    db_get_active_mode_for_device,
    db_get_device_settings,
    safe_format,
    mode_needs_lang,
    build_system_prompt,
    db_list_modes,
    db_set_active_mode_for_device,
    db_upsert_mode,
    db_delete_mode,
    build_role_introduction_for_xiaozhi,
    db_add_message,
    db_try_set_thread_title_from_first_user,
    db_get_thread_owner_device,
    db_get_or_create_latest_thread_id,
    db_list_owned_tokens,
    db_upsert_conn_status_ok,
    db_upsert_conn_status_err,
    maybe_roll_summary,
)
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
    MCPTool(
        "delete_mode",
        "Delete mode by id or name (default modes are protected).",
        {
            "type": "object",
            "properties": {"mode_id": {"type": "integer"}, "name": {"type": "string"}},
        },
    ),
    MCPTool(
        "get_role_introduction",
        "Get text for Xiaozhi role config. Optional args: style=strict|natural",
        {
            "type": "object",
            "properties": {
                "style": {"type": "string", "enum": ["strict", "natural"]},
            },
        },
    ),
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
    vars_dict.setdefault("glossary", settings.translation_glossary or "")
    vars_dict.setdefault("dont_translate", settings.translation_do_not_translate or "")

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
        vars_dict.setdefault("glossary", settings.translation_glossary or "")
        vars_dict.setdefault("dont_translate", settings.translation_do_not_translate or "")
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
            days = _retention_days()
            history = build_recent_chat_context(
                engine,
                device_id,
                days=days,
                limit_messages=80,
                max_chars=4000,
            )
            if history:
                prompt = prompt + f"\n\nCONTEXT (last {days} days):\n" + history
        except Exception:
            pass

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
        dev_in = (args.get("device_id") or "").strip()
        if not dev_in:
            st = db_get_app_state(engine, "last_device_id")
            dev_in = (st.get("v") if st else "") or ""
        if not dev_in:
            st = db_get_app_state(engine, "active_psid")
            dev_in = (st.get("v") if st else "") or DEFAULT_BUCKET

        res = db_set_active_mode_for_device(engine, dev_in, mode_id=mid, name=args.get("name"))
        psid = effective_device_id(dev_in)

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
                if manager_ref is not None:
                    await manager_ref.broadcast({
                        "type": "route_changed",
                        "physical_device_id": normalize_device_id(last_phys),
                        "psid": psid,
                        "timestamp": time.time()
                    })
            except Exception:
                pass

        if manager_ref is not None:
            try:
                await manager_ref.broadcast({
                    "type": "active_mode_changed",
                    "device_id": psid,
                    "mode_id": res.get("id"),
                    "timestamp": time.time()
                })
            except Exception:
                pass

        return _mcp_text_result(f"Active mode for {psid} changed to: {res['title']}")

    if name == "upsert_mode":
        res = db_upsert_mode(engine, args["name"], args["title"], args["introduction"])
        if manager_ref is not None:
            try:
                await manager_ref.broadcast({"type": "modes_updated", "timestamp": time.time()})
            except Exception:
                pass
        return _mcp_text_result(res)

    if name == "delete_mode":
        res = db_delete_mode(
            engine,
            mode_id=(int(args["mode_id"]) if args.get("mode_id") is not None else None),
            mode_name=(args.get("name") or None),
        )
        if manager_ref is not None:
            try:
                await manager_ref.broadcast({"type": "modes_updated", "timestamp": time.time()})
            except Exception:
                pass
        return _mcp_text_result(res)

    if name == "get_role_introduction":
        return _mcp_text_result(build_role_introduction_for_xiaozhi(style=str(args.get("style") or "strict")))

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
    url = normalize_ws_url(ws_url)
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
                logger.info(f"[MCP:{code}] Connected")
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
        except socket.gaierror as e:
            # DNS resolution failure; back off harder to avoid noisy logs
            msg = f"DNS error while connecting: {e}"
            delay = max(300, int(settings.mcp_max_reconnect_delay))
            max_delay = max(max_delay, delay)
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
                        code = (r.get("code") or str(cid)).strip()
                        raw_token = r.get("token") or ""
                        try:
                            ws_url = decrypt_token(raw_token)
                        except Exception as e:
                            msg = f"Token decrypt failed: {e}"
                            logger.warning(f"[MCP_SUP] {msg} (code={code})")
                            db_upsert_conn_status_err(engine, cid, msg)
                            continue
                        ws_url = normalize_ws_url(ws_url)
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


def stop_mcp_worker(code_id: int) -> None:
    try:
        cid = int(code_id)
    except Exception:
        return
    task = mcp_supervisor.tasks.pop(cid, None)
    if task:
        task.cancel()


# =========================================================
