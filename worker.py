from __future__ import annotations

import asyncio
import sys

from db import (
    settings,
    init_db,
    cleanup_worker,
    engine,
    db_encrypt_tokens_if_needed,
    db_backfill_token_hash,
    _security_startup_warnings,
)
from mcp import mcp_supervisor


async def run_worker() -> None:
    init_db(engine)
    _security_startup_warnings()
    try:
        db_encrypt_tokens_if_needed(engine)
        db_backfill_token_hash(engine)
    except Exception as e:
        print(f"[SECURITY] token encryption migration failed: {e}")

    if not settings.mcp_supervisor_enabled:
        print("[WORKER] MCP_SUPERVISOR_ENABLED=false. Worker exiting.")
        return

    tasks = []
    if settings.cleanup_enabled:
        tasks.append(asyncio.create_task(cleanup_worker(engine)))

    tasks.append(asyncio.create_task(mcp_supervisor.run(engine, manager_ref=None)))
    print("[WORKER] MCP worker started.")

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        for t in tasks:
            t.cancel()
        try:
            await mcp_supervisor.shutdown()
        except Exception:
            pass


def main() -> int:
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
