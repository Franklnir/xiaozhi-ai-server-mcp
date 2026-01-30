from __future__ import annotations

import argparse
import threading
import time
import random
from typing import Optional

import requests


def login(session: requests.Session, base_url: str, username: str, password: str) -> None:
    resp = session.post(
        f"{base_url}/login",
        data={"username": username, "password": password},
        timeout=10,
        allow_redirects=False,
    )
    if resp.status_code not in (302, 303):
        raise RuntimeError(f"Login failed: {resp.status_code}")


def worker(
    stop: threading.Event,
    base_url: str,
    mode: str,
    username: Optional[str],
    password: Optional[str],
    msgs_per_min: int,
    stats: dict,
) -> None:
    session = requests.Session()
    if mode in ("config", "chat"):
        if not username or not password:
            raise RuntimeError("username/password required for this mode")
        login(session, base_url, username, password)

    interval = 60.0 / max(1, msgs_per_min)
    next_t = time.time()

    while not stop.is_set():
        next_t += interval
        try:
            if mode == "health":
                r = session.get(f"{base_url}/healthz", timeout=10)
            elif mode == "config":
                r = session.get(f"{base_url}/api/config", timeout=10)
            else:
                # chat mode: create thread and post message
                r = session.post(
                    f"{base_url}/api/chats/new",
                    json={"device_id": "default"},
                    timeout=10,
                )
                if r.status_code == 200:
                    tid = r.json().get("id")
                    if tid:
                        _ = session.post(
                            f"{base_url}/api/chats/send",
                            json={
                                "device_id": "default",
                                "thread_id": int(tid),
                                "message": "load_test " + str(random.randint(1, 9999)),
                            },
                            timeout=10,
                        )
            with stats["lock"]:
                stats["total"] += 1
                if r.status_code >= 400:
                    stats["err"] += 1
        except Exception:
            with stats["lock"]:
                stats["total"] += 1
                stats["err"] += 1

        sleep_for = next_t - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", choices=["health", "config", "chat"], default="health")
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--rate", type=int, default=20, help="messages per minute per user")
    ap.add_argument("--duration", type=int, default=60, help="seconds")
    ap.add_argument("--username")
    ap.add_argument("--password")
    args = ap.parse_args()

    stop = threading.Event()
    stats = {"total": 0, "err": 0, "lock": threading.Lock()}

    threads = []
    for _ in range(max(1, args.users)):
        t = threading.Thread(
            target=worker,
            args=(
                stop,
                args.base_url.rstrip("/"),
                args.mode,
                args.username,
                args.password,
                args.rate,
                stats,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    time.sleep(max(1, args.duration))
    stop.set()
    for t in threads:
        t.join(timeout=2)

    total = stats["total"]
    err = stats["err"]
    rps = total / max(1, args.duration)
    print(f"total={total} err={err} rps={rps:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
