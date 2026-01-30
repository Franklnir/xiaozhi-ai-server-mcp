# SciG Mode MCP - Architecture (Pilot -> Scale)

This document describes a scalable architecture for the FastAPI + MCP worker stack, starting from a single mini-PC pilot and scaling to VPS or multi-node production.

## Components

1) API/Web App (FastAPI)
- Serves dashboard and admin.
- Handles login, user settings, mode selection, and chat viewing.
- Writes chat logs to DB.
- No MCP WebSocket connections if MCP supervisor is disabled.

2) MCP Worker Service
- Maintains 1 WebSocket per MCP token.
- Receives tool calls from Xiaozhi and writes chat logs to DB.
- Should run as a dedicated process or node in production.

3) Database (MySQL)
- Stores users, auth codes, tokens, chats, summaries, and audit logs.
- Retention cleanup runs daily/hourly based on config.

4) Reverse Proxy / Load Balancer (production)
- Routes HTTP(S) traffic to API instances.
- Keeps WebSocket upgrade for /ws.

## Data Flow (chat)

User device -> Xiaozhi -> MCP Worker -> DB
Dashboard -> API -> DB -> Dashboard

## Scaling Strategy

Phase 1 (Pilot, single box)
- 1 process for API + MCP supervisor enabled.
- Local MySQL.
- Suitable for a small number of concurrent users.

Phase 2 (Split worker)
- API node: MCP supervisor disabled.
- Worker node: MCP supervisor enabled, API can be closed by firewall.
- Shared DB.

Phase 3 (Horizontal scale)
- Multiple API nodes behind a load balancer.
- One or more worker nodes, each owning a shard of tokens.
- DB on dedicated server or managed service.

## Key Toggles

- MCP_SUPERVISOR_ENABLED
  - true on MCP worker nodes
  - false on API nodes

- CLEANUP_ENABLED
  - enable on one node only to avoid duplicate cleanup work

## Notes

Avoid running multiple MCP supervisors against the same database without sharding, or the same token may be connected multiple times.
