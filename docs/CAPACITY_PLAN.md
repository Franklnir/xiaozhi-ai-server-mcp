# Capacity Plan (Pilot and Scale)

This plan uses a simple formula:

messages_per_minute = active_users * messages_per_user_per_minute
messages_per_second = messages_per_minute / 60

Each message triggers multiple DB writes (chat_messages insert, chat_threads update, summary update).
Expect 2-3 writes per message.

## Target Scenario (large scale)

Example: 5,000 active users * 15 msg/min = 75,000 msg/min (~1,250 msg/sec)
Estimated DB writes: 150,000 - 225,000 writes/min (2,500 - 3,750 writes/sec)
This requires multiple servers and a tuned DB. A mini-PC will not handle this.

## Pilot on Mini-PC (i3 gen10)

Recommended starting target:
- 200 to 400 active users
- 3 to 5 msg/min per user
- 600 to 2,000 msg/min total (10 to 33 msg/sec)
- DB writes ~ 20 to 100 writes/sec

This is realistic for a single i3 with local MySQL if tuned and not overloaded by other services.

If you must allow more users, reduce message rate or spread load across nodes.

## VPS Scale Plan (phased)

Phase 1:
- 1 API node + 1 MCP worker node
- Shared DB (separate server)
- Capacity: 1,000 to 2,000 active users

Phase 2:
- 2-4 API nodes + 2 MCP worker nodes
- Token sharding across workers
- Capacity: 3,000 to 6,000 active users

Phase 3:
- More worker nodes
- DB replica for reads
- Capacity: 10,000+ active users

## Pilot Mode Settings

To keep the system stable on a mini-PC:

- Enable MCP supervisor only once
  MCP_SUPERVISOR_ENABLED=true

- Limit retries and reconnect storms
  MCP_RECONNECT_DELAY=5
  MCP_MAX_RECONNECT_DELAY=60

- Keep cleanup on (remove old data)
  CLEANUP_ENABLED=true
  CLEANUP_DAYS=30

- Keep rate limits
  RL_LOGIN_PER_MINUTE=12
  RL_REGISTER_PER_MINUTE=6
  RL_ADMIN_PER_MINUTE=60

## Measuring Stability

Monitor:
- CPU usage (target under 70 percent)
- Memory usage (avoid swapping)
- MySQL slow queries
- WebSocket count (MCP tokens)
- DB writes per second
