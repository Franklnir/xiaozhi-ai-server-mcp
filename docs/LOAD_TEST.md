# Load Test (simple)

This is a lightweight script to estimate capacity.

## Install

`requests` is already in requirements.

## Usage

Health endpoint (no login):
```
python tools/load_test.py --base-url http://127.0.0.1:8000 --mode health --users 10 --rate 20 --duration 60
```

Config endpoint (requires login):
```
python tools/load_test.py --base-url http://127.0.0.1:8000 --mode config --users 10 --rate 20 --duration 60 --username youruser --password yourpass
```

Chat endpoint (requires login, will create new threads):
```
python tools/load_test.py --base-url http://127.0.0.1:8000 --mode chat --users 10 --rate 20 --duration 60 --username youruser --password yourpass
```

## Notes

- Chat mode creates threads and sends messages.
- Use small values in production testing.
