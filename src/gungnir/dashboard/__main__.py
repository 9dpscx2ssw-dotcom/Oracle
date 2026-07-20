"""Run the dashboard:  python -m gungnir.dashboard [--host 0.0.0.0] [--port 8080]"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Odin monitoring dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    import logging
    import uvicorn

    # Keep API keys passed as query params (e.g. FRED) out of request logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    uvicorn.run("gungnir.dashboard.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
