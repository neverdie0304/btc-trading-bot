"""대시보드 서버 실행 진입점.

사용법:
    python main_dashboard.py
    python main_dashboard.py --port 8080
"""

import argparse
import logging
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/dashboard.log"),
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Bot Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="서버 포트 (기본값: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트 (기본값: 0.0.0.0)")
    args = parser.parse_args()

    print(f"\n  Trading Bot Dashboard")
    print(f"  http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(
        "dashboard.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
