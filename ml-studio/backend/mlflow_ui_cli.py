"""Launch MLflow UI with the same --backend-store-uri as the ML Studio API."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

from mlflow_tracking import configure_mlflow_tracking


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start MLflow UI using this project's tracking URI (SQLite or MLFLOW_TRACKING_URI)."
    )
    parser.add_argument("--host", default=os.environ.get("MLFLOW_UI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MLFLOW_UI_PORT", "5000")))
    args, passthrough = parser.parse_known_args()

    uri, is_remote = configure_mlflow_tracking()
    if is_remote:
        print(
            "Tracking URI is remote (HTTP). Open your provider's MLflow UI instead of running this command.",
            file=sys.stderr,
        )
        print(f"URI: {uri}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        uri,
        "--host",
        args.host,
        "--port",
        str(args.port),
        *passthrough,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
