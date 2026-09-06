"""Run interactively on the application server. Never accepts keys in argv."""

import getpass
import argparse
import os
from pathlib import Path
import re
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="Server-only environment file"
    )
    destination = parser.parse_args().output
    if os.name != "posix":
        raise SystemExit("Run on the POSIX application server to enforce mode 0600.")
    secret = getpass.getpass("DeepInfra API Key (hidden): ").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,200}", secret):
        raise SystemExit("Key 格式不正确；没有写入")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Only this isolated service reads this file. Do not replace the existing
    # shared OpenRouter credentials or expose this value in logs or CLI args.
    fd, temporary = tempfile.mkstemp(prefix=".deepinfra-", dir=destination.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("DEEPINFRA_API_KEY=" + secret + "\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("DeepInfra credential configured; permissions 0600; value not displayed.")


if __name__ == "__main__":
    main()
