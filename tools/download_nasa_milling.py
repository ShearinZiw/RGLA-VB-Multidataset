"""Download the checksum-pinned NASA Milling Parquet transport snapshot."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path


URL = "https://huggingface.co/datasets/jonasmaltebecker/nasa_milling/resolve/main/data.parquet"
SHA256 = "04819835e2747b9951a0d4415f5d0ce9d339ae6a0985ba3d6098c0b69116b1be"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "nasa_milling" / "data.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.force:
        digest = sha256(output)
        if digest == SHA256:
            print(f"Already verified: {output}")
            return
        raise ValueError(f"Existing file has unexpected SHA-256: {digest}; use --force to replace it")

    partial = output.with_suffix(output.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(URL, headers={"User-Agent": "RGLA-VB/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        digest = sha256(partial)
        if digest != SHA256:
            raise ValueError(f"Downloaded file has unexpected SHA-256: {digest}")
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()
    print(f"Downloaded and verified: {output}")


if __name__ == "__main__":
    main()
