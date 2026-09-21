"""Download and validate the official Open-D LST20 ZIP locally.

This helper downloads from the official NECTEC Open-D resource. It never uploads,
commits, or redistributes corpus files. The repository's .gitignore excludes
data/ by default.

Usage:
    python scripts/fetch_lst20_opend.py

If the Open-D route is slow from your network, download the same official ZIP in
a browser and use --zip-path to validate/extract the local file instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

OPEN_D_URL = (
    "https://opend-portal.nectec.or.th/dataset/"
    "d1364791-84bc-4b65-9904-79aa0aa2c5a6/resource/"
    "063e2392-1eba-4099-a732-fbaf1ba9a293/download/"
    "opend_lst20_corpus.zip"
)
EXPECTED_SIZE = 16_117_011
DEFAULT_ZIP = Path("data/opend_lst20_corpus.zip")
DEFAULT_EXTRACT_DIR = Path("data/lst20_opend")


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{value} B"


def download_official_zip(url: str, destination: Path, timeout: int = 60) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 Thai-NER-2026-Modernization "
                "(research/open-source benchmark)"
            )
        },
    )

    print(f"Downloading official Open-D LST20 resource to {destination}")
    started = time.monotonic()
    downloaded = 0
    last_report = 0

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                remote_size = int(content_length)
                print(f"Server Content-Length: {human_bytes(remote_size)}")
                if remote_size != EXPECTED_SIZE:
                    print(
                        "Warning: current server size differs from the size "
                        f"recorded when this helper was written ({EXPECTED_SIZE} bytes)."
                    )

            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)

                    if downloaded - last_report >= 4 * 1024 * 1024:
                        elapsed = max(time.monotonic() - started, 0.001)
                        speed = downloaded / elapsed
                        print(
                            f"  {human_bytes(downloaded)} downloaded "
                            f"({human_bytes(int(speed))}/s)"
                        )
                        last_report = downloaded
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    temporary.replace(destination)
    print(f"Downloaded {human_bytes(downloaded)}")


def validate_zip(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    size = path.stat().st_size
    print(f"ZIP size: {size:,} bytes ({human_bytes(size)})")
    if size != EXPECTED_SIZE:
        print(
            "Warning: file size differs from the currently documented Open-D "
            f"resource size ({EXPECTED_SIZE:,} bytes)."
        )

    with path.open("rb") as handle:
        if handle.read(2) != b"PK":
            raise ValueError("File does not start with ZIP magic bytes (PK).")

    with zipfile.ZipFile(path) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise ValueError(f"ZIP integrity check failed at member: {broken}")
        print(f"ZIP integrity OK: {len(archive.infolist())} members")


def safe_extract(path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()

    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (extract_dir / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        archive.extractall(extract_dir)

    print(f"Extracted to {extract_dir}")


def find_corpus_root(extract_dir: Path) -> Path:
    candidates = [extract_dir, *(p for p in extract_dir.rglob("*") if p.is_dir())]
    for candidate in sorted(candidates, key=lambda p: len(p.parts)):
        subdirs = {p.name.lower(): p for p in candidate.iterdir() if p.is_dir()}
        if {"train", "eval", "test"}.issubset(subdirs):
            return candidate
    raise FileNotFoundError(
        "Could not find a directory containing train/, eval/, and test/."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/validate/extract official NECTEC Open-D LST20 locally."
    )
    parser.add_argument("--url", default=OPEN_D_URL)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument(
        "--existing-zip",
        action="store_true",
        help="Skip network download and validate/extract --zip-path.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and validate the ZIP without extracting it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        "LST20 usage reminder: follow NECTEC's applicable license/usage terms. "
        "Do not commit or redistribute the corpus through this repository."
    )

    if not args.existing_zip:
        download_official_zip(args.url, args.zip_path)

    validate_zip(args.zip_path)

    if args.download_only:
        return

    safe_extract(args.zip_path, args.extract_dir)
    corpus_root = find_corpus_root(args.extract_dir)

    print("\nLST20 corpus root:")
    print(corpus_root)
    print("\nNext step:")
    print(f'thai-ner-audit --data-dir "{corpus_root}"')
    print(
        f'thai-ner-train --data-dir "{corpus_root}" '
        '--smoke-limit 100 --epochs 1 --output-dir outputs/smoke-2026'
    )


if __name__ == "__main__":
    main()
