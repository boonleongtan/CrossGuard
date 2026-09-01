#!/usr/bin/env python
"""Download WildFake archives from ModelScope to local disk (parallel).

    python scripts/download_wildfake.py                        # all archives
    python scripts/download_wildfake.py --group gan             # GAN_based only
    python scripts/download_wildfake.py --group diffusion       # Diffusion_based only
    python scripts/download_wildfake.py --group other           # Other_based only
    python scripts/download_wildfake.py --group reals           # real archives only
    python scripts/download_wildfake.py --out-dir /path/to/dir  # custom output
    python scripts/download_wildfake.py --workers 4             # parallel downloads

Resumable: tracks completed downloads in a checkpoint file. Interrupted
downloads resume from where they left off (HTTP Range header). Safe to
ctrl-C and restart.
"""
from __future__ import annotations

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "hy2628982280/WildFake"
BASE_URL = ("https://modelscope.cn/api/v1/datasets/{repo}/repo"
            "?Revision=master&FilePath={path}")

ARCHIVES = {
    "gan": [
        "Images/GAN_based.zip",
    ],
    "diffusion": [
        "Images/Diffusion_based/ADM.zip",
        "Images/Diffusion_based/DALLE.zip",
        "Images/Diffusion_based/DDIM.zip",
        "Images/Diffusion_based/DDPM.zip",
        "Images/Diffusion_based/Imagen.zip",
        "Images/Diffusion_based/VQDM.zip",
        "Images/Diffusion_based/SD/personalizedSD.zip",
        "Images/Diffusion_based/SD/SDwithAdaptor.zip",
        *[f"Images/Diffusion_based/SD/originalSD/Typical/part_{i}.zip" for i in range(1, 4)],
        *[f"Images/Diffusion_based/SD/originalSD/Advanced/part_{i}.zip" for i in range(1, 8)],
        *[f"Images/Diffusion_based/Midjourney/Advanced/part_{i}.zip" for i in range(1, 8)],
        *[f"Images/Diffusion_based/Midjourney/Typical/part_{i}.zip" for i in range(1, 5)],
    ],
    "other": [
        "Images/Other_based.zip",
    ],
    "reals": [
        "Images/Real/church.zip",
        "Images/Real/laion5b.zip",
        "Images/Real/ffhq.zip",
        "Images/Real/celebahq.zip",
        "Images/Real/afhq.zip",
        "Images/Real/imagenet.zip",
        "Images/Real/coco.zip",
    ],
}

CHUNK = 4 * 1024 * 1024  # 4 MB per read

_print_lock = threading.Lock()
_ckpt_lock = threading.Lock()


def _log(tag: str, msg: str) -> None:
    with _print_lock:
        print(f"[{tag}] {msg}", flush=True)


def _url(path: str) -> str:
    return BASE_URL.format(repo=REPO, path=path)


def _sizeof_fmt(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def _save_checkpoint(ckpt_file: Path, done: set[str]) -> None:
    with _ckpt_lock:
        ckpt_file.write_text(json.dumps(sorted(done), indent=2))


def download_one(path: str, out_dir: Path, done: set[str],
                 ckpt_file: Path) -> bool:
    """Download one archive with resume support. Returns True if newly completed."""
    import requests

    tag = path.split("/")[-1]
    session = requests.Session()
    session.headers["User-Agent"] = "CrossGuard-Downloader/1.0"

    dest = out_dir / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    url = _url(path)

    for attempt in range(5):
        try:
            head = session.head(url, allow_redirects=True, timeout=30)
            head.raise_for_status()
            total = int(head.headers["content-length"])
            break
        except Exception as e:
            if attempt == 4:
                _log(tag, f"FAILED to get size after 5 attempts: {e}")
                return False
            _log(tag, f"HEAD failed ({e}), retry {attempt+1}/4...")
            time.sleep(5 * (2 ** attempt))

    existing = tmp.stat().st_size if tmp.exists() else 0

    if dest.exists() and dest.stat().st_size == total:
        _log(tag, f"already complete ({_sizeof_fmt(total)})")
        return False

    if existing >= total:
        tmp.rename(dest)
        _log(tag, f"part file complete, renamed ({_sizeof_fmt(total)})")
        return True

    headers = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        _log(tag, f"resuming from {_sizeof_fmt(existing)} / {_sizeof_fmt(total)}")
    else:
        _log(tag, f"downloading {_sizeof_fmt(total)}")

    start = time.time()
    downloaded = existing
    last_report = start

    for attempt in range(5):
        try:
            r = session.get(url, headers=headers, stream=True,
                            allow_redirects=True, timeout=60)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 4:
                _log(tag, f"FAILED to start download after 5 attempts: {e}")
                return False
            _log(tag, f"GET failed ({e}), retry {attempt+1}/4...")
            time.sleep(5 * (2 ** attempt))

    mode = "ab" if existing > 0 else "wb"
    try:
        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_report >= 10:
                    elapsed = now - start
                    speed = (downloaded - existing) / elapsed if elapsed > 0 else 0
                    eta = (total - downloaded) / speed if speed > 0 else 0
                    pct = downloaded / total * 100
                    _log(tag, f"{pct:.1f}%  {_sizeof_fmt(downloaded)}/{_sizeof_fmt(total)}  "
                         f"{_sizeof_fmt(speed)}/s  ETA {int(eta)}s")
                    last_report = now
    except (requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            OSError) as e:
        _log(tag, f"interrupted at {_sizeof_fmt(downloaded)}: {e}")
        return False

    if downloaded >= total:
        tmp.rename(dest)
        elapsed = time.time() - start
        speed = (downloaded - existing) / elapsed if elapsed > 0 else 0
        _log(tag, f"done in {int(elapsed)}s ({_sizeof_fmt(speed)}/s)")
        done.add(path)
        _save_checkpoint(ckpt_file, done)
        return True
    else:
        _log(tag, f"incomplete: got {downloaded}/{total}")
        return False


def main():
    import argparse
    p = argparse.ArgumentParser(description="Download WildFake from ModelScope")
    p.add_argument("--group", choices=["gan", "diffusion", "other", "reals", "all"],
                   default="all", help="which archive group to download")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "wildfake_zips",
                   help="output directory")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel download threads (default: 4)")
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ckpt_file = out / ".download_checkpoint.json"

    if ckpt_file.exists():
        done = set(json.loads(ckpt_file.read_text()))
    else:
        done = set()

    if args.group == "all":
        archives = [a for group in ARCHIVES.values() for a in group]
    else:
        archives = ARCHIVES[args.group]

    pending = [a for a in archives if a not in done]
    print(f"{len(archives)} archives in group '{args.group}', "
          f"{len(done)} already done, {len(pending)} remaining")
    print(f"using {args.workers} parallel workers\n")

    if not pending:
        print("nothing to download")
        return

    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, path, out, done, ckpt_file): path
            for path in pending
        }
        try:
            for fut in as_completed(futures):
                path = futures[fut]
                tag = path.split("/")[-1]
                try:
                    ok = fut.result()
                    if ok:
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    _log(tag, f"unexpected error: {e}")
                    failed += 1
        except KeyboardInterrupt:
            print("\nCtrl-C — shutting down. Run again to resume.")
            pool.shutdown(wait=False, cancel_futures=True)
            return

    total_done = len([a for a in archives if a in done])
    print(f"\n{total_done}/{len(archives)} archives complete "
          f"(+{completed} this run, {failed} failed)")
    if total_done < len(archives):
        missing = [a for a in archives if a not in done]
        print("missing:")
        for m in missing:
            print(f"  {m}")
        print("\nrun again to retry")


if __name__ == "__main__":
    main()
