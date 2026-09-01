"""Read a member out of a remote zip without downloading the archive.

WildFake ships as zip archives of 13-372 GB on ModelScope. Downloading one to
sample a few thousand images is absurd, and ModelScope serves HTTP range
requests, so we do not have to: read the central directory (a few hundred KB at
the end of the file), then fetch only the byte ranges of the members we want.
Sampling 5,000 images from the 372 GB Midjourney archive costs 5,000 images of
bandwidth.

Why not the `remotezip` package: it seeks with SUFFIX ranges (`Range:
bytes=-65536`) to find the end-of-central-directory record, and ModelScope
answers those with 400. Every seek here is resolved against a known
content-length and issued as an explicit `bytes=start-end`, which it accepts.
"""
from __future__ import annotations

import io
import zipfile

DATASET_URL = ("https://modelscope.cn/api/v1/datasets/{repo}/repo"
               "?Revision={revision}&FilePath={path}")


class RangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP, using explicit byte ranges only."""

    def __init__(self, url: str, session=None, timeout: int = 180):
        import requests
        self.url = url
        self.timeout = timeout
        # One session across every read: the archives hold hundreds of
        # thousands of members, and a fresh TLS handshake per member is most of
        # the wall clock.
        self.s = session or requests.Session()
        h = self.s.head(url, allow_redirects=True, timeout=timeout)
        h.raise_for_status()
        self.size = int(h.headers["content-length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = (off if whence == 0 else
                    self.pos + off if whence == 1 else
                    self.size + off)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        r = self.s.get(self.url, allow_redirects=True, timeout=self.timeout,
                       headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        r.raise_for_status()
        b = r.content[:n]
        self.pos += len(b)
        return b


def open_remote_zip(repo: str, path: str, revision: str = "master"):
    """(ZipFile, RangeFile) for one archive in a ModelScope dataset."""
    rf = RangeFile(DATASET_URL.format(repo=repo, revision=revision, path=path))
    return zipfile.ZipFile(rf), rf
