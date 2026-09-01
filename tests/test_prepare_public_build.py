from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_public_build import _iter_parquet_candidates


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (12, 10), color).save(buf, format="PNG")
    return buf.getvalue()


class PreparePublicBuildTest(unittest.TestCase):
    def test_huggingface_image_struct_parquet_yields_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            data.mkdir()
            pd.DataFrame([
                {
                    "img_id": "sid-real",
                    "image": {"bytes": _png_bytes((255, 0, 0)), "path": None},
                    "label": 0,
                },
                {
                    "img_id": "sid-fake",
                    "image": {"bytes": _png_bytes((0, 255, 0)), "path": None},
                    "label": 1,
                },
                {
                    "img_id": "sid-tampered",
                    "image": {"bytes": _png_bytes((0, 0, 255)), "path": None},
                    "label": 2,
                },
            ]).to_parquet(data / "validation-00000-of-00001.parquet", index=False)

            rows = list(_iter_parquet_candidates("SID_Set", root, batch_size=2))

        self.assertEqual(len(rows), 2)
        self.assertEqual({r[0].label for r in rows}, {0, 1})
        self.assertEqual({r[0].raw_id for r in rows}, {"sid-real", "sid-fake"})

    def test_requirements_file_is_ascii_decodable(self):
        req = Path(__file__).resolve().parents[1] / "requirements.txt"
        req.read_text(encoding="ascii")


if __name__ == "__main__":
    unittest.main()
