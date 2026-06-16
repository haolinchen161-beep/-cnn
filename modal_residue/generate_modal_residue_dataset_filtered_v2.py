# -*- coding: utf-8 -*-
"""Dataset generator entry point.

This file reconstructs the uploaded generator source from text parts stored in
`modal_residue/source_parts/` and executes it.
"""
from __future__ import annotations

import base64
import gzip
from pathlib import Path

_parts_dir = Path(__file__).with_name("source_parts")
_parts = sorted(_parts_dir.glob("part*.txt"))
if not _parts:
    raise FileNotFoundError(f"missing source parts: {_parts_dir}/part*.txt")

_packed = "".join(p.read_text(encoding="ascii").strip() for p in _parts)
_source = gzip.decompress(base64.b64decode(_packed.encode("ascii"))).decode("utf-8")
exec(compile(_source, __file__, "exec"), globals(), globals())
