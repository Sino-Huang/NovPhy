"""Fetch a public MEGA.nz file link with integrity check (#135 DDLP checkpoint route).

The DDLP README (taldatech/ddlp @ 5686275) hosts its checkpoints and the PHYRE data on MEGA.nz. No MEGA client is
installed on this host; this is a minimal implementation of MEGA's documented public-link protocol:

* ``POST https://g.api.mega.co.nz/cs`` with ``[{"a": "g", "g": 1, "p": <file id>}]`` returns the size ``s``, the
  encrypted attributes ``at`` and a temporary download URL ``g``;
* the 256-bit link key ``k`` (eight big-endian uint32) gives the AES-128 key ``k[0:4] ^ k[4:8]``, the CTR nonce
  ``k[4:6]`` and the meta-MAC ``k[6:8]``;
* the payload is AES-128-CTR; integrity is MEGA's chunked CBC-MAC (chunks 128 KiB, 256 KiB, ..., 1 MiB, then 1 MiB),
  condensed to the meta-MAC, which must equal ``k[6:8]`` (the download is rejected otherwise).

Writes ``<out_dir>/<name from attributes>`` and appends an entry to ``<out_dir>/MANIFEST.json`` (url, bytes, sha256,
downloaded_at, meta_mac_ok). Needs ``pycryptodome`` (env b127-ddlp).

    python scripts/b127_ddlp_mega.py --url 'https://mega.nz/file/<id>#<key>' --out data/external/ddlp/checkpoints
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

API = "https://g.api.mega.co.nz/cs"


def b64url(s: str) -> bytes:
    s = s.replace("-", "+").replace("_", "/").replace(",", "")
    return base64.b64decode(s + "=" * (-len(s) % 4))


def parse_link(url: str) -> tuple[str, bytes]:
    if "/file/" not in url or "#" not in url:
        raise ValueError(f"not a MEGA public file link: {url}")
    fid, key = url.split("/file/", 1)[1].split("#", 1)
    return fid, b64url(key)


def split_key(raw: bytes) -> tuple[bytes, bytes, bytes]:
    if len(raw) != 32:
        raise ValueError(f"MEGA file key must be 32 bytes, got {len(raw)}")
    a = struct.unpack(">8I", raw)
    aes = struct.pack(">4I", a[0] ^ a[4], a[1] ^ a[5], a[2] ^ a[6], a[3] ^ a[7])
    nonce = struct.pack(">2I", a[4], a[5])
    meta_mac = struct.pack(">2I", a[6], a[7])
    return aes, nonce, meta_mac


def chunk_bounds(size: int):
    start, step = 0, 0x20000
    while start < size:
        end = min(size, start + step)
        yield start, end
        start = end
        if step < 0x100000:
            step += 0x20000


def api_get(fid: str) -> dict:
    req = urllib.request.Request(
        API + "?id=1", data=json.dumps([{"a": "g", "g": 1, "p": fid}]).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    if not isinstance(out, list) or not isinstance(out[0], dict):
        raise RuntimeError(f"MEGA API error for {fid}: {out}")
    return out[0]


def decrypt_attrs(at: str, aes: bytes) -> dict:
    from Crypto.Cipher import AES

    plain = AES.new(aes, AES.MODE_CBC, iv=b"\0" * 16).decrypt(b64url(at)).rstrip(b"\0")
    if not plain.startswith(b"MEGA"):
        raise RuntimeError("attribute decryption failed (wrong key)")
    return json.loads(plain[4:].decode())


def fetch(url: str, out_dir: Path) -> dict:
    from Crypto.Cipher import AES
    from Crypto.Util import Counter

    fid, raw = parse_link(url)
    aes, nonce, meta_mac = split_key(raw)
    info = api_get(fid)
    attrs = decrypt_attrs(info["at"], aes)
    size = int(info["s"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / attrs["n"]
    tmp = dest.with_suffix(dest.suffix + ".part")
    ctr = AES.new(aes, AES.MODE_CTR, counter=Counter.new(64, prefix=nonce, initial_value=0))
    ecb = AES.new(aes, AES.MODE_ECB)
    sha = hashlib.sha256()
    file_mac = b"\0" * 16
    bounds = list(chunk_bounds(size))
    with urllib.request.urlopen(info["g"], timeout=120) as r, tmp.open("wb") as f:
        for start, end in bounds:
            enc = b""
            while len(enc) < end - start:
                piece = r.read(end - start - len(enc))
                if not piece:
                    raise RuntimeError(f"short read at {start + len(enc)} of {size}")
                enc += piece
            plain = ctr.decrypt(enc)
            f.write(plain)
            sha.update(plain)
            padded = plain + b"\0" * (-len(plain) % 16)
            chunk_mac = AES.new(aes, AES.MODE_CBC, iv=nonce + nonce).encrypt(padded)[-16:]
            file_mac = ecb.encrypt(bytes(x ^ y for x, y in zip(file_mac, chunk_mac)))
    w = struct.unpack(">4I", file_mac)
    got = struct.pack(">2I", w[0] ^ w[1], w[2] ^ w[3])
    ok = got == meta_mac
    if not ok:
        tmp.unlink()
        raise RuntimeError(f"meta-MAC mismatch for {attrs['n']}: download corrupt")
    tmp.replace(dest)
    entry = {
        "url": url, "file": dest.name, "bytes": size, "sha256": "sha256:" + sha.hexdigest(),
        "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "meta_mac_ok": ok, "method": "scripts/b127_ddlp_mega.py (MEGA public-link API + AES-CTR + meta-MAC)",
    }
    manifest = out_dir / "MANIFEST.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else {"files": []}
    data["files"] = [e for e in data["files"] if e["file"] != dest.name] + [entry]
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    return entry


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", required=True, action="append")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    for url in a.url:
        print(json.dumps(fetch(url, a.out)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
