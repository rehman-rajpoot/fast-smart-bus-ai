# -*- coding: utf-8 -*-
"""
FAST-NUCES CFD Smart Bus System - Authenticated Media Encryption & Session Tokens
Encrypt-then-MAC (SHA-256 CTR + HMAC-SHA256) using Python Standard Library only.
Ensures NO image (student reference photo or driver safety evidence) is stored or
transferred in plaintext outside authenticated portal sessions.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional

MAGIC_HEADER = b"FSE1"
DEFAULT_SECRET = os.environ.get(
    "FAST_MEDIA_SECRET_KEY",
    "FAST-CFD-2026-SMART-BUS-SECURE-MEDIA-ENVELOPE-V1"
)


def _derive_subkeys(secret: Optional[str], nonce: bytes):
    master = (secret or DEFAULT_SECRET).encode("utf-8")
    enc_key = hashlib.sha256(b"enc|" + master + nonce).digest()
    mac_key = hashlib.sha256(b"mac|" + master + nonce).digest()
    return enc_key, mac_key


def _xor_ctr_stream(data: bytes, enc_key: bytes) -> bytes:
    out = bytearray(len(data))
    block_size = 32
    counter = 0
    for offset in range(0, len(data), block_size):
        chunk = data[offset:offset + block_size]
        ks = hashlib.sha256(enc_key + counter.to_bytes(4, "big")).digest()
        c_int = int.from_bytes(chunk, "big") ^ int.from_bytes(ks[:len(chunk)], "big")
        out[offset:offset + len(chunk)] = c_int.to_bytes(len(chunk), "big")
        counter += 1
    return bytes(out)


def is_encrypted_blob(blob: bytes) -> bool:
    return isinstance(blob, (bytes, bytearray)) and len(blob) >= 52 and blob[:4] == MAGIC_HEADER


def encrypt_bytes(raw_bytes: bytes, secret: Optional[str] = None) -> bytes:
    """Encrypt raw image bytes into an authenticated FSE1 binary envelope."""
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise TypeError("raw_bytes must be bytes")
    nonce = secrets.token_bytes(16)
    enc_key, mac_key = _derive_subkeys(secret, nonce)
    ciphertext = _xor_ctr_stream(bytes(raw_bytes), enc_key)
    tag = hmac.new(mac_key, MAGIC_HEADER + nonce + ciphertext, hashlib.sha256).digest()
    return MAGIC_HEADER + nonce + tag + ciphertext


def decrypt_bytes(blob: bytes, secret: Optional[str] = None) -> bytes:
    """Verify HMAC-SHA256 tag and decrypt an FSE1 encrypted binary envelope."""
    if not is_encrypted_blob(blob):
        raise ValueError("Invalid encrypted media envelope (missing FSE1 header)")
    nonce = blob[4:20]
    tag = blob[20:52]
    ciphertext = blob[52:]
    enc_key, mac_key = _derive_subkeys(secret, nonce)
    expected_tag = hmac.new(mac_key, MAGIC_HEADER + nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected_tag):
        raise ValueError("Media authentication failed: tampered data or wrong secret key")
    return _xor_ctr_stream(ciphertext, enc_key)


def encrypt_b64(raw_bytes: bytes, secret: Optional[str] = None) -> str:
    return base64.b64encode(encrypt_bytes(raw_bytes, secret=secret)).decode("ascii")


def decrypt_b64(encoded_str: str, secret: Optional[str] = None) -> bytes:
    return decrypt_bytes(base64.b64decode(encoded_str.encode("ascii")), secret=secret)


def issue_session_token(
    username: str,
    role: str,
    ttl_seconds: int = 86400,
    secret: Optional[str] = None,
) -> str:
    """Create a stateless HMAC-SHA256 signed session token for authenticated API media reads."""
    master = (secret or DEFAULT_SECRET).encode("utf-8")
    payload = {
        "u": str(username),
        "r": str(role),
        "exp": int(time.time()) + int(ttl_seconds),
        "jti": secrets.token_hex(8),
    }
    raw_payload = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    b64_payload = base64.urlsafe_b64encode(raw_payload).rstrip(b"=").decode("ascii")
    sig = hmac.new(master, b"tok|" + b64_payload.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{b64_payload}.{sig}"


def verify_session_token(token: Optional[str], secret: Optional[str] = None) -> Optional[dict]:
    """Validate an authenticated portal session token."""
    if not token or "." not in token:
        return None
    try:
        b64_payload, sig = token.strip().split(".", 1)
        master = (secret or DEFAULT_SECRET).encode("utf-8")
        expected_sig = hmac.new(
            master, b"tok|" + b64_payload.encode("ascii"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        padding = "=" * (-len(b64_payload) % 4)
        payload = json.loads(base64.urlsafe_b64decode(b64_payload + padding).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None

