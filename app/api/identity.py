"""Demo identity held in a signed cookie.

This is **not** authentication. It exists so the browser demo resolves identity the
same way a deployment would - on the server, from something the client cannot rewrite -
instead of trusting a `customer_id` sent in the request body.

The signing key is generated per process, so restarting the server ends every session.
A real deployment replaces this module with a verified session from an auth provider;
nothing downstream changes, because `SessionContext` is already the only identity the
pipeline accepts.
"""

import hashlib
import hmac
import secrets

COOKIE_NAME = "ss_demo_identity"
_SPLIT = "."


class DemoIdentitySigner:
    """Signs and verifies `customer_id` values for the demo cookie."""

    def __init__(self, secret: bytes | None = None):
        self._secret = secret or secrets.token_bytes(32)

    def _mac(self, customer_id: str) -> str:
        return hmac.new(self._secret, customer_id.encode("utf-8"), hashlib.sha256).hexdigest()

    def sign(self, customer_id: str) -> str:
        return f"{customer_id}{_SPLIT}{self._mac(customer_id)}"

    def verify(self, value: str | None) -> str | None:
        """Return the signed customer id, or None for anything unsigned or tampered."""
        if not value or _SPLIT not in value:
            return None
        customer_id, _, mac = value.rpartition(_SPLIT)
        if not customer_id:
            return None
        if not hmac.compare_digest(mac, self._mac(customer_id)):
            return None
        return customer_id
