"""Paid-mode service authentication, bounded ingress, and nonqueued admission."""

from __future__ import annotations

import asyncio
import hmac
from threading import BoundedSemaphore

from pydantic import SecretStr
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_INGRESS_BYTES = 512 * 1024
BODY_READ_TIMEOUT_SECONDS = 5.0


class PaidServiceBoundary:
    def __init__(self, app: ASGIApp, service_key: SecretStr, max_concurrency: int) -> None:
        self.app = app
        self._key = service_key.get_secret_value().encode("ascii")
        self._slots = BoundedSemaphore(max_concurrency)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        # Starlette routes relative to root_path, including mounted apps and
        # reverse-proxy prefixes. Apply the same segment-boundary stripping.
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path):
            relative = path[len(root_path) :]
            if not relative or relative.startswith("/"):
                path = relative
        if scope["type"] != "http" or not (path == "/v1" or path.startswith("/v1/")):
            await self.app(scope, receive, send)
            return
        authorization = [
            value for name, value in scope.get("headers", []) if name.lower() == b"authorization"
        ]
        if len(authorization) != 1:
            await self._reject(401, "Service authentication required.", scope, receive, send)
            return
        scheme, separator, supplied = authorization[0].partition(b" ")
        if (
            scheme.lower() != b"bearer"
            or not separator
            or not hmac.compare_digest(supplied, self._key)
        ):
            await self._reject(401, "Service authentication required.", scope, receive, send)
            return

        # Reads and operator/identity commands also use the same DB pool.
        # Share the cap across all authenticated routes so they cannot starve
        # an async provider turn through synchronous database checkout.
        if not self._slots.acquire(blocking=False):
            await self._reject(
                429, "Paid request capacity reached. Retry later.", scope, receive, send
            )
            return
        try:
            body = bytearray()
            try:
                async with asyncio.timeout(BODY_READ_TIMEOUT_SECONDS):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        chunk = message.get("body", b"")
                        if len(body) + len(chunk) > MAX_INGRESS_BYTES:
                            await self._reject(
                                413, "Request exceeds the size limit.", scope, receive, send
                            )
                            return
                        body.extend(chunk)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                await self._reject(408, "Request body timed out.", scope, receive, send)
                return
            pending = True

            async def replay() -> Message:
                nonlocal pending
                if pending:
                    pending = False
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, send)
        finally:
            self._slots.release()

    @staticmethod
    async def _reject(status: int, detail: str, scope: Scope, receive: Receive, send: Send) -> None:
        headers = {"www-authenticate": "Bearer"} if status == 401 else None
        response = JSONResponse({"detail": detail}, status_code=status, headers=headers)
        await response(scope, receive, send)
