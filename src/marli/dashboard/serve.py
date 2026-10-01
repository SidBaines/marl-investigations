"""Serve a dashboard's live page over HTTP while the verb keeps refreshing it.

Only the page and its snapshot are served (an allowlist: no listings, no other
files from the out dir). Off loopback every request needs the access key:
``/?key=<key>`` sets an HttpOnly cookie and redirects to ``/``, so the key
leaves the address bar and the page's snapshot polls carry the cookie. The key
comes from ``MARLI_DASHBOARD_KEY`` (a stable bookmark across restarts) or is
random per start; it never enters a config or manifest, only ``serve.json``
(mode 0600) and the log.
"""

from __future__ import annotations

import hmac
import http.cookies
import http.server
import ipaddress
import logging
import os
import secrets
import threading
import urllib.parse
from pathlib import Path

log = logging.getLogger("marli")

KEY_ENV = "MARLI_DASHBOARD_KEY"
COOKIE = "marli_dash_key"
HTML = "text/html; charset=utf-8"
TEXT = "text/plain; charset=utf-8"
ROUTES = {
    "/": ("standalone.html", HTML),
    "/standalone.html": ("standalone.html", HTML),
    "/snapshot.json": ("snapshot.json", "application/json"),
}


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def access_key(host: str, env: dict[str, str] | None = None) -> str | None:
    """None on loopback (the box's own users only); otherwise the env key or a fresh one."""
    if is_loopback(host):
        return None
    configured = (os.environ if env is None else env).get(KEY_ENV, "").strip()
    return configured or secrets.token_urlsafe(24)


def _handler(directory: Path, key: str | None) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "marli-dashboard"
        sys_version = ""

        def log_message(self, format: str, *args: object) -> None:
            log.debug("dashboard http: " + format, *args)

        def do_GET(self) -> None:
            self._serve(body=True)

        def do_HEAD(self) -> None:
            self._serve(body=False)

        def _send(self, code: int, data: bytes, ctype: str, body: bool) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if body:
                self.wfile.write(data)

        def _key_matches(self, given: str | None) -> bool:
            return (
                given is not None
                and key is not None
                and hmac.compare_digest(given.encode("utf-8", "replace"), key.encode())
            )

        def _serve(self, body: bool) -> None:
            url = urllib.parse.urlsplit(self.path)
            if key is not None:
                given = urllib.parse.parse_qs(url.query).get("key", [None])[0]
                if self._key_matches(given):
                    self.send_response(303)
                    secure = self.headers.get("X-Forwarded-Proto", "").lower() == "https"
                    self.send_header(
                        "Set-Cookie",
                        f"{COOKIE}={key}; Path=/; HttpOnly; SameSite=Lax"
                        + ("; Secure" if secure else ""),
                    )
                    self.send_header("Location", url.path or "/")
                    self.send_header("Content-Length", "0")
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.end_headers()
                    return
                try:
                    morsel = http.cookies.SimpleCookie(self.headers.get("Cookie", "")).get(COOKIE)
                except http.cookies.CookieError:
                    morsel = None
                if not self._key_matches(morsel.value if morsel is not None else None):
                    message = b"Forbidden: open the URL with ?key=... printed when it started.\n"
                    self._send(403, message, TEXT, body)
                    return
            route = ROUTES.get(url.path)
            if route is None:
                self._send(404, b"Not found.\n", TEXT, body)
                return
            name, ctype = route
            try:
                data = (directory / name).read_bytes()
            except FileNotFoundError:
                self._send(503, b"No snapshot yet; retry in a moment.\n", TEXT, body)
                return
            self._send(200, data, ctype, body)

    return Handler


class DashboardServer:
    """A threaded HTTP server for one dashboard out dir; ``port=0`` picks a free port."""

    def __init__(self, directory: Path, host: str, port: int, key: str | None) -> None:
        self.httpd = http.server.ThreadingHTTPServer((host, port), _handler(directory, key))
        self.httpd.daemon_threads = True
        self.host = host
        self.port = int(self.httpd.server_address[1])
        self.key = key
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, name="marli-dashboard-http", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5)

    def url(self, *, with_key: bool = True) -> str:
        host = "127.0.0.1" if self.host in ("", "0.0.0.0") else self.host
        host = "[::1]" if host == "::" else f"[{host}]" if ":" in host else host
        suffix = f"?key={self.key}" if with_key and self.key else ""
        return f"http://{host}:{self.port}/{suffix}"
