import json
import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OFFLINE_ROOT = os.path.join(PROJECT_ROOT, "11_testing", "offline_dashboard")
STATIC_ROOT = os.path.join(OFFLINE_ROOT, "s3")
MODELS_ROOT = os.path.join(OFFLINE_ROOT, "models")
BACKEND_ROOT = os.path.join(OFFLINE_ROOT, "code", "backend")


def _ensure_paths() -> None:
    missing = [p for p in (STATIC_ROOT, BACKEND_ROOT) if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            "Offline bundle missing required paths: " + ", ".join(missing) +
            ". Run 11_testing/offline_dashboard_download.ps1 first."
        )


def _load_lambda():
    # Ensure offline backend is importable
    if BACKEND_ROOT not in sys.path:
        sys.path.insert(0, BACKEND_ROOT)

    # Force lambda to prefer local models, avoiding S3
    if os.path.exists(MODELS_ROOT):
        os.environ.setdefault("MODEL_BASE_PATH", MODELS_ROOT)

    import lambda_function  # type: ignore
    return lambda_function


class OfflineHandler(SimpleHTTPRequestHandler):
    # Serve static files from STATIC_ROOT
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_ROOT, **kwargs)

    def _send_json(self, status: int, obj: object) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_lambda(self) -> None:
        lf = self.server.lambda_function  # type: ignore[attr-defined]

        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        query_params = {k: v[-1] for k, v in qs.items() if v}

        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length) if length > 0 else b""

        event = {
            "httpMethod": self.command,
            "path": path,
            "headers": {k: v for k, v in self.headers.items()},
            "queryStringParameters": query_params or None,
            "body": raw_body.decode("utf-8") if raw_body else None,
            "isBase64Encoded": False,
        }

        try:
            resp = lf.lambda_handler(event, None)
        except Exception as e:
            self._send_json(500, {"error": str(e)})
            return

        status = int(resp.get("statusCode", 200))
        headers = resp.get("headers") or {}
        body = resp.get("body")
        if body is None:
            body_bytes = b""
        elif isinstance(body, (dict, list)):
            body_bytes = json.dumps(body).encode("utf-8")
            headers = {**headers, "Content-Type": "application/json"}
        else:
            body_bytes = str(body).encode("utf-8")

        self.send_response(status)
        for hk, hv in headers.items():
            self.send_header(hk, hv)
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        if body_bytes:
            self.wfile.write(body_bytes)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/prod/"):
            return self._handle_lambda()
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/prod/"):
            return self._handle_lambda()
        self._send_json(404, {"error": "POST only supported under /prod/* in offline mode"})


def main() -> None:
    _ensure_paths()
    lf = _load_lambda()

    port = int(os.environ.get("OFFLINE_DASHBOARD_PORT", "8000"))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), OfflineHandler)
    httpd.lambda_function = lf  # type: ignore[attr-defined]

    print("Offline dashboard server running")
    print(f"- Static root: {STATIC_ROOT}")
    print(f"- Lambda backend: {BACKEND_ROOT}")
    print(f"- Models root: {MODELS_ROOT} (exists={os.path.exists(MODELS_ROOT)})")
    print(f"Open: http://127.0.0.1:{port}/index.html?apiBase=http://127.0.0.1:{port}/prod&staticBase=http://127.0.0.1:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
