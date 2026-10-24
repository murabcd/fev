"""Small synchronous reference HTTP server. Put behind a managed reverse proxy.

One resident model, serialized requests, bounded bodies, no request-content logs.
This is an integration reference, not a qualified high-throughput serving stack.
"""
import hmac
import io
import json
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from .decisions import MODEL_ID, validate_request

MAX_BODY_BYTES = 1024 * 1024


class HTTPError(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON keys are unsupported')
        result[key] = value
    return result


def create_server(model, address=('127.0.0.1', 8000), *, api_key=None):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *_):
            pass

        def respond(self, status, body):
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('x-typesafe-request-id', str(uuid.uuid4()))
            self.end_headers()
            self.wfile.write(data)

        def authorize(self):
            if api_key and not hmac.compare_digest(
                    self.headers.get('Authorization', '').encode(), ('Bearer ' + api_key).encode()):
                raise HTTPError(401, 'Invalid bearer token')

        def do_GET(self):
            try:
                self.authorize()
                if self.path == '/v1/models':
                    self.respond(200, {'models': [{'name': MODEL_ID, 'description':
                        'Fev-4B text decisions: Choice, Noul and Score; zero generated tokens.',
                        'release_date': '2026-10-24'}]})
                else:
                    raise HTTPError(404, 'Unknown endpoint')
            except HTTPError as error:
                self.respond(error.status, {'detail': error.detail})

        def read_payload(self):
            if self.headers.get('Transfer-Encoding'):
                raise HTTPError(400, 'Chunked request bodies are unsupported')
            if len(self.headers.get_all('Content-Length', [])) != 1:
                raise HTTPError(411, 'One Content-Length header is required')
            try:
                size = int(self.headers['Content-Length'])
            except ValueError:
                raise HTTPError(400, 'Invalid Content-Length') from None
            if not 0 < size <= MAX_BODY_BYTES:
                raise HTTPError(413, 'Request body exceeds the 1 MiB limit')
            if self.headers.get_content_type() != 'application/json':
                raise HTTPError(415, 'Content-Type must be application/json')
            data = self.rfile.read(size)
            if len(data) != size:
                raise HTTPError(400, 'Incomplete request body')
            encoding = self.headers.get('Content-Encoding', 'identity').lower()
            if encoding == 'zstd':
                import zstandard
                try:
                    with zstandard.ZstdDecompressor(max_window_size=1024).stream_reader(io.BytesIO(data)) as stream:
                        data = stream.read(MAX_BODY_BYTES + 1)
                except zstandard.ZstdError:
                    raise HTTPError(400, 'Invalid or oversized zstd frame') from None
                if len(data) > MAX_BODY_BYTES:
                    raise HTTPError(413, 'Decoded request exceeds the 1 MiB limit')
            elif encoding != 'identity':
                raise HTTPError(415, 'Supported Content-Encoding: identity, zstd')
            return json.loads(data.decode('utf-8'), object_pairs_hook=_unique_object,
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON')))

        def do_POST(self):
            try:
                self.authorize()
                if self.path != '/v1/systemone':
                    raise HTTPError(404, 'Unknown endpoint')
                payload = self.read_payload()
                if not isinstance(payload, dict) or set(payload) != {'model', 'state', 'questions'}:
                    raise ValueError('Expected exactly model, state and questions; assets are unsupported')
                if payload['model'] != MODEL_ID:
                    raise ValueError('Only model fev-4b is available')
                request = validate_request({'state': payload['state'], 'questions': payload['questions']})
                self.respond(200, model.decide(request))
            except HTTPError as error:
                self.respond(error.status, {'detail': error.detail})
            except (ValueError, UnicodeError, RecursionError):
                self.respond(422, {'detail': 'Invalid request or context limit exceeded; see API.md'})
            except (TimeoutError, ConnectionError):
                self.close_connection = True
            except Exception:
                self.respond(500, {'detail': 'Inference failed; no partial answers were returned'})

    return HTTPServer(address, Handler)
