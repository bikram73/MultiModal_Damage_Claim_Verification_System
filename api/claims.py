"""Vercel serverless function — GET /api/claims"""
import json
import os
from http.server import BaseHTTPRequestHandler


def _load_claims():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, 'code', 'claims_data.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return []


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = _load_claims()
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)
