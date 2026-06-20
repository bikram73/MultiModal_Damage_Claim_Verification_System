"""Vercel serverless function — GET /api/metrics"""
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
        claims = _load_claims()
        total = len(claims)
        supported = sum(1 for c in claims if c.get('claim_status') == 'supported')
        contradicted = sum(1 for c in claims if c.get('claim_status') == 'contradicted')
        manual_review = sum(
            1 for c in claims
            if 'manual_review_required' in c.get('risk_flags', [])
            or c.get('claim_status') == 'not_enough_information'
        )
        data = {
            'total_claims': total,
            'supported': supported,
            'contradicted': contradicted,
            'manual_review': manual_review,
        }
        body = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)
