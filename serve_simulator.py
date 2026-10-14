import http.server
import socketserver
import os

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class NoCacheHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Disable all caching so browser always fetches the latest file
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.send_response(302)
            self.send_header('Location', '/interactive_simulator/')
            self.end_headers()
            return
        return super().do_GET()

    def log_message(self, format, *args):
        print(f"[AURA-Cache] {self.address_string()} - {format % args}", flush=True)

def run_server():
    for p in [8000, 8080, 5000, 3000]:
        try:
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("", p), NoCacheHTTPHandler) as httpd:
                print(f"[AURA-Cache] Serving at: http://localhost:{p}/interactive_simulator/", flush=True)
                print(f"[AURA-Cache] No-cache mode ON - browser always gets fresh files.", flush=True)
                httpd.serve_forever()
        except OSError:
            continue

if __name__ == "__main__":
    run_server()
