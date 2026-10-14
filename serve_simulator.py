import http.server
import socketserver
import os
import sys

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class SimulatorHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.send_response(302)
            self.send_header('Location', '/interactive_simulator/')
            self.end_headers()
            return
        return super().do_GET()

def run_server():
    port = PORT
    for p in [8000, 8080, 5000, 3000]:
        try:
            with socketserver.TCPServer(("", p), SimulatorHTTPHandler) as httpd:
                print(f"[AURA-Cache] Serving local frontend at: http://localhost:{p}/interactive_simulator/", flush=True)
                print(f"[AURA-Cache] Root redirect enabled: http://localhost:{p}/", flush=True)
                httpd.serve_forever()
        except OSError:
            continue

if __name__ == "__main__":
    run_server()
