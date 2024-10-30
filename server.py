import http.server
import socketserver
import json
import os
from urllib.parse import urlparse, parse_qs
import cgi


class RecorderHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve the main page
        if self.path == '/':
            self.path = '/recorder.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

        # Handle sentence requests
        if self.path.startswith('/sentences/'):
            try:
                index = int(self.path.split('/')[-1])
                with open(f'data/raw/sample{str(index).zfill(3)}.txt', 'r') as f:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(
                        {'text': f.read().strip()}).encode())
                    return
            except:
                self.send_response(404)
                self.end_headers()
                return

        # Handle recording check requests
        if self.path.startswith('/check_recording/'):
            try:
                index = int(self.path.split('/')[-1])
                filename = f'data/raw/sample{str(index).zfill(3)}.wav'
                exists = os.path.exists(filename)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'exists': exists}).encode())
                return
            except:
                self.send_response(404)
                self.end_headers()
                return

        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == '/upload':
            # Parse the multipart form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST',
                         'CONTENT_TYPE': self.headers['Content-Type']}
            )

            # Get the file and index
            file_item = form['audio']
            index = form.getvalue('index', '1')

            # Save the file to data/raw/
            filename = f'data/raw/sample{str(int(index)).zfill(3)}.wav'
            with open(filename, 'wb') as f:
                f.write(file_item.file.read())

            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            return

        self.send_response(404)
        self.end_headers()


# Try different ports if 8000 is in use
for port in range(8000, 8010):
    try:
        with socketserver.TCPServer(("", port), RecorderHandler) as httpd:
            print(f"Server running at http://localhost:{port}")
            print("\nFeatures:")
            print("- Type a sample number and press Enter/Go to jump to it")
            print("- Space to start/stop recording")
            print("- Left/Right arrows to navigate")
            print("- Green indicator shows if a recording exists")
            print("- Progress is saved automatically")
            print("\nRecordings will be saved to data/raw/")
            httpd.serve_forever()
            break
    except OSError:
        continue
