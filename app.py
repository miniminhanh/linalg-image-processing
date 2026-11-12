"""
A small web UI for the filters in this repo.

Run:    python app.py
Then open http://localhost:8000 in your browser.

Only uses the standard library plus OpenCV and NumPy, so nothing extra to install.

Settings (environment variables), all optional:
    HOST       address to listen on. Default 127.0.0.1 (this computer only).
               Use 0.0.0.0 on a server so other machines can reach it.
    PORT       port to listen on. Default 8000.
    MAX_SIDE   uploads larger than this many pixels on their longest side are shrunk
               before processing, to keep memory use down. Default 2000.
    MAX_JOBS   how many images can be processed at the same time. Default 2.
"""
import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Refuse to decode images over 50 megapixels (e.g. a tiny PNG that expands to a huge image).
# Must be set before cv2 is imported.
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', str(50_000_000))

import cv2
import numpy as np

from filters import FILTERS, KERNEL_SIZES, SIGMA_MIN, SIGMA_MAX, auto_kernel_size

HOST = os.environ.get('HOST', '127.0.0.1')
PORT = int(os.environ.get('PORT', 8000))
MAX_SIDE = int(os.environ.get('MAX_SIDE', 2000))
MAX_JOBS = int(os.environ.get('MAX_JOBS', 2))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

# Limits how many images are processed at once, so a few big uploads can't run the server out of memory
job_slots = threading.BoundedSemaphore(MAX_JOBS)
INDEX_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')


def shrink_to_fit(image, max_side):
    """Scale the image down (keeping its aspect ratio) so its longest side is at most max_side."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image, None
    scale = max_side / max(h, w)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    # INTER_AREA averages pixels together, which is the best choice for shrinking
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), new_size


def encode_png(image):
    ok, buf = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('Could not encode result image')
    return base64.b64encode(buf.tobytes()).decode('ascii')


class Handler(BaseHTTPRequestHandler):
    timeout = 30  # drop connections that stall for 30 s, so slow clients can't tie up threads

    def send_json(self, status, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(INDEX_HTML, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/filters':
            self.send_json(200, {
                'kernel_sizes': KERNEL_SIZES,
                'sigma_min': SIGMA_MIN,
                'sigma_max': SIGMA_MAX,
                'filters': [
                    {
                        'id': fid,
                        'name': f['name'],
                        'description': f['description'],
                        'controls': f['controls'],
                        'defaults': f['defaults'],
                    }
                    for fid, f in FILTERS.items()
                ],
            })
        else:
            self.send_json(404, {'error': 'Not found'})

    def do_POST(self):
        if self.path != '/process':
            self.send_json(404, {'error': 'Not found'})
            return

        length = int(self.headers.get('Content-Length', 0))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self.send_json(413, {'error': 'Image is empty or larger than 25 MB'})
            return

        try:
            payload = json.loads(self.rfile.read(length))
            filter_id = payload['filter']
            if filter_id not in FILTERS:
                self.send_json(400, {'error': f'Unknown filter: {filter_id}'})
                return

            # The browser sends the file as a base64 data URL
            data = payload['image'].split(',', 1)[-1]
            image = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                self.send_json(400, {'error': 'Could not read that file as an image'})
                return

            image, resized_to = shrink_to_fit(image, MAX_SIDE)

            f = FILTERS[filter_id]
            kwargs = {}
            if f['controls'] == 'gaussian':
                sigma = float(payload.get('sigma', f['defaults']['sigma']))
                if not SIGMA_MIN <= sigma <= SIGMA_MAX:
                    self.send_json(400, {'error': f'Sigma must be between {SIGMA_MIN} and {SIGMA_MAX}'})
                    return
                kwargs['sigma'] = sigma

            if f['controls'] in ('gaussian', 'kernel'):
                if f['controls'] == 'gaussian' and payload.get('auto_kernel'):
                    # Automatic mode: the kernel size follows from sigma
                    kernel_size = auto_kernel_size(kwargs['sigma'])
                else:
                    kernel_size = int(payload.get('kernel_size', f['defaults']['kernel_size']))
                    if kernel_size not in KERNEL_SIZES:
                        self.send_json(400, {'error': f'Kernel size must be one of {KERNEL_SIZES}'})
                        return

                if kernel_size > min(image.shape[:2]):
                    self.send_json(400, {'error': f'Image is too small for a {kernel_size}x{kernel_size} kernel'})
                    return
                kwargs['kernel_size'] = kernel_size

            # Wait up to 60 s for a free processing slot
            if not job_slots.acquire(timeout=60):
                self.send_json(503, {'error': 'The server is busy, please try again in a moment'})
                return
            try:
                outputs = f['func'](image, **kwargs)
                result = [{'title': title, 'image': encode_png(img)} for title, img in outputs]
            finally:
                job_slots.release()

            self.send_json(200, {
                'outputs': result,
                'resized_to': resized_to,
                'kernel_size': kwargs.get('kernel_size'),  # the size actually used (matters in automatic mode)
            })
        except Exception as e:  # report errors to the page instead of crashing the server
            self.send_json(500, {'error': str(e)})

    def log_message(self, fmt, *args):
        sys.stderr.write('[app] ' + (fmt % args) + '\n')


if __name__ == '__main__':
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    shown_host = 'localhost' if HOST in ('127.0.0.1', '0.0.0.0') else HOST
    print(f'Image filter UI running at http://{shown_host}:{PORT}  (listening on {HOST}, Ctrl+C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
        server.server_close()
