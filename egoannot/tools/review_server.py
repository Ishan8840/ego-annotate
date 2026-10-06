"""Local static review server with byte ranges for reliable browser video seeking."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class ReviewHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range = None
        value = self.headers.get('Range')
        path = Path(self.translate_path(self.path))
        if not value or not path.is_file() or self.headers.get('If-Range'):
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
        start, end = 0, size - 1
        if match and any(match.groups()):
            first, last = match.groups()
            if first:
                start, end = int(first), min(int(last), size - 1) if last else size - 1
            else:
                start = max(0, size - int(last))
        else:
            start = size
        if start >= size or end < start:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        fh = path.open('rb')
        fh.seek(start)
        self.byte_range = (start, end)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        return fh

    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def copyfile(self, source, outputfile):
        if self.byte_range is None:
            return super().copyfile(source, outputfile)
        remaining = self.byte_range[1] - self.byte_range[0] + 1
        while remaining:
            buf = source.read(min(remaining, 1024 * 1024))
            if not buf:
                break
            outputfile.write(buf)
            remaining -= len(buf)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--directory', required=True)
    p.add_argument('--port', type=int, default=8081)
    args = p.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(ReviewHandler, directory=args.directory))
    print(f'Review: http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()
