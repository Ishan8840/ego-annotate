from functools import partial
from http.server import ThreadingHTTPServer
import threading
import urllib.error
import urllib.request

from egoannot.tools.review_server import ReviewHandler


def test_video_ranges_allow_seeking_and_reject_past_eof(tmp_path):
    (tmp_path/'clip.mp4').write_bytes(b'0123456789')
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(ReviewHandler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/clip.mp4'
        for byte_range, expected, header in [('bytes=3-5', b'345', 'bytes 3-5/10'),
                                             ('bytes=7-', b'789', 'bytes 7-9/10'),
                                             ('bytes=-2', b'89', 'bytes 8-9/10')]:
            request = urllib.request.Request(url, headers={'Range': byte_range})
            with urllib.request.urlopen(request) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == header
                assert response.read() == expected
        request = urllib.request.Request(url, headers={'Range': 'bytes=20-'})
        try:
            urllib.request.urlopen(request)
            raise AssertionError('range past EOF must fail')
        except urllib.error.HTTPError as exc:
            assert exc.code == 416
        with urllib.request.urlopen(url) as response:
            assert response.status == 200 and response.read() == b'0123456789'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
