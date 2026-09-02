"""
Video decode, and the frame sampler for captioning.

`SegmentFrames` replaces a mutable-default `cache={}` that held fully decoded
RGB frame lists forever. Measured on this corpus that cache reached 8.2 GB
across the 12 rendered segments (pp_noodles alone: 1800 frames x 512x384x3 =
1.06 GB), while the whole run needed roughly 400 individual frames. It also
competed for RAM with a locally hosted VLM, which is the shape of the
zero-caption batches in the old ablation logs.

The fix is to ask for every frame the run needs up front, decode each segment
once sequentially, and keep only the requested frames as encoded JPEG bytes:
about 15 MB for the same work, a ~500x reduction.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np


def decode_stream(h264_bytes: bytes, src_fps: float, out_fps: float,
                  size: tuple[int, int] | None = None):
    """Yield grayscale frames as uint8 arrays, streamed from ffmpeg."""
    if size:
        vf = f"fps={out_fps},scale={size[0]}:{size[1]}"
        w, h = size
    else:
        vf = f"fps={out_fps}"
        w = h = None
    with tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as f:
        f.write(h264_bytes)
        tmp = f.name
    try:
        if w is None:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0", tmp],
                capture_output=True, text=True)
            w, h = [int(x) for x in probe.stdout.strip().split(",")[:2]]
        cmd = ["ffmpeg", "-v", "error", "-f", "h264", "-r", str(src_fps), "-i", tmp,
               "-vf", vf, "-pix_fmt", "gray", "-f", "rawvideo", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        n = w * h
        while True:
            buf = proc.stdout.read(n)
            if not buf or len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w)
        proc.stdout.close()
        proc.wait()
    finally:
        os.unlink(tmp)


def extract_h264(mcap_path, topic: str = "/sensor/camera/head_left/video") -> bytes:
    """Pull one camera's raw H.264 bitstream out of an mcap."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    buf = bytearray()
    with open(str(mcap_path), "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for _, _, _, dec in reader.iter_decoded_messages(topics=[topic]):
            buf += dec.data
    return bytes(buf)


class SegmentFrames:
    """
    JPEG frame store for rendered segment mp4s, built from a known request set.

    Usage:
        store = SegmentFrames(segment_dir)
        store.plan(spans, n_per_span)     # declare everything the run needs
        jpegs = store.get(span)           # cheap, from memory
        store.release("pp_noodles")       # free a finished segment
    """

    def __init__(self, segment_dir, jpeg_quality: int = 80):
        self.dir = str(segment_dir)
        self.quality = int(jpeg_quality)
        self._wanted: dict[str, set[int]] = {}
        self._frames: dict[str, dict[int, bytes]] = {}
        self._fps: dict[str, float] = {}
        self.bytes_held = 0
        self.bytes_peak = 0

    # -- planning ---------------------------------------------------------
    def path_for(self, segment: str) -> str:
        return os.path.join(self.dir, segment + ".mp4")

    def _probe_fps(self, segment: str) -> float:
        if segment not in self._fps:
            import cv2
            path = self.path_for(segment)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"no rendered clip for segment {segment!r} at {path} - run "
                    f"`python -m egoannot segments render --only {segment}`")
            cap = cv2.VideoCapture(path)
            self._fps[segment] = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            cap.release()
        return self._fps[segment]

    def indices_for(self, segment: str, v_start: float, v_end: float, n: int) -> list[int]:
        """The frame indices this span samples: n evenly spaced, span-local times."""
        fps = self._probe_fps(segment)
        lo, hi = v_start * fps, max(v_start * fps, v_end * fps - 1)
        return sorted({int(round(x)) for x in np.linspace(lo, hi, n)})

    def plan(self, spans, n_per_span: int) -> dict[str, int]:
        """Declare every frame the run will ask for. Returns per-segment counts."""
        for s in spans:
            idx = self.indices_for(s["segment"], s["v_start"], s["v_end"], n_per_span)
            self._wanted.setdefault(s["segment"], set()).update(idx)
        return {k: len(v) for k, v in self._wanted.items()}

    # -- decoding ---------------------------------------------------------
    def _load(self, segment: str) -> None:
        """Decode the segment once, keeping only planned frames, JPEG-encoded."""
        import cv2
        want = self._wanted.get(segment)
        path = os.path.join(self.dir, segment + ".mp4")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            # Returning silently here meant the captioner ran blind on that
            # segment -- prompts with zero frames, and nothing in the output
            # saying so.
            raise RuntimeError(f"cannot decode {path}")
        store: dict[int, bytes] = {}
        last = max(want) if want else -1
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if want is None or i in want:
                enc, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if enc:
                    store[i] = buf.tobytes()
                    self.bytes_held += len(store[i])
                    self.bytes_peak = max(self.bytes_peak, self.bytes_held)
            if want is not None and i >= last:
                break
            i += 1
        self._fps[segment] = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()
        self._frames[segment] = store

    def get(self, span, n: int) -> list[bytes]:
        """JPEG bytes for one span's frames, in time order."""
        seg = span["segment"]
        if seg not in self._frames:
            self._load(seg)
        store = self._frames[seg]
        if not store:
            raise RuntimeError(
                f"decoded no frames for segment {seg!r} from {self.path_for(seg)}")
        keys = sorted(store)
        out = []
        for i in self.indices_for(seg, span["v_start"], span["v_end"], n):
            if i in store:
                out.append(store[i])
            else:                      # nearest planned frame
                out.append(store[min(keys, key=lambda k: abs(k - i))])
        return out

    def release(self, segment: str) -> None:
        """Free a finished segment's frames."""
        for b in self._frames.pop(segment, {}).values():
            self.bytes_held -= len(b)
        self._wanted.pop(segment, None)
