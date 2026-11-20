"""USB serial link to the XIAO camera (protocol in esp32/cat_dog_usb/cat_dog_usb.ino)."""
import io
import json
import math
import queue
import random
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

TAGS = (b"IMG1", b"RES1", b"TXT1", b"ERR1")
MAX_LEN = 4_000_000            # anything bigger is a corrupted header, not a real image
ESPRESSIF_USB_VID = 0x303A     # USB vendor ID of Espressif chips (finds the XIAO automatically)
RESOLUTIONS = {"320 x 240": "q", "640 x 480": "v", "800 x 600": "s", "1024 x 768": "x"}

class DeviceError(Exception):
    """The ESP answered, but with an error (ERR1)."""

class EspLink:
    def __init__(self, ser):
        self.ser = ser
        self.buf = bytearray()
        try:
            ser.reset_input_buffer()  # throw away boot messages / half-sent data
        except Exception:
            pass

    def read_packet(self, timeout):
        """Collect bytes until one complete tagged packet is in the buffer.
        Reads can arrive in pieces, so we keep adding to self.buf; junk before a tag is skipped."""
        deadline = time.perf_counter() + timeout
        while True:
            hits = [i for i in (self.buf.find(t) for t in TAGS) if i >= 0]
            if hits:
                start = min(hits)
                if start:
                    del self.buf[:start]                     # skip junk (e.g. boot text)
                if len(self.buf) >= 8:
                    n = int.from_bytes(self.buf[4:8], "little")
                    if n > MAX_LEN:                          # implausible length: resync
                        del self.buf[:4]
                        continue
                    if len(self.buf) >= 8 + n:
                        tag, data = bytes(self.buf[:4]), bytes(self.buf[8:8 + n])
                        del self.buf[:8 + n]
                        return tag, data
            elif len(self.buf) > 3:
                del self.buf[:-3]                            # keep a possible partial tag
            if time.perf_counter() > deadline:
                self.buf.clear()                             # don't let a half packet poison the next one
                raise TimeoutError("no complete reply from the ESP in time")
            chunk = self.ser.read(max(1, self.ser.in_waiting))
            if chunk:
                self.buf += chunk

    def request(self, payload, expect, timeout=3.0):
        self.ser.write(payload)
        tag, data = self.read_packet(timeout)
        if tag == b"ERR1":
            raise DeviceError(data.decode(errors="replace") or "error from ESP")
        if tag != expect:
            raise IOError(f"expected {expect.decode()}, got {tag.decode()}")
        return data

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

def find_port():
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid == ESPRESSIF_USB_VID:
            return p.device
    if len(ports) == 1:
        return ports[0].device
    names = ", ".join(p.device for p in ports) or "none"
    raise IOError(f"XIAO not found (ports: {names}). Plug it in or pass --port COM10")

def open_serial(port=None):
    import serial
    s = serial.Serial()
    s.port = port or find_port()
    s.baudrate = 115200          # ignored by native USB, but required by the API
    s.timeout = 0.05
    s.write_timeout = 5
    s.dtr = False                # set before opening so the board isn't reset
    s.rts = False
    s.open()
    return s

def center_square(w, h):
    side = min(w, h)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return (x0, y0, x0 + side, y0 + side)

def prepare(img, n):
    """Same as training: centre square crop, resize to n x n, RGB 0..255.
    (The ESP then scales to -1..1 and converts to int8 using the model's own numbers.)"""
    box = center_square(*img.size)
    small = img.crop(box).resize((n, n), Image.BILINEAR)
    return np.asarray(small, dtype=np.uint8), box

class CameraWorker(threading.Thread):
    """Runs the USB conversation in the background and posts events for the GUI:
    ("status", text) ("info", text) ("frame", img, fps, kb) ("capturing",)
    ("result", img, pixels, box, scores_dict, seconds) ("result_error", text) ("error", text)"""

    def __init__(self, open_fn, events):
        super().__init__(daemon=True)
        self.open_fn = open_fn
        self.events = events
        self.lock = threading.Lock()
        self.pressed_at = None
        self.pending_res = None
        self.rotate180 = False
        self.reconnect_now = threading.Event()
        self.stopped = threading.Event()

    def request_capture(self):
        with self.lock:
            self.pressed_at = time.perf_counter()

    def set_resolution(self, code):
        self.pending_res = code

    def reconnect(self):
        self.reconnect_now.set()

    def stop(self):
        self.stopped.set()
        self.reconnect_now.set()

    def _fail_pending_capture(self, why):
        with self.lock:
            had = self.pressed_at is not None
            self.pressed_at = None
        if had:
            self.events.put(("result_error", why))

    def run(self):
        while not self.stopped.is_set():
            link = None
            try:
                self.events.put(("status", "Connecting..."))
                link = EspLink(self.open_fn())
                info = link.request(b"I", b"TXT1", timeout=4).decode(errors="replace")
                size = 96
                for line in info.splitlines():
                    if line.startswith("input_size="):
                        size = int(line.split("=")[1])
                self.events.put(("info", info))
                self.events.put(("status", f"Connected ({getattr(link.ser, 'port', 'USB')})"))
                self.reconnect_now.clear()
                fps, last = 0.0, time.perf_counter()
                while not self.stopped.is_set() and not self.reconnect_now.is_set():
                    if self.pending_res:
                        code, self.pending_res = self.pending_res, None
                        link.request(b"R" + code.encode(), b"TXT1", timeout=4)
                    sent_at = time.perf_counter()
                    jpeg = link.request(b"C", b"IMG1", timeout=4)
                    img = Image.open(io.BytesIO(jpeg))
                    img = img.convert("RGB")
                    if self.rotate180:
                        img = img.rotate(180)
                    now = time.perf_counter()
                    fps = 0.8 * fps + 0.2 / max(now - last, 1e-3) if fps else 1 / max(now - last, 1e-3)
                    last = now
                    self.events.put(("frame", img, fps, len(jpeg) / 1024))

                    with self.lock:  # the first frame REQUESTED after the button press becomes the snapshot
                        pressed = self.pressed_at if (self.pressed_at and sent_at >= self.pressed_at) else None
                        if pressed:
                            self.pressed_at = None
                    if pressed:
                        self.events.put(("capturing",))
                        pixels, box = prepare(img, size)
                        try:
                            scores = json.loads(link.request(b"K" + pixels.tobytes(), b"RES1", timeout=20))
                            self.events.put(("result", img, pixels, box, scores, time.perf_counter() - pressed))
                        except DeviceError as e:
                            self.events.put(("result_error", f"ESP error: {e}"))
                    last = time.perf_counter() if pressed else last
            except Exception as e:  # unplugged, timeout, port busy, ...
                self.events.put(("error", f"{type(e).__name__}: {e}"))
                self._fail_pending_capture("No result: connection problem")
            finally:
                if link:
                    link.close()
            if not self.stopped.is_set():
                self.reconnect_now.wait(2.0)  # retry every 2 s, or right away on Reconnect

class FakeEspSerial:
    """Pretends to be the XIAO: answers C/K/I/R with the real packet format, makes synthetic
    camera frames, and sometimes sends junk bytes to exercise the resync logic."""

    def __init__(self, junk=True):
        self.out = bytearray()
        self.inbox = bytearray()
        self.size = (640, 480)
        self.t0 = time.perf_counter()
        self.junk = junk
        self.port = "SIMULATED"
        self.out += b"ESP-ROM:esp32s3-20210327\r\nentry 0x403c88b8\r\n"  # boot text, like the real board

    @property
    def in_waiting(self):
        return len(self.out)

    def reset_input_buffer(self):
        self.out.clear()

    def _packet(self, tag, data):
        if self.junk and random.random() < 0.05:
            self.out += b"\x00junk"
        self.out += tag + len(data).to_bytes(4, "little") + data

    def _frame(self):
        w, h = self.size
        t = time.perf_counter() - self.t0
        img = Image.new("RGB", (w, h), (70, 90, 120))
        d = ImageDraw.Draw(img)
        cx, cy, r = w / 2 + math.sin(t) * w / 5, h / 2, h / 4
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(200, 150, 90))
        d.text((10, 10), f"SIMULATED XIAO CAMERA  {w}x{h}", fill=(255, 255, 255))
        b = io.BytesIO()
        img.save(b, "JPEG", quality=85)
        return b.getvalue()

    def write(self, data):
        self.inbox += data
        while self.inbox:
            c = self.inbox[:1]
            if c == b"C":
                del self.inbox[:1]
                time.sleep(0.03)
                self._packet(b"IMG1", self._frame())
            elif c == b"I":
                del self.inbox[:1]
                self._packet(b"TXT1", b"SIMULATED board\nModel ready\ninput_size=96\n")
            elif c == b"R":
                if len(self.inbox) < 2:
                    return
                code = chr(self.inbox[1])
                del self.inbox[:2]
                self.size = {"q": (320, 240), "v": (640, 480), "s": (800, 600), "x": (1024, 768)}[code]
                self._packet(b"TXT1", b"resolution set")
            elif c == b"K":
                need = 1 + 96 * 96 * 3
                if len(self.inbox) < need:
                    return
                px = np.frombuffer(bytes(self.inbox[1:need]), np.uint8).reshape(96, 96, 3)
                del self.inbox[:need]
                time.sleep(0.25)  # pretend the model takes a while
                warm = float(px[..., 0].mean() - px[..., 2].mean()) / 255  # "orange-ish = cat" fake rule
                cat = 1 / (1 + math.exp(-8 * warm))
                self._packet(b"RES1", json.dumps({"ms": 250, "labels": ["cat", "dog"],
                                                  "probs": [round(cat, 4), round(1 - cat, 4)]}).encode())
            else:
                del self.inbox[:1]
        return len(data)

    def read(self, n=1):
        if not self.out:
            time.sleep(0.005)
            return b""
        chunk = bytes(self.out[:min(n, 4096)])  # deliver in pieces, like real USB
        del self.out[:len(chunk)]
        return chunk

    def close(self):
        pass
