"""Tkinter app: live camera, capture and classify on the ESP32.

Usage: python cat_dog_gui.py [--port COM10] [--simulate]
"""
import argparse
import csv
import queue
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk

from camlink import RESOLUTIONS, CameraWorker, FakeEspSerial, center_square, open_serial

PANEL_W, PANEL_H = 400, 300   # each video panel; the window is about 1,180 px wide
BG, CARD, INK, DIM, ACC, BAD = "#0f1115", "#181b22", "#e8eaed", "#9aa0a6", "#4f8cff", "#e5534b"
HERE = Path(__file__).resolve().parent

def fit(img, w=PANEL_W, h=PANEL_H):
    """Scale an image to fit the panel; return it and the scale factor."""
    s = min(w / img.width, h / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR), s

class App:
    def __init__(self, root, worker, events, threshold, save_dir):
        self.root, self.worker, self.events = root, worker, events
        self.threshold, self.save_dir = threshold, save_dir
        self.photos = {}   # Tk needs us to keep references to displayed images
        self.busy = False

        root.title("Cat or Dog - XIAO ESP32S3 over USB")
        root.configure(bg=BG)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=8, font=("Segoe UI", 11))
        style.configure("Accent.TButton", background=ACC, foreground="white", font=("Segoe UI", 12, "bold"))
        style.map("Accent.TButton", background=[("disabled", "#2a3550"), ("active", "#6a9cff")])

        panels = tk.Frame(root, bg=BG)
        panels.pack(padx=12, pady=(12, 6))
        self.live = self._panel(panels, "LIVE VIDEO", 0)
        self.snap = self._panel(panels, "SNAPSHOT  (box = what the model sees)", 1)
        result_box = self._card(panels, "RESULT", 2)
        self.result_label = tk.Label(result_box, text="-", font=("Segoe UI", 44, "bold"), bg=CARD, fg=INK)
        self.result_label.pack(pady=(8, 0))
        self.score_label = tk.Label(result_box, text="Press Capture & Classify", font=("Segoe UI", 13), bg=CARD, fg=DIM)
        self.score_label.pack()
        self.timing_label = tk.Label(result_box, text="", font=("Segoe UI", 11), bg=CARD, fg=DIM)
        self.timing_label.pack(pady=(6, 0))
        tk.Label(result_box, text="Model input (96 x 96)", font=("Segoe UI", 9), bg=CARD, fg=DIM).pack(pady=(10, 2))
        self.blank = ImageTk.PhotoImage(Image.new("RGB", (128, 128), "black"))
        self.input_view = tk.Label(result_box, image=self.blank, bg=CARD, bd=0)
        self.input_view.pack()

        controls = tk.Frame(root, bg=BG)
        controls.pack(fill="x", padx=12)
        self.capture_btn = ttk.Button(controls, text="Capture & Classify  (Space)", style="Accent.TButton",
                                      command=self.capture)
        self.capture_btn.pack(side="left")
        ttk.Button(controls, text="Clear", command=self.clear).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Reconnect", command=self.reconnect).pack(side="left", padx=8)
        tk.Label(controls, text="Resolution", bg=BG, fg=DIM).pack(side="left", padx=(16, 4))
        self.res = ttk.Combobox(controls, values=list(RESOLUTIONS), width=11, state="readonly")
        self.res.set("640 x 480")
        self.res.bind("<<ComboboxSelected>>", lambda e: worker.set_resolution(RESOLUTIONS[self.res.get()]))
        self.res.pack(side="left")
        self.rot = tk.BooleanVar(value=False)
        tk.Checkbutton(controls, text="Rotate 180°", variable=self.rot, bg=BG, fg=INK, selectcolor=CARD,
                       activebackground=BG, command=lambda: setattr(worker, "rotate180", self.rot.get())
                       ).pack(side="left", padx=12)
        self.save = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Save captures", variable=self.save, bg=BG, fg=INK, selectcolor=CARD,
                       activebackground=BG).pack(side="left")

        self.status = tk.Label(root, text="Starting...", anchor="w", bg=BG, fg=DIM, font=("Segoe UI", 10))
        self.status.pack(fill="x", padx=12, pady=(6, 12))
        self.fps_text, self.msg, self.msg_error = "", "Starting...", False
        root.bind("<space>", lambda e: self.capture())
        root.after(15, self.poll)

    def _card(self, parent, title, col):
        f = tk.Frame(parent, bg=CARD, highlightbackground="#2a2e37", highlightthickness=1)
        f.grid(row=0, column=col, padx=6, sticky="n")
        tk.Label(f, text=title, bg=CARD, fg=DIM, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        inner = tk.Frame(f, bg=CARD, width=PANEL_W if col < 2 else 300, height=PANEL_H)
        inner.pack(padx=10, pady=(0, 10))
        inner.pack_propagate(False)
        return inner

    def _panel(self, parent, title, col):
        box = self._card(parent, title, col)
        c = tk.Canvas(box, width=PANEL_W, height=PANEL_H, bg="#000", highlightthickness=0)
        c.pack()
        return c

    def show(self, canvas, key, img, box=None, outline=ACC):
        shown, s = fit(img)
        if box:  # draw the region of interest the model uses
            d = ImageDraw.Draw(shown)
            d.rectangle([v * s for v in box], outline=outline, width=2)
        self.photos[key] = ImageTk.PhotoImage(shown)
        canvas.delete("all")
        canvas.create_image(PANEL_W // 2, PANEL_H // 2, image=self.photos[key])

    def capture(self):
        if self.busy:
            return
        self.busy = True
        self.capture_btn.state(["disabled"])
        self.clear(keep_status=True)   # never leave a stale answer on screen
        self.set_status("Capturing the next frame...")
        self.worker.request_capture()

    def clear(self, keep_status=False):
        self.snap.delete("all")
        self.result_label.config(text="-", fg=INK)
        self.score_label.config(text="")
        self.timing_label.config(text="")
        self.input_view.config(image=self.blank)
        if not keep_status:
            self.set_status("Cleared")

    def reconnect(self):
        self.set_status("Reconnecting...")
        self.worker.reconnect()

    def set_status(self, text, error=False):
        self.msg, self.msg_error = text, error
        self.render_status()

    def render_status(self):
        sep = "    |    " if self.fps_text else ""
        self.status.config(text=f"{self.msg}{sep}{self.fps_text}", fg=BAD if self.msg_error else DIM)

    def done(self):
        self.busy = False
        self.capture_btn.state(["!disabled"])

    def poll(self):
        latest_frame = None
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "frame":
                    latest_frame = ev           # only draw the newest frame
                elif kind == "status":
                    self.set_status(ev[1])
                elif kind == "info":
                    print("ESP status:\n" + ev[1])
                elif kind == "capturing":
                    self.set_status("Classifying on the ESP...")
                elif kind == "result":
                    self.on_result(*ev[1:])
                elif kind == "result_error":
                    self.clear(keep_status=True)
                    self.result_label.config(text="Error", fg=BAD)
                    self.score_label.config(text=ev[1])
                    self.set_status(ev[1], error=True)
                    self.done()
                elif kind == "error":
                    self.live.delete("all")
                    self.live.create_text(PANEL_W // 2, PANEL_H // 2, text="No video\n" + ev[1][:120], fill=BAD,
                                          font=("Segoe UI", 12), width=PANEL_W - 40, justify="center")
                    self.fps_text = ""
                    self.set_status("Connection problem - retrying (or press Reconnect). Close Serial Monitor if open.",
                                    error=True)
                    if self.busy:
                        self.done()
        except queue.Empty:
            pass
        if latest_frame:
            _, img, fps, kb = latest_frame
            self.fps_text = f"{fps:4.1f} fps · {img.width}x{img.height} · {kb:.0f} KB/frame"
            self.show(self.live, "live", img, center_square(*img.size))
            if self.msg_error:  # video is flowing again, so the old error no longer applies
                self.set_status("Connected")
            self.render_status()
        self.root.after(15, self.poll)

    def on_result(self, img, pixels, box, scores, seconds):
        self.show(self.snap, "snap", img, box)
        inp = Image.fromarray(pixels).resize((128, 128), Image.NEAREST)  # blocky on purpose: real pixels
        self.photos["inp"] = ImageTk.PhotoImage(inp)
        self.input_view.config(image=self.photos["inp"])
        labels, probs = scores["labels"], scores["probs"]
        best = max(range(len(probs)), key=lambda i: probs[i])
        label = labels[best].capitalize() if probs[best] >= self.threshold else "Not sure"
        self.result_label.config(text=label, fg=INK if label != "Not sure" else DIM)
        self.score_label.config(text="  ·  ".join(f"{l.capitalize()} {p:.0%}" for l, p in zip(labels, probs)))
        self.timing_label.config(text=f"Model on ESP32: {scores['ms']} ms\nButton to result: {seconds:.2f} s")
        self.set_status(f"Done: {label}")
        if self.save.get():
            self.save_capture(img, pixels, label, probs, seconds)
        self.done()

    def save_capture(self, img, pixels, label, probs, seconds):
        self.save_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{label.replace(' ', '_').lower()}"
        img.save(self.save_dir / f"{name}.jpg", quality=92)
        Image.fromarray(pixels).save(self.save_dir / f"{name}_input.png")
        log = self.save_dir / "captures.csv"
        new = not log.exists()
        with log.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["file", "result", "cat", "dog", "seconds"])
            w.writerow([f"{name}.jpg", label, *[f"{p:.4f}" for p in probs], f"{seconds:.3f}"])

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="e.g. COM10 (default: find the XIAO automatically)")
    ap.add_argument("--simulate", action="store_true", help="fake camera + model, no hardware")
    ap.add_argument("--threshold", type=float, default=0.70, help="below this top score, show 'Not sure'")
    args = ap.parse_args()

    events = queue.Queue()
    open_fn = (lambda: FakeEspSerial()) if args.simulate else (lambda: open_serial(args.port))
    worker = CameraWorker(open_fn, events)
    root = tk.Tk()
    App(root, worker, events, args.threshold, HERE / "captures")
    worker.start()
    root.protocol("WM_DELETE_WINDOW", lambda: (worker.stop(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
