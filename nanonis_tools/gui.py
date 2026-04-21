"""ProbeFlow -- graphical interface for Createc-to-Nanonis file conversion."""

import json
import logging
import queue
import re as _re
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable, List, Optional
import webbrowser

import numpy as np
from PIL import Image, ImageTk

CONFIG_PATH     = Path.home() / ".probeflow_config.json"
REPO_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_CUSHION = REPO_ROOT / "src" / "file_cushions"
LOGO_PATH       = REPO_ROOT / "assets" / "logo.png"
GITHUB_URL      = "https://github.com/SPMQT-Lab/Createc-to-Nanonis-file-conversion"

NAVBAR_BG = "#3273dc"
NAVBAR_FG = "#ffffff"
NAVBAR_H  = 58

THEMES = {
    "dark": {
        "bg":         "#1e1e2e",
        "fg":         "#cdd6f4",
        "entry_bg":   "#313244",
        "btn_bg":     "#45475a",
        "btn_fg":     "#cdd6f4",
        "log_bg":     "#181825",
        "log_fg":     "#cdd6f4",
        "ok_fg":      "#a6e3a1",
        "err_fg":     "#f38ba8",
        "warn_fg":    "#fab387",
        "accent_bg":  "#89b4fa",
        "accent_fg":  "#1e1e2e",
        "sep":        "#45475a",
        "sub_fg":     "#6c7086",
        "sidebar_bg": "#181825",
        "main_bg":    "#1e1e2e",
        "status_bg":  "#313244",
        "status_fg":  "#6c7086",
        "card_bg":    "#313244",
        "card_sel":   "#4a4f6a",
        "card_fg":    "#cdd6f4",
        "tab_act":    "#313244",
        "tab_inact":  "#1e1e2e",
        "tree_bg":    "#181825",
        "tree_fg":    "#cdd6f4",
        "tree_sel":   "#45475a",
        "tree_head":  "#313244",
    },
    "light": {
        "bg":         "#f8f9fa",
        "fg":         "#1e1e2e",
        "entry_bg":   "#ffffff",
        "btn_bg":     "#e0e0e0",
        "btn_fg":     "#1e1e2e",
        "log_bg":     "#ffffff",
        "log_fg":     "#1e1e2e",
        "ok_fg":      "#1a7a1a",
        "err_fg":     "#c0392b",
        "warn_fg":    "#b07800",
        "accent_bg":  "#3273dc",
        "accent_fg":  "#ffffff",
        "sep":        "#dee2e6",
        "sub_fg":     "#6c757d",
        "sidebar_bg": "#eff5fb",
        "main_bg":    "#eef6fc",
        "status_bg":  "#f5f5f5",
        "status_fg":  "#6c757d",
        "card_bg":    "#d0e4f4",
        "card_sel":   "#b8d4ee",
        "card_fg":    "#1e1e2e",
        "tab_act":    "#ffffff",
        "tab_inact":  "#d8eaf8",
        "tree_bg":    "#ffffff",
        "tree_fg":    "#1e1e2e",
        "tree_sel":   "#cce0f5",
        "tree_head":  "#e8f0f8",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Colormaps
# ─────────────────────────────────────────────────────────────────────────────

def _make_lut(name: str) -> np.ndarray:
    x = np.linspace(0, 1, 256)
    if name == "hot":
        r = np.clip(x * 3.0,       0.0, 1.0)
        g = np.clip(x * 3.0 - 1.0, 0.0, 1.0)
        b = np.clip(x * 3.0 - 2.0, 0.0, 1.0)
    elif name == "gray":
        r = g = b = x
    elif name == "viridis":
        r = np.interp(x, [0, 0.25, 0.5, 0.75, 1], [0.267, 0.127, 0.128, 0.479, 0.993])
        g = np.interp(x, [0, 0.25, 0.5, 0.75, 1], [0.005, 0.290, 0.566, 0.798, 0.906])
        b = np.interp(x, [0, 0.25, 0.5, 0.75, 1], [0.329, 0.528, 0.551, 0.271, 0.144])
    elif name == "plasma":
        r = np.interp(x, [0, 0.25, 0.5, 0.75, 1], [0.050, 0.490, 0.798, 0.974, 0.940])
        g = np.interp(x, [0, 0.25, 0.5, 0.75, 1], [0.030, 0.011, 0.200, 0.556, 0.975])
        b = np.interp(x, [0, 0.25, 0.5, 0.75, 1], [0.530, 0.680, 0.360, 0.035, 0.131])
    else:
        r = np.clip(x * 3.0, 0.0, 1.0)
        g = np.clip(x * 3.0 - 1.0, 0.0, 1.0)
        b = np.clip(x * 3.0 - 2.0, 0.0, 1.0)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)

_LUTS: dict = {}
CMAP_DISPLAY = ["Hot", "Gray", "Viridis", "Plasma"]
CMAP_KEY     = {d: d.lower() for d in CMAP_DISPLAY}

def _get_lut(name: str) -> np.ndarray:
    if name not in _LUTS:
        _LUTS[name] = _make_lut(name)
    return _LUTS[name]


# ─────────────────────────────────────────────────────────────────────────────
# SXM data model & rendering
# ─────────────────────────────────────────────────────────────────────────────

PLANE_NAMES = ["Z fwd", "Z bwd", "I fwd", "I bwd"]

@dataclass
class SxmFile:
    path: Path
    stem: str
    Nx:   int = 512
    Ny:   int = 512


def parse_sxm_header(sxm_path: Path) -> dict:
    params: dict = {}
    current_key: Optional[str] = None
    buf: List[str] = []

    def _flush():
        if current_key is not None:
            params[current_key] = " ".join(buf).strip()

    try:
        with open(sxm_path, "rb") as fh:
            for raw in fh:
                if raw.strip() == b":SCANIT_END:":
                    break
                line = raw.decode("latin-1", errors="replace").rstrip("\r\n")
                if line.startswith(":") and line.endswith(":") and len(line) > 2:
                    _flush()
                    current_key = line[1:-1]
                    buf = []
                elif current_key is not None:
                    s = line.strip()
                    if s:
                        buf.append(s)
        _flush()
    except Exception:
        pass
    return params


def _sxm_dims(hdr: dict):
    nums = [int(x) for x in _re.findall(r"\d+", hdr.get("SCAN_PIXELS", ""))]
    return (nums[0], nums[1]) if len(nums) >= 2 else (512, 512)


def render_sxm_plane(
    sxm_path:  Path,
    plane_idx: int   = 0,
    colormap:  str   = "hot",
    clip_low:  float = 1.0,
    clip_high: float = 99.0,
    size:      tuple = (148, 116),
) -> Optional[Image.Image]:
    """Read one data plane from an SXM file and return a colormapped PIL Image."""
    try:
        hdr = parse_sxm_header(sxm_path)
        Nx, Ny = _sxm_dims(hdr)
        if Nx <= 0 or Ny <= 0:
            return None

        data_offset = int((DEFAULT_CUSHION / "data_offset.txt").read_text().strip())
        raw = sxm_path.read_bytes()

        plane_bytes = Ny * Nx * 4
        start = data_offset + plane_idx * plane_bytes
        if start + plane_bytes > len(raw):
            return None

        arr = np.frombuffer(raw[start : start + plane_bytes], dtype=">f4").copy()
        arr = arr.reshape((Ny, Nx))

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        vmin = float(np.percentile(finite, clip_low))
        vmax = float(np.percentile(finite, clip_high))
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max())
        if vmax <= vmin:
            return None

        safe = np.where(np.isfinite(arr), arr, vmin).astype(np.float64)
        u8 = np.clip((safe - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        colored = _get_lut(colormap)[u8]
        img = Image.fromarray(colored, mode="RGB")
        img.thumbnail(size, Image.LANCZOS)
        return img
    except Exception:
        return None


def scan_sxm_folder(root: Path) -> List[SxmFile]:
    entries = []
    for sxm in sorted(Path(root).rglob("*.sxm")):
        try:
            hdr  = parse_sxm_header(sxm)
            Nx, Ny = _sxm_dims(hdr)
            entries.append(SxmFile(path=sxm, stem=sxm.stem, Nx=Nx, Ny=Ny))
        except Exception:
            entries.append(SxmFile(path=sxm, stem=sxm.stem))
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    defaults = {
        "dark_mode":  False,
        "input_dir":  "",
        "output_dir": "",
        "do_png":     False,
        "do_sxm":     True,
        "clip_low":   1.0,
        "clip_high":  99.0,
        "colormap":   "Hot",
    }
    try:
        if CONFIG_PATH.exists():
            defaults.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    return defaults


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Queue log handler
# ─────────────────────────────────────────────────────────────────────────────

class QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put(record)


# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail grid
# ─────────────────────────────────────────────────────────────────────────────

class ThumbnailGrid(tk.Frame):
    CARD_W = 168
    CARD_H = 152
    IMG_W  = 152
    IMG_H  = 120
    GAP    = 10

    def __init__(self, parent, on_select: Callable, theme: dict, **kw):
        super().__init__(parent, **kw)
        self._on_select  = on_select
        self._t          = theme
        self._entries:   List[SxmFile] = []
        self._photos:    dict = {}
        self._selected:  Optional[str] = None
        self._load_token = object()
        self._colormap   = "hot"
        self._cached_cols = 1

        self._canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self._vsb    = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vsb.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._vsb.pack(side="right", fill="y")

        self._canvas.bind("<Configure>", lambda e: self._layout())
        self._canvas.bind("<MouseWheel>", self._scroll)
        self._canvas.bind("<Button-4>",   self._scroll)
        self._canvas.bind("<Button-5>",   self._scroll)

    def load(self, entries: List[SxmFile], colormap: str = "hot") -> None:
        self._entries    = entries
        self._colormap   = colormap
        self._photos     = {}
        self._selected   = None
        self._load_token = object()
        self._canvas.delete("all")
        self._layout()
        token = self._load_token
        threading.Thread(
            target=self._load_bg, args=(list(entries), colormap, token), daemon=True
        ).start()

    def set_colormap(self, colormap: str) -> None:
        if colormap != self._colormap:
            self.load(self._entries, colormap)

    def _load_bg(self, entries: List[SxmFile], colormap: str, token) -> None:
        for entry in entries:
            if token is not self._load_token:
                return
            img = render_sxm_plane(entry.path, 0, colormap,
                                   size=(self.IMG_W, self.IMG_H))
            if img is not None:
                self._canvas.after(0, self._place_photo, entry.stem, img, token)

    def _place_photo(self, stem: str, img: Image.Image, token) -> None:
        if token is not self._load_token:
            return
        try:
            self._photos[stem] = ImageTk.PhotoImage(img)
        except Exception:
            return
        self._redraw_card(stem)

    def _layout(self) -> None:
        cw = self._canvas.winfo_width()
        if cw < 10:
            self.after(50, self._layout)
            return
        cols = max(1, (cw + self.GAP) // (self.CARD_W + self.GAP))
        self._cached_cols = cols
        t = self._t
        self._canvas.delete("all")

        for i, entry in enumerate(self._entries):
            self._draw_card(i, entry, cols, t)

        total_rows = max(1, (len(self._entries) + cols - 1) // cols)
        total_h    = total_rows * (self.CARD_H + self.GAP) + self.GAP
        self._canvas.configure(scrollregion=(0, 0, cw, max(total_h, 1)))
        self._canvas.tag_bind("card", "<Button-1>", self._on_click)

    def _draw_card(self, i: int, entry: SxmFile, cols: int, t: dict) -> None:
        row, col = divmod(i, cols)
        x0 = col * (self.CARD_W + self.GAP) + self.GAP
        y0 = row * (self.CARD_H + self.GAP) + self.GAP
        x1, y1 = x0 + self.CARD_W, y0 + self.CARD_H
        sel = (entry.stem == self._selected)
        tag = f"s:{entry.stem}"

        self._canvas.create_rectangle(
            x0, y0, x1, y1,
            fill=t["card_sel"] if sel else t["card_bg"],
            outline=t["accent_bg"] if sel else t["sep"],
            width=3 if sel else 1,
            tags=("card", tag),
        )
        photo = self._photos.get(entry.stem)
        if photo:
            self._canvas.create_image(
                x0 + self.CARD_W // 2, y0 + self.IMG_H // 2 + 4,
                image=photo, tags=("card", tag),
            )
        else:
            # Placeholder while loading
            self._canvas.create_rectangle(
                x0 + 8, y0 + 4, x1 - 8, y0 + self.IMG_H + 4,
                fill=t["entry_bg"] if "entry_bg" in t else "#444",
                outline="", tags=("card", tag),
            )
            self._canvas.create_text(
                x0 + self.CARD_W // 2, y0 + self.IMG_H // 2 + 4,
                text="loading...", font=("Helvetica", 7),
                fill=t["sub_fg"], tags=("card", tag),
            )
        label = entry.stem if len(entry.stem) <= 22 else entry.stem[:20] + ".."
        self._canvas.create_text(
            x0 + self.CARD_W // 2, y1 - 12,
            text=label, font=("Helvetica", 7),
            fill=t["accent_fg"] if sel else t["card_fg"],
            tags=("card", tag),
        )

    def _redraw_card(self, stem: str) -> None:
        idx = next((i for i, e in enumerate(self._entries) if e.stem == stem), None)
        if idx is None:
            return
        self._canvas.delete(f"s:{stem}")
        self._draw_card(idx, self._entries[idx], self._cached_cols, self._t)
        self._canvas.tag_bind("card", "<Button-1>", self._on_click)

    def _on_click(self, event: tk.Event) -> None:
        items = self._canvas.find_closest(event.x, event.y)
        if not items:
            return
        for tag in self._canvas.gettags(items[0]):
            if tag.startswith("s:"):
                stem = tag[2:]
                for entry in self._entries:
                    if entry.stem == stem:
                        self._selected = stem
                        self._layout()
                        self._on_select(entry)
                        return

    def _scroll(self, event: tk.Event) -> None:
        if event.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(int(-event.delta / 120), "units")

    def apply_theme(self, t: dict) -> None:
        self._t = t
        self._canvas.configure(bg=t["main_bg"])
        self._layout()


# ─────────────────────────────────────────────────────────────────────────────
# About popup
# ─────────────────────────────────────────────────────────────────────────────

def _show_about(parent: tk.Tk, dark: bool) -> None:
    t = THEMES["dark" if dark else "light"]
    win = tk.Toplevel(parent)
    win.title("About ProbeFlow")
    win.resizable(False, False)
    win.configure(bg=t["bg"])
    win.grab_set()

    try:
        img = Image.open(LOGO_PATH).convert("RGBA")
        img.thumbnail((300, 100), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=photo, bg=t["bg"])
        lbl.image = photo
        lbl.pack(pady=(18, 6))
    except Exception:
        pass

    def row(text, size=10, bold=False, color=None):
        tk.Label(win, text=text,
                 font=("Helvetica", size, "bold" if bold else "normal"),
                 bg=t["bg"], fg=color or t["fg"],
                 wraplength=360, justify="center").pack(pady=2, padx=24)

    row("ProbeFlow", 15, bold=True)
    row("Createc -> Nanonis File Conversion", 10, color=t["sub_fg"])
    tk.Frame(win, height=1, bg=t["sep"]).pack(fill="x", padx=24, pady=10)
    row("Developed at SPMQT-Lab", 10, bold=True)
    row("Under the supervision of Dr. Peter Jacobson\nThe University of Queensland",
        9, color=t["sub_fg"])
    tk.Frame(win, height=1, bg=t["sep"]).pack(fill="x", padx=24, pady=10)
    row("Original code by Rohan Platts", 10, bold=True)
    row("The core conversion algorithms were built by Rohan Platts.\n"
        "This software is a refactored and extended version of his work,\n"
        "developed within SPMQT-Lab.", 9, color=t["sub_fg"])
    tk.Frame(win, height=1, bg=t["sep"]).pack(fill="x", padx=24, pady=10)
    tk.Button(win, text="View on GitHub", bg=NAVBAR_BG, fg=NAVBAR_FG,
              relief="flat", cursor="hand2", font=("Helvetica", 9),
              command=lambda: webbrowser.open(GITHUB_URL)
              ).pack(pady=(0, 18), ipadx=14, ipady=5)


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class ProbeFlowGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ProbeFlow")
        self.root.minsize(960, 650)
        self.root.resizable(True, True)

        self.cfg = load_config()
        self.log_queue: queue.Queue = queue.Queue()
        self._running = False
        self._adv_vis = False
        self._mode    = "browse"

        self.dark_mode  = tk.BooleanVar(value=self.cfg["dark_mode"])
        self.input_dir  = tk.StringVar(value=self.cfg["input_dir"])
        self.output_dir = tk.StringVar(value=self.cfg["output_dir"])
        self.do_png     = tk.BooleanVar(value=self.cfg.get("do_png", False))
        self.do_sxm     = tk.BooleanVar(value=self.cfg.get("do_sxm", True))
        self.clip_low   = tk.DoubleVar(value=self.cfg["clip_low"])
        self.clip_high  = tk.DoubleVar(value=self.cfg["clip_high"])
        self._cmap_var  = tk.StringVar(value=self.cfg.get("colormap", "Hot"))
        self._status    = tk.StringVar(value="Open a folder to browse scans")
        self._sel_info  = tk.StringVar(value="")
        self._n_loaded  = 0

        self._ch_photos: List[Optional[ImageTk.PhotoImage]] = [None] * 4

        self._build_ui()
        self._apply_theme()
        self._poll_log()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Build ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._build_navbar()
        self._build_body()
        self._build_statusbar()

    def _build_navbar(self) -> None:
        nav = tk.Frame(self.root, bg=NAVBAR_BG, height=NAVBAR_H)
        nav.pack(fill="x")
        nav.pack_propagate(False)
        self._nav = nav

        # Logo
        try:
            img = Image.open(LOGO_PATH).convert("RGBA")
            img.thumbnail((120, 44), Image.LANCZOS)
            data = img.getdata()
            img.putdata([(r, g, b, 0) if r > 220 and g > 220 and b > 220 else (r, g, b, a)
                         for r, g, b, a in data])
            self._nav_logo = ImageTk.PhotoImage(img)
            lbl = tk.Label(nav, image=self._nav_logo, bg=NAVBAR_BG, cursor="hand2")
            lbl.pack(side="left", padx=(8, 0), pady=6)
            lbl.bind("<Button-1>", lambda e: webbrowser.open(GITHUB_URL))
        except Exception:
            pass

        tf = tk.Frame(nav, bg=NAVBAR_BG)
        tf.pack(side="left", padx=(6, 0))
        tk.Label(tf, text="ProbeFlow", font=("Helvetica", 14, "bold"),
                 bg=NAVBAR_BG, fg=NAVBAR_FG).pack(anchor="w")
        tk.Label(tf, text="Createc -> Nanonis", font=("Helvetica", 8),
                 bg=NAVBAR_BG, fg="#a8c8f0").pack(anchor="w")

        def _nb(text, cmd):
            return tk.Button(nav, text=text, bg=NAVBAR_BG, fg=NAVBAR_FG,
                             relief="flat", cursor="hand2", font=("Helvetica", 9),
                             bd=0, padx=10, activebackground="#2563c0",
                             activeforeground=NAVBAR_FG, command=cmd)

        _nb("About",  lambda: _show_about(self.root, self.dark_mode.get())).pack(side="right", pady=14)
        _nb("GitHub", lambda: webbrowser.open(GITHUB_URL)).pack(side="right", pady=14)
        _nb("Open folder", self._open_browse_folder).pack(side="right", pady=14)
        self._theme_btn = _nb(
            "Light mode" if self.dark_mode.get() else "Dark mode",
            self._toggle_theme,
        )
        self._theme_btn.pack(side="right", pady=14)

    def _build_body(self) -> None:
        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True)
        self._body = body

        # Left: tab bar + content panels
        left = tk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self._left = left

        tabbar = tk.Frame(left)
        tabbar.pack(fill="x")
        self._tabbar = tabbar
        self._tab_browse = tk.Button(tabbar, text="Browse",
                                     font=("Helvetica", 9, "bold"),
                                     relief="flat", cursor="hand2", bd=0,
                                     padx=16, pady=6,
                                     command=lambda: self._switch_mode("browse"))
        self._tab_browse.pack(side="left")
        self._tab_convert = tk.Button(tabbar, text="Convert",
                                      font=("Helvetica", 9, "bold"),
                                      relief="flat", cursor="hand2", bd=0,
                                      padx=16, pady=6,
                                      command=lambda: self._switch_mode("convert"))
        self._tab_convert.pack(side="left")

        # Browse panel (default)
        self._browse_frame = tk.Frame(left)
        self._grid = ThumbnailGrid(
            self._browse_frame, self._on_entry_select,
            THEMES["dark" if self.dark_mode.get() else "light"],
        )
        self._grid.pack(fill="both", expand=True)

        # Convert panel
        self._conv_frame = tk.Frame(left)
        self._build_convert_panel(self._conv_frame)

        # Show browse by default
        self._browse_frame.pack(fill="both", expand=True)

        # Right sidebar (300px)
        self._sidebar = tk.Frame(body, width=300)
        self._sidebar.pack(side="right", fill="y")
        self._sidebar.pack_propagate(False)

        self._browse_sidebar = tk.Frame(self._sidebar)
        self._build_browse_sidebar(self._browse_sidebar)

        self._conv_sidebar = tk.Frame(self._sidebar)
        self._build_convert_sidebar(self._conv_sidebar)

        # Default: browse sidebar
        self._browse_sidebar.pack(fill="both", expand=True)

    # ── Convert panel ────────────────────────────────────────────────────────

    def _build_convert_panel(self, p: tk.Frame) -> None:
        self._folder_row(p, "Input folder:",  self.input_dir,  self._browse_input)
        self._folder_row(p, "Output folder:", self.output_dir, self._browse_output)
        tk.Frame(p, height=1).pack(fill="x", padx=12, pady=6)

        log_hdr = tk.Frame(p)
        log_hdr.pack(fill="x", padx=16, pady=(2, 0))
        tk.Label(log_hdr, text="Conversion log", font=("Helvetica", 9, "bold"),
                 anchor="w").pack(side="left")
        tk.Button(log_hdr, text="Clear", relief="flat", cursor="hand2",
                  font=("Helvetica", 8), command=self._clear_log).pack(side="right")

        self.log_text = tk.Text(p, height=14, wrap="word", relief="flat", bd=0,
                                font=("Courier", 9), state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=16, pady=(2, 8))

    def _build_convert_sidebar(self, s: tk.Frame) -> None:
        self._slbl(s, "Output format")
        cbf = tk.Frame(s)
        cbf.pack(fill="x", padx=16, pady=4)
        self.png_cb = tk.Checkbutton(cbf, text="PNG preview",   variable=self.do_png)
        self.sxm_cb = tk.Checkbutton(cbf, text="SXM (Nanonis)", variable=self.do_sxm)
        self.png_cb.pack(anchor="w", pady=2)
        self.sxm_cb.pack(anchor="w", pady=2)
        tk.Frame(s, height=1).pack(fill="x", padx=12, pady=6)

        adv_hdr = tk.Frame(s)
        adv_hdr.pack(fill="x", padx=12)
        self._adv_btn = tk.Button(adv_hdr, text="[+] Advanced",
                                  relief="flat", bd=0, cursor="hand2",
                                  font=("Helvetica", 9), anchor="w",
                                  command=self._toggle_adv)
        self._adv_btn.pack(side="left")
        self._adv_frame = tk.Frame(s)
        self._slider_row(self._adv_frame, "Clip low (%):",  self.clip_low,   0.0,  10.0)
        self._slider_row(self._adv_frame, "Clip high (%):", self.clip_high, 90.0, 100.0)
        tk.Frame(s, height=1).pack(fill="x", padx=12, pady=6)

        rf = tk.Frame(s)
        rf.pack(fill="x", padx=16, pady=10)
        self.run_btn = tk.Button(rf, text="  RUN  ",
                                 font=("Helvetica", 12, "bold"),
                                 relief="flat", cursor="hand2", command=self._run)
        self.run_btn.pack(fill="x", ipady=8)
        tk.Frame(s, height=1).pack(fill="x", padx=12, pady=6)

        self._fcount_var = tk.StringVar(value="")
        tk.Label(s, textvariable=self._fcount_var, font=("Helvetica", 8),
                 anchor="w", wraplength=260, justify="left").pack(fill="x", padx=14)
        self.input_dir.trace_add("write", lambda *_: self._update_count())

        tk.Label(s,
                 text="SPMQT-Lab  |  Dr. Peter Jacobson\nThe University of Queensland\nOriginal code by Rohan Platts",
                 font=("Helvetica", 7), justify="center",
                 ).pack(side="bottom", pady=8)

    # ── Browse sidebar ────────────────────────────────────────────────────────

    def _build_browse_sidebar(self, s: tk.Frame) -> None:
        # ── Header: file info ────────────────────────────────────────────────
        hf = tk.Frame(s)
        hf.pack(fill="x", padx=10, pady=(10, 4))
        self._sel_name = tk.Label(hf, text="No scan selected",
                                  font=("Helvetica", 9, "bold"), anchor="w",
                                  wraplength=270, justify="left")
        self._sel_name.pack(fill="x")
        self._sel_dim = tk.Label(hf, text="", font=("Helvetica", 8), anchor="w")
        self._sel_dim.pack(fill="x")

        tk.Frame(s, height=1).pack(fill="x", padx=10, pady=4)

        # ── Colormap selector ────────────────────────────────────────────────
        cf = tk.Frame(s)
        cf.pack(fill="x", padx=12, pady=(0, 6))
        tk.Label(cf, text="Colormap:", font=("Helvetica", 8), anchor="w",
                 width=10).pack(side="left")
        self._cmap_cb = ttk.Combobox(cf, textvariable=self._cmap_var,
                                     values=CMAP_DISPLAY, state="readonly", width=9)
        self._cmap_cb.pack(side="left", padx=4)
        self._cmap_cb.bind("<<ComboboxSelected>>", self._on_colormap_change)

        tk.Frame(s, height=1).pack(fill="x", padx=10, pady=4)

        # ── 4-channel thumbnails (2x2) ───────────────────────────────────────
        self._slbl(s, "Channels")
        self._ch_outer = tk.Frame(s)
        self._ch_outer.pack(fill="x", padx=8, pady=4)
        self._ch_labels: List[tk.Label] = []
        self._ch_name_labels: List[tk.Label] = []
        for i, name in enumerate(PLANE_NAMES):
            r, c = divmod(i, 2)
            cell = tk.Frame(self._ch_outer)
            cell.grid(row=r * 2, column=c, padx=4, pady=2, sticky="nsew")
            img_lbl = tk.Label(cell, relief="flat", bd=1)
            img_lbl.pack()
            nm_lbl = tk.Label(cell, text=name, font=("Helvetica", 7), anchor="center")
            nm_lbl.pack()
            self._ch_labels.append(img_lbl)
            self._ch_name_labels.append(nm_lbl)
        self._ch_outer.grid_columnconfigure(0, weight=1)
        self._ch_outer.grid_columnconfigure(1, weight=1)

        tk.Frame(s, height=1).pack(fill="x", padx=10, pady=4)

        # ── Metadata table ────────────────────────────────────────────────────
        meta_top = tk.Frame(s)
        meta_top.pack(fill="x", padx=10, pady=(4, 2))
        self._slbl(meta_top, "Metadata", inline=True)
        self._search_var = tk.StringVar()
        search_entry = tk.Entry(meta_top, textvariable=self._search_var,
                                font=("Helvetica", 8), width=12, relief="flat", bd=1)
        search_entry.pack(side="right")
        self._search_var.trace_add("write", lambda *_: self._filter_metadata())

        meta_frame = tk.Frame(s)
        meta_frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        style = ttk.Style()
        style.configure("Meta.Treeview",
                        rowheight=18, font=("Helvetica", 8),
                        borderwidth=0, relief="flat")
        style.configure("Meta.Treeview.Heading",
                        font=("Helvetica", 8, "bold"))

        self._tree = ttk.Treeview(meta_frame, columns=("p", "v"), show="headings",
                                  height=14, style="Meta.Treeview")
        self._tree.heading("p", text="Parameter")
        self._tree.heading("v", text="Value")
        self._tree.column("p", width=110, anchor="w", stretch=False)
        self._tree.column("v", width=160, anchor="w")

        meta_vsb = ttk.Scrollbar(meta_frame, orient="vertical",
                                  command=self._tree.yview)
        self._tree.configure(yscrollcommand=meta_vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        meta_vsb.pack(side="right", fill="y")

        self._meta_rows: List[tuple] = []   # (param, value) pairs

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self._statusbar = bar
        self._status_lbl = tk.Label(bar, textvariable=self._status,
                                    font=("Helvetica", 8), anchor="w")
        self._status_lbl.pack(side="left", padx=12)
        self._sel_info_lbl = tk.Label(bar, textvariable=self._sel_info,
                                      font=("Helvetica", 8), anchor="center")
        self._sel_info_lbl.pack(side="left", expand=True)

    # ── Mode switching ────────────────────────────────────────────────────────

    def _switch_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        if mode == "browse":
            self._conv_frame.pack_forget()
            self._browse_frame.pack(fill="both", expand=True)
            self._conv_sidebar.pack_forget()
            self._browse_sidebar.pack(fill="both", expand=True)
            n = len(self._grid._entries)
            self._status.set(f"{n} scan(s) loaded" if n else "Open a folder to browse scans")
        else:
            self._browse_frame.pack_forget()
            self._conv_frame.pack(fill="both", expand=True)
            self._browse_sidebar.pack_forget()
            self._conv_sidebar.pack(fill="both", expand=True)
            self._update_count()
        self._update_tabs()
        self._apply_theme()

    def _update_tabs(self) -> None:
        t = THEMES["dark" if self.dark_mode.get() else "light"]
        for btn, name in ((self._tab_browse, "browse"),
                          (self._tab_convert, "convert")):
            active = (self._mode == name)
            btn.configure(bg=t["tab_act"] if active else t["tab_inact"], fg=t["fg"])

    # ── Browse actions ────────────────────────────────────────────────────────

    def _open_browse_folder(self) -> None:
        d = filedialog.askdirectory(title="Open folder containing .sxm files")
        if not d:
            return
        self._switch_mode("browse")
        entries = scan_sxm_folder(Path(d))
        cmap = CMAP_KEY.get(self._cmap_var.get(), "hot")
        self._grid.load(entries, cmap)
        self._n_loaded = len(entries)
        self._status.set(f"Selected: 0 / {self._n_loaded}")
        self._clear_browse_sidebar()

    def _on_entry_select(self, entry: SxmFile) -> None:
        self._sel_name.configure(text=entry.stem)
        self._sel_dim.configure(text=f"{entry.Nx} x {entry.Ny} px")
        self._sel_info.set(f"{entry.stem}.sxm  |  Z / Current  |  4 channels")

        idx = next((i for i, e in enumerate(self._grid._entries)
                    if e.stem == entry.stem), 0) + 1
        self._status.set(f"Selected: {idx} / {self._n_loaded}")

        self._load_channel_thumbnails(entry)
        self._load_metadata(entry)

    def _load_channel_thumbnails(self, entry: SxmFile) -> None:
        t = THEMES["dark" if self.dark_mode.get() else "light"]
        cmap = CMAP_KEY.get(self._cmap_var.get(), "hot")

        def _do():
            for i, lbl in enumerate(self._ch_labels):
                img = render_sxm_plane(entry.path, i, cmap, size=(110, 86))
                if img:
                    photo = ImageTk.PhotoImage(img)
                    self._ch_photos[i] = photo
                    lbl.configure(image=photo, bg=t["sidebar_bg"])
                else:
                    self._ch_photos[i] = None
                    lbl.configure(image="", bg=t["sidebar_bg"])

        threading.Thread(target=_do, daemon=True).start()

    def _load_metadata(self, entry: SxmFile) -> None:
        hdr = parse_sxm_header(entry.path)
        # Build ordered rows: key SXM fields first, then the rest
        priority = [
            "REC_DATE", "REC_TIME", "SCAN_PIXELS", "SCAN_RANGE",
            "SCAN_OFFSET", "SCAN_ANGLE", "SCAN_DIR", "BIAS",
            "REC_TEMP", "ACQ_TIME", "SCAN_TIME", "COMMENT",
            "Clip_percentile_Lower", "Clip_percentile_Higher",
        ]
        rows = []
        seen = set()
        for k in priority:
            if k in hdr and hdr[k].strip():
                rows.append((k, hdr[k].strip()))
                seen.add(k)
        for k, v in hdr.items():
            if k not in seen and v.strip():
                rows.append((k, v.strip()))

        self._meta_rows = rows
        self._filter_metadata()

    def _filter_metadata(self) -> None:
        query = self._search_var.get().lower()
        self._tree.delete(*self._tree.get_children())
        for param, value in self._meta_rows:
            if not query or query in param.lower() or query in value.lower():
                self._tree.insert("", "end", values=(param, value))

    def _clear_browse_sidebar(self) -> None:
        self._sel_name.configure(text="No scan selected")
        self._sel_dim.configure(text="")
        self._sel_info.set("")
        t = THEMES["dark" if self.dark_mode.get() else "light"]
        for i, lbl in enumerate(self._ch_labels):
            self._ch_photos[i] = None
            lbl.configure(image="", bg=t["sidebar_bg"])
        self._meta_rows = []
        self._tree.delete(*self._tree.get_children())

    def _on_colormap_change(self, _=None) -> None:
        cmap = CMAP_KEY.get(self._cmap_var.get(), "hot")
        self._grid.set_colormap(cmap)
        # Reload channel thumbnails if something is selected
        sel = self._grid._selected
        if sel:
            entry = next((e for e in self._grid._entries if e.stem == sel), None)
            if entry:
                self._load_channel_thumbnails(entry)

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _slbl(self, parent, text: str, inline: bool = False) -> None:
        lbl = tk.Label(parent, text=text, font=("Helvetica", 9, "bold"), anchor="w")
        if inline:
            lbl.pack(side="left")
        else:
            lbl.pack(fill="x", padx=14, pady=(8, 2))

    def _folder_row(self, parent, label: str, var: tk.StringVar, cmd) -> None:
        f = tk.Frame(parent)
        f.pack(fill="x", padx=16, pady=4)
        tk.Label(f, text=label, width=13, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=var, relief="flat", bd=2).pack(
            side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(f, text="Browse", relief="flat", cursor="hand2",
                  font=("Helvetica", 8), command=cmd).pack(side="right")

    def _slider_row(self, parent, label: str, var: tk.DoubleVar,
                    from_: float, to: float) -> None:
        f = tk.Frame(parent)
        f.pack(fill="x", padx=16, pady=4)
        tk.Label(f, text=label, width=13, anchor="w").pack(side="left")
        tk.Scale(f, variable=var, from_=from_, to=to, resolution=0.5,
                 orient="horizontal", length=170, sliderlength=14,
                 relief="flat", bd=0, highlightthickness=0).pack(side="left")

    # ── Theme ────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        t = THEMES["dark" if self.dark_mode.get() else "light"]
        self.root.configure(bg=t["main_bg"])

        for w in (self._body, self._left, self._tabbar,
                  self._browse_frame, self._conv_frame):
            try:
                w.configure(bg=t["main_bg"])
            except Exception:
                pass

        for w in (self._sidebar, self._browse_sidebar, self._conv_sidebar,
                  self._adv_frame, self._ch_outer):
            try:
                w.configure(bg=t["sidebar_bg"])
            except Exception:
                pass

        self._repaint(self._conv_frame,    t, "main")
        self._repaint(self._conv_sidebar,  t, "sidebar")
        self._repaint(self._browse_sidebar, t, "sidebar")
        self._repaint(self._tabbar,         t, "main")

        if hasattr(self, "log_text"):
            self.log_text.configure(bg=t["log_bg"], fg=t["log_fg"],
                                    insertbackground=t["fg"])
            for tag, col in (("ok",   t["ok_fg"]),  ("err",  t["err_fg"]),
                             ("warn", t["warn_fg"]), ("info", t["log_fg"])):
                self.log_text.tag_config(tag, foreground=col)

        if hasattr(self, "run_btn"):
            self.run_btn.configure(bg=t["accent_bg"], fg=t["accent_fg"],
                                   activebackground=t["accent_bg"],
                                   activeforeground=t["accent_fg"])

        self._statusbar.configure(bg=t["status_bg"])
        self._status_lbl.configure(bg=t["status_bg"], fg=t["status_fg"])
        self._sel_info_lbl.configure(bg=t["status_bg"], fg=t["status_fg"])

        # ttk treeview style
        style = ttk.Style()
        style.configure("Meta.Treeview",
                        background=t["tree_bg"], foreground=t["tree_fg"],
                        fieldbackground=t["tree_bg"])
        style.map("Meta.Treeview",
                  background=[("selected", t["tree_sel"])],
                  foreground=[("selected", t["tree_fg"])])
        style.configure("Meta.Treeview.Heading",
                        background=t["tree_head"], foreground=t["tree_fg"])

        self._grid.apply_theme(t)
        self._update_tabs()

    def _repaint(self, widget, t: dict, zone: str = "main") -> None:
        bg = t["main_bg"] if zone == "main" else t["sidebar_bg"]
        cls = widget.winfo_class()
        try:
            if cls == "Frame":
                widget.configure(bg=bg)
            elif cls == "Label":
                widget.configure(bg=bg, fg=t["fg"])
            elif cls == "Button":
                widget.configure(bg=t["btn_bg"], fg=t["btn_fg"],
                                 activebackground=bg, activeforeground=t["fg"],
                                 relief="flat")
            elif cls == "Checkbutton":
                widget.configure(bg=bg, fg=t["fg"], selectcolor=t["entry_bg"],
                                 activebackground=bg, activeforeground=t["fg"])
            elif cls == "Entry":
                widget.configure(bg=t["entry_bg"], fg=t["fg"],
                                 insertbackground=t["fg"], relief="flat")
            elif cls == "Scale":
                widget.configure(bg=bg, fg=t["fg"],
                                 troughcolor=t["entry_bg"],
                                 activebackground=t["accent_bg"])
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._repaint(child, t, zone)

    def _toggle_theme(self) -> None:
        self.dark_mode.set(not self.dark_mode.get())
        self._theme_btn.configure(
            text="Light mode" if self.dark_mode.get() else "Dark mode")
        self._apply_theme()

    # ── Advanced toggle ───────────────────────────────────────────────────────

    def _toggle_adv(self) -> None:
        if self._adv_vis:
            self._adv_frame.pack_forget()
            self._adv_btn.configure(text="[+] Advanced")
        else:
            self._adv_frame.pack(fill="x")
            t = THEMES["dark" if self.dark_mode.get() else "light"]
            self._repaint(self._adv_frame, t, "sidebar")
            self._adv_btn.configure(text="[-] Advanced")
        self._adv_vis = not self._adv_vis

    # ── File count ────────────────────────────────────────────────────────────

    def _update_count(self) -> None:
        d = self.input_dir.get().strip()
        if d and Path(d).is_dir():
            n = len(list(Path(d).glob("*.dat")))
            self._fcount_var.set(f"{n} .dat file(s) in input folder")
            self._status.set(f"{n} .dat file(s) found")
        else:
            self._fcount_var.set("")
            self._status.set("Ready")

    # ── Folder pickers ────────────────────────────────────────────────────────

    def _browse_input(self) -> None:
        d = filedialog.askdirectory(title="Select input folder with .dat files")
        if d:
            self.input_dir.set(d)

    def _browse_output(self) -> None:
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_dir.set(d)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, msg: str, tag: str = "info") -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _poll_log(self) -> None:
        try:
            while True:
                rec = self.log_queue.get_nowait()
                msg = rec.getMessage()
                tag = ("err"  if rec.levelno >= logging.ERROR  else
                       "warn" if rec.levelno == logging.WARNING else
                       "ok"   if "[OK]" in msg else "info")
                self._log(msg, tag)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log)

    # ── Conversion ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        if self._running:
            return
        in_dir  = self.input_dir.get().strip()
        out_dir = self.output_dir.get().strip()
        if not in_dir:
            self._log("ERROR: Please select an input folder.", "err"); return
        if not out_dir:
            self._log("ERROR: Please select an output folder.", "err"); return
        if not self.do_png.get() and not self.do_sxm.get():
            self._log("ERROR: Select at least one output format.", "err"); return
        if not Path(in_dir).is_dir():
            self._log(f"ERROR: Input folder not found: {in_dir}", "err"); return

        self._running = True
        self.run_btn.configure(text="  Running...  ", state="disabled")
        self._status.set("Converting...")
        self._clear_log()
        handler = QueueHandler(self.log_queue)
        handler.setLevel(logging.DEBUG)
        threading.Thread(target=self._worker, args=(in_dir, out_dir, handler),
                         daemon=True).start()

    def _worker(self, in_dir: str, out_dir: str, handler: QueueHandler) -> None:
        logger = logging.getLogger("nanonis_tools")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        clip_low  = self.clip_low.get()
        clip_high = self.clip_high.get()
        in_path   = Path(in_dir)
        out_path  = Path(out_dir)
        try:
            if self.do_png.get():
                from nanonis_tools.dats_to_pngs import main as png_main
                logger.info("-- PNG conversion --")
                png_main(src=in_path, out_root=out_path / "png",
                         clip_low=clip_low, clip_high=clip_high, verbose=True)

            if self.do_sxm.get():
                from nanonis_tools.dat_sxm_cli import convert_dat_to_sxm
                logger.info("-- SXM conversion --")
                files = sorted(in_path.glob("*.dat"))
                if not files:
                    logger.warning("No .dat files found in %s", in_path)
                else:
                    sxm_out = out_path / "sxm"
                    sxm_out.mkdir(parents=True, exist_ok=True)
                    errors = {}
                    logger.info("Found %d .dat file(s)", len(files))
                    for i, dat in enumerate(files, 1):
                        logger.info("[%d/%d] %s ...", i, len(files), dat.name)
                        try:
                            convert_dat_to_sxm(dat, sxm_out, DEFAULT_CUSHION,
                                               clip_low, clip_high)
                        except Exception as exc:
                            logger.error("FAILED %s: %s", dat.name, exc)
                            errors[dat.name] = str(exc)
                    if errors:
                        import json as _j
                        (sxm_out / "errors.json").write_text(_j.dumps(errors, indent=2))
                        logger.warning("%d file(s) failed -- see errors.json", len(errors))
                    else:
                        logger.info("All SXM files processed successfully.")
                    logger.info("Output: %s", sxm_out)
        except Exception as exc:
            logger.error("Unexpected error: %s", exc)
        finally:
            logger.removeHandler(handler)
            self.root.after(0, lambda: self._done(out_dir))

    def _done(self, out_dir: str) -> None:
        self._running = False
        self.run_btn.configure(text="  RUN  ", state="normal")
        sxm_dir = Path(out_dir) / "sxm"
        entries = scan_sxm_folder(sxm_dir) if sxm_dir.exists() else []
        if entries:
            cmap = CMAP_KEY.get(self._cmap_var.get(), "hot")
            self._grid.load(entries, cmap)
            self._n_loaded = len(entries)
            self._switch_mode("browse")
            self._status.set(f"Done -- Selected: 0 / {self._n_loaded}")
        else:
            self._status.set("Done")

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        save_config({
            "dark_mode":  self.dark_mode.get(),
            "input_dir":  self.input_dir.get(),
            "output_dir": self.output_dir.get(),
            "do_png":     self.do_png.get(),
            "do_sxm":     self.do_sxm.get(),
            "clip_low":   self.clip_low.get(),
            "clip_high":  self.clip_high.get(),
            "colormap":   self._cmap_var.get(),
        })
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ProbeFlowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
