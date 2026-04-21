"""ProbeFlow -- graphical interface for Createc-to-Nanonis file conversion."""

import json
import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
import webbrowser

from PIL import Image, ImageTk

CONFIG_PATH     = Path.home() / ".probeflow_config.json"
REPO_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_CUSHION = REPO_ROOT / "src" / "file_cushions"
LOGO_PATH       = REPO_ROOT / "assets" / "logo.png"
GITHUB_URL      = "https://github.com/SPMQT-Lab/Createc-to-Nanonis-file-conversion"

NAVBAR_BG  = "#3273dc"
NAVBAR_FG  = "#ffffff"
NAVBAR_H   = 58

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
    },
}


def load_config() -> dict:
    defaults = {
        "dark_mode": False, "input_dir": "", "output_dir": "",
        "do_png": True, "do_sxm": True, "clip_low": 1.0, "clip_high": 99.0,
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


class QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put(record)


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
    row("Under the supervision of Dr. Peter Jacobson\nThe University of Queensland", 9, color=t["sub_fg"])
    tk.Frame(win, height=1, bg=t["sep"]).pack(fill="x", padx=24, pady=10)
    row("Original code by Rohan Platts", 10, bold=True)
    row("The core conversion algorithms were built by Rohan Platts.\n"
        "This software is a refactored and extended version of his work,\n"
        "developed within SPMQT-Lab.", 9, color=t["sub_fg"])
    tk.Frame(win, height=1, bg=t["sep"]).pack(fill="x", padx=24, pady=10)
    tk.Button(win, text="View on GitHub", bg=NAVBAR_BG, fg=NAVBAR_FG,
              relief="flat", cursor="hand2", font=("Helvetica", 9),
              command=lambda: webbrowser.open(GITHUB_URL)).pack(pady=(0, 18), ipadx=14, ipady=5)


class ProbeFlowGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ProbeFlow")
        self.root.minsize(860, 600)
        self.root.resizable(True, True)

        self.cfg = load_config()
        self.log_queue: queue.Queue = queue.Queue()
        self._running = False
        self._adv_visible = False
        self._theme_widgets: list = []

        self.dark_mode  = tk.BooleanVar(value=self.cfg["dark_mode"])
        self.input_dir  = tk.StringVar(value=self.cfg["input_dir"])
        self.output_dir = tk.StringVar(value=self.cfg["output_dir"])
        self.do_png     = tk.BooleanVar(value=self.cfg["do_png"])
        self.do_sxm     = tk.BooleanVar(value=self.cfg["do_sxm"])
        self.clip_low   = tk.DoubleVar(value=self.cfg["clip_low"])
        self.clip_high  = tk.DoubleVar(value=self.cfg["clip_high"])
        self._status_text = tk.StringVar(value="Ready")
        self._file_count  = tk.StringVar(value="No files selected")

        self._build_ui()
        self._apply_theme()
        self._poll_log()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_navbar()
        self._build_body()
        self._build_statusbar()

    def _build_navbar(self) -> None:
        nav = tk.Frame(self.root, bg=NAVBAR_BG, height=NAVBAR_H)
        nav.pack(fill="x")
        nav.pack_propagate(False)

        # Logo in top-left corner
        try:
            img = Image.open(LOGO_PATH).convert("RGBA")
            img.thumbnail((120, 44), Image.LANCZOS)
            # make white pixels transparent so logo blends into navbar
            data = img.getdata()
            img.putdata([(r, g, b, 0) if r > 220 and g > 220 and b > 220 else (r, g, b, a)
                         for r, g, b, a in data])
            self._nav_logo = ImageTk.PhotoImage(img)
            logo_lbl = tk.Label(nav, image=self._nav_logo, bg=NAVBAR_BG, cursor="hand2")
            logo_lbl.pack(side="left", padx=(8, 0), pady=6)
            logo_lbl.bind("<Button-1>", lambda e: webbrowser.open(GITHUB_URL))
        except Exception:
            pass

        title_frame = tk.Frame(nav, bg=NAVBAR_BG)
        title_frame.pack(side="left", padx=(6, 0))
        tk.Label(title_frame, text="ProbeFlow", font=("Helvetica", 14, "bold"),
                 bg=NAVBAR_BG, fg=NAVBAR_FG).pack(anchor="w")
        tk.Label(title_frame, text="Createc -> Nanonis", font=("Helvetica", 8),
                 bg=NAVBAR_BG, fg="#a8c8f0").pack(anchor="w")

        # Right-side nav buttons
        tk.Button(nav, text="About",
                  bg=NAVBAR_BG, fg=NAVBAR_FG, relief="flat", cursor="hand2",
                  font=("Helvetica", 9), bd=0, padx=10,
                  activebackground="#2563c0", activeforeground=NAVBAR_FG,
                  command=lambda: _show_about(self.root, self.dark_mode.get())
                  ).pack(side="right", pady=14)

        tk.Button(nav, text="GitHub",
                  bg=NAVBAR_BG, fg=NAVBAR_FG, relief="flat", cursor="hand2",
                  font=("Helvetica", 9), bd=0, padx=10,
                  activebackground="#2563c0", activeforeground=NAVBAR_FG,
                  command=lambda: webbrowser.open(GITHUB_URL)
                  ).pack(side="right", pady=14)

        self._theme_btn = tk.Button(
            nav, text="Light mode" if self.dark_mode.get() else "Dark mode",
            bg=NAVBAR_BG, fg=NAVBAR_FG, relief="flat", cursor="hand2",
            font=("Helvetica", 9), bd=0, padx=10,
            activebackground="#2563c0", activeforeground=NAVBAR_FG,
            command=self._toggle_theme,
        )
        self._theme_btn.pack(side="right", pady=14)

    def _build_body(self) -> None:
        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True)
        self._body = body

        # Left: main content panel (~70%)
        self._main = tk.Frame(body)
        self._main.pack(side="left", fill="both", expand=True)

        # Right: sidebar (~280px)
        self._sidebar = tk.Frame(body, width=290)
        self._sidebar.pack(side="right", fill="y")
        self._sidebar.pack_propagate(False)

        self._build_main_panel()
        self._build_sidebar()

    def _build_main_panel(self) -> None:
        p = self._main
        pad = {"padx": 16, "pady": 6}

        # Section: Folders
        self._section_label(p, "Folders")
        self._folder_row(p, "Input folder:",  self.input_dir,  self._browse_input)
        self._folder_row(p, "Output folder:", self.output_dir, self._browse_output)

        self._hsep(p)

        # Section: Log
        log_hdr = tk.Frame(p)
        log_hdr.pack(fill="x", **pad)
        self._section_label(log_hdr, "Conversion log", inline=True)
        tk.Button(log_hdr, text="Clear", relief="flat", cursor="hand2",
                  font=("Helvetica", 8), command=self._clear_log
                  ).pack(side="right")

        self.log_text = tk.Text(p, height=16, wrap="word",
                                relief="flat", bd=0, font=("Courier", 9),
                                state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=16, pady=(2, 8))

    def _build_sidebar(self) -> None:
        s = self._sidebar

        self._section_label(s, "Convert to", padx=12)
        cb_frame = tk.Frame(s)
        cb_frame.pack(fill="x", padx=16, pady=4)
        self.png_cb = tk.Checkbutton(cb_frame, text="PNG preview", variable=self.do_png)
        self.png_cb.pack(anchor="w", pady=2)
        self.sxm_cb = tk.Checkbutton(cb_frame, text="SXM (Nanonis)", variable=self.do_sxm)
        self.sxm_cb.pack(anchor="w", pady=2)

        self._hsep(s)

        # Advanced options (collapsible)
        adv_hdr = tk.Frame(s)
        adv_hdr.pack(fill="x", padx=12, pady=(2, 0))
        self._adv_btn = tk.Button(adv_hdr, text="[+] Advanced options",
                                  relief="flat", bd=0, cursor="hand2", anchor="w",
                                  font=("Helvetica", 9),
                                  command=self._toggle_advanced)
        self._adv_btn.pack(side="left")

        self._adv_frame = tk.Frame(s)
        self._slider_row(self._adv_frame, "Clip low (%):",  self.clip_low,   0.0,  10.0)
        self._slider_row(self._adv_frame, "Clip high (%):", self.clip_high, 90.0, 100.0)

        self._hsep(s)

        # RUN button
        run_frame = tk.Frame(s)
        run_frame.pack(fill="x", padx=16, pady=10)
        self.run_btn = tk.Button(run_frame, text="  RUN  ",
                                 font=("Helvetica", 12, "bold"),
                                 relief="flat", cursor="hand2",
                                 command=self._run)
        self.run_btn.pack(fill="x", ipady=8)

        self._hsep(s)

        # Info section
        info_frame = tk.Frame(s)
        info_frame.pack(fill="x", padx=12, pady=4)
        self._file_count_lbl = tk.Label(info_frame, textvariable=self._file_count,
                                        font=("Helvetica", 8), anchor="w",
                                        wraplength=250, justify="left")
        self._file_count_lbl.pack(anchor="w")

        # Footer attribution (bottom of sidebar)
        footer = tk.Frame(s)
        footer.pack(side="bottom", fill="x", padx=12, pady=8)
        self._footer_lbl = tk.Label(
            footer,
            text="SPMQT-Lab  |  Dr. Peter Jacobson\nThe University of Queensland\nOriginal code by Rohan Platts",
            font=("Helvetica", 7), anchor="center", justify="center",
        )
        self._footer_lbl.pack()

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------

    def _section_label(self, parent, text: str, padx: int = 16, inline: bool = False) -> None:
        kwargs = {"padx": padx, "pady": (10, 2)}
        lbl = tk.Label(parent, text=text, font=("Helvetica", 9, "bold"), anchor="w")
        if inline:
            lbl.pack(side="left")
        else:
            lbl.pack(fill="x", **kwargs)

    def _hsep(self, parent) -> None:
        sep = tk.Frame(parent, height=1)
        sep.pack(fill="x", padx=12, pady=6)
        self._theme_widgets.append((sep, "sep"))

    def _folder_row(self, parent, label: str, var: tk.StringVar, cmd) -> None:
        f = tk.Frame(parent)
        f.pack(fill="x", padx=16, pady=4)
        tk.Label(f, text=label, width=13, anchor="w").pack(side="left")
        e = tk.Entry(f, textvariable=var, relief="flat", bd=2)
        e.pack(side="left", fill="x", expand=True, padx=(0, 6))
        b = tk.Button(f, text="Browse", relief="flat", cursor="hand2", command=cmd,
                      font=("Helvetica", 8))
        b.pack(side="right")
        self._theme_widgets.extend([(e, "entry"), (b, "btn"), (f, "frame")])
        var.trace_add("write", lambda *_: self._update_file_count())

    def _slider_row(self, parent, label: str, var: tk.DoubleVar,
                    from_: float, to: float) -> None:
        f = tk.Frame(parent)
        f.pack(fill="x", padx=16, pady=4)
        tk.Label(f, text=label, width=13, anchor="w").pack(side="left")
        s = tk.Scale(f, variable=var, from_=from_, to=to, resolution=0.5,
                     orient="horizontal", length=180, sliderlength=14,
                     relief="flat", bd=0, highlightthickness=0)
        s.pack(side="left")
        self._theme_widgets.append((s, "slider"))

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        t = THEMES["dark" if self.dark_mode.get() else "light"]
        self.root.configure(bg=t["bg"])

        for frame in (self._body, self._main, self._sidebar, self._adv_frame):
            frame.configure(bg=t["main_bg"] if frame is self._main else
                            t["sidebar_bg"] if frame is self._sidebar else t["bg"])

        self._repaint(self._main,    t, main=True)
        self._repaint(self._sidebar, t, sidebar=True)

        self.log_text.configure(bg=t["log_bg"], fg=t["log_fg"], insertbackground=t["fg"])
        self.log_text.tag_config("ok",   foreground=t["ok_fg"])
        self.log_text.tag_config("err",  foreground=t["err_fg"])
        self.log_text.tag_config("warn", foreground=t["warn_fg"])
        self.log_text.tag_config("info", foreground=t["log_fg"])

        self.run_btn.configure(bg=t["accent_bg"], fg=t["accent_fg"],
                               activebackground=t["accent_bg"], activeforeground=t["accent_fg"])

        self._status_bar.configure(bg=t["status_bg"])
        self._status_lbl.configure(bg=t["status_bg"], fg=t["status_fg"])

        self._footer_lbl.configure(bg=t["sidebar_bg"], fg=t["sub_fg"])
        self._file_count_lbl.configure(bg=t["sidebar_bg"], fg=t["sub_fg"])

    def _repaint(self, widget, t: dict, main: bool = False, sidebar: bool = False) -> None:
        panel_bg = t["main_bg"] if main else (t["sidebar_bg"] if sidebar else t["bg"])
        cls = widget.winfo_class()
        try:
            if cls == "Frame":
                widget.configure(bg=panel_bg)
            elif cls == "Label":
                widget.configure(bg=panel_bg, fg=t["fg"])
            elif cls == "Button":
                widget.configure(bg=t["btn_bg"], fg=t["btn_fg"],
                                 activebackground=panel_bg, activeforeground=t["fg"],
                                 relief="flat")
            elif cls == "Checkbutton":
                widget.configure(bg=panel_bg, fg=t["fg"], selectcolor=t["entry_bg"],
                                 activebackground=panel_bg, activeforeground=t["fg"])
            elif cls == "Entry":
                widget.configure(bg=t["entry_bg"], fg=t["fg"],
                                 insertbackground=t["fg"], relief="flat")
            elif cls == "Scale":
                widget.configure(bg=panel_bg, fg=t["fg"],
                                 troughcolor=t["entry_bg"], activebackground=t["accent_bg"])
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._repaint(child, t, main=main, sidebar=sidebar)

    def _toggle_theme(self) -> None:
        self.dark_mode.set(not self.dark_mode.get())
        self._theme_btn.config(
            text="Light mode" if self.dark_mode.get() else "Dark mode")
        self._apply_theme()

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self._status_bar = bar
        self._status_lbl = tk.Label(bar, textvariable=self._status_text,
                                    font=("Helvetica", 8), anchor="w")
        self._status_lbl.pack(side="left", padx=12)

    # ------------------------------------------------------------------
    # Advanced options toggle
    # ------------------------------------------------------------------

    def _toggle_advanced(self) -> None:
        if self._adv_visible:
            self._adv_frame.pack_forget()
            self._adv_btn.config(text="[+] Advanced options")
        else:
            self._adv_frame.pack(fill="x")
            t = THEMES["dark" if self.dark_mode.get() else "light"]
            self._repaint(self._adv_frame, t, sidebar=True)
            self._adv_btn.config(text="[-] Advanced options")
        self._adv_visible = not self._adv_visible

    # ------------------------------------------------------------------
    # File count helper
    # ------------------------------------------------------------------

    def _update_file_count(self) -> None:
        d = self.input_dir.get().strip()
        if d and Path(d).is_dir():
            n = len(list(Path(d).glob("*.dat")))
            self._file_count.set(f"{n} .dat file(s) in input folder")
            self._status_text.set(f"{n} .dat file(s) found")
        else:
            self._file_count.set("No files selected")
            self._status_text.set("Ready")

    # ------------------------------------------------------------------
    # Folder pickers
    # ------------------------------------------------------------------

    def _browse_input(self) -> None:
        d = filedialog.askdirectory(title="Select input folder containing .dat files")
        if d:
            self.input_dir.set(d)

    def _browse_output(self) -> None:
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_dir.set(d)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

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
                record = self.log_queue.get_nowait()
                msg = record.getMessage()
                tag = ("err" if record.levelno >= logging.ERROR else
                       "warn" if record.levelno == logging.WARNING else
                       "ok" if "[OK]" in msg else "info")
                self._log(msg, tag)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log)

    # ------------------------------------------------------------------
    # Run conversion
    # ------------------------------------------------------------------

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
        self._status_text.set("Converting...")
        self._clear_log()
        handler = QueueHandler(self.log_queue)
        handler.setLevel(logging.DEBUG)
        threading.Thread(target=self._worker, args=(in_dir, out_dir, handler), daemon=True).start()

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
                            convert_dat_to_sxm(dat, sxm_out, DEFAULT_CUSHION, clip_low, clip_high)
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
            self.root.after(0, self._done)

    def _done(self) -> None:
        self._running = False
        self.run_btn.configure(text="  RUN  ", state="normal")
        self._status_text.set("Done")

    def _on_close(self) -> None:
        save_config({
            "dark_mode":  self.dark_mode.get(),
            "input_dir":  self.input_dir.get(),
            "output_dir": self.output_dir.get(),
            "do_png":     self.do_png.get(),
            "do_sxm":     self.do_sxm.get(),
            "clip_low":   self.clip_low.get(),
            "clip_high":  self.clip_high.get(),
        })
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ProbeFlowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
