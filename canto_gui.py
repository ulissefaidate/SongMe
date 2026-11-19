#!/usr/bin/env python3
"""Canto - live karaoke with a graphical interface.      Start with: python canto_gui.py

The window reuses the analysis functions of canto.py (same folder). Use headphones!

Tools: PySide6 (Qt) for the window, pyqtgraph for the fast scrolling graph,
sounddevice for audio in/out, torchcrepe for the pitch of your voice."""
import sys, json, argparse, contextlib, io, time, warnings
from math import gcd
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly, resample
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import sounddevice as sd
import soundfile as sf
import canto as C

WIN, BEHIND = 8.0, 2.4          # seconds shown on screen / of which "past" (live view)
REVIEW_SPAN = 12.0              # seconds shown when the final trace opens
GOOD, MID, BAD = "#4ade80", "#fbbf24", "#f87171"


def col(e):
    """Error in semitones -> traffic-light colour: green <= 40 cents, yellow <= 1 semitone, red above."""
    return GOOD if e <= .4 else MID if e <= 1 else BAD


class NoteAxis(pg.AxisItem):
    """Vertical axis that shows note names (C4, D4...) instead of raw MIDI numbers."""
    def tickStrings(self, values, scale, spacing):
        return [C.note(v) for v in values]


class KaraokeBar(QtWidgets.QWidget):
    """Karaoke strip: the CURRENT word is fixed in the centre and stays until the next one starts;
    it is filled with blue while it is being sung. Neighbouring words fade and shrink with distance.
    """

    def __init__(self):
        super().__init__()
        self.words, self.starts, self.t = [], np.array([]), 0.0
        self.setMinimumHeight(90)

    def set_time(self, words, starts, t):
        """Give the bar the word list and the current song time, then ask Qt to repaint."""
        self.words, self.starts, self.t = words, starts, t
        self.update()

    def paintEvent(self, ev):
        """Called by Qt whenever the bar must be redrawn (about 30 times per second).

        1. find the current word: the last one whose start time is <= t;
        2. measure the width of each visible word (size depends on its distance from the current one);
        3. shift everything so that the current word is centred;
        4. draw each word with a transparency that decreases with distance;
        5. clip-draw the current word in blue to show how much of it has been sung.
        """
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QtGui.QColor("#1e1e1c"))
        ws, st, n, t = self.words, self.starts, len(self.words), self.t
        if not n:
            return
        idx = max(0, min(int(np.searchsorted(st, t, "right")) - 1, n - 1))
        gap, items, x, cen = 26, [], 0.0, {}
        for i in range(max(0, idx - 7), min(n, idx + 8)):
            d = abs(i - idx)
            fnt = QtGui.QFont()
            fnt.setPixelSize(46 if d == 0 else max(16, 30 - 5 * d)); fnt.setBold(d == 0)
            wd = QtGui.QFontMetrics(fnt).horizontalAdvance(ws[i]["w"])
            items.append((i, d, fnt, wd)); cen[i] = x + wd / 2; x += wd + gap
        off = w / 2 - cen[idx]                              # the current word is always at the centre
        for i, d, fnt, wd in items:
            x0 = cen[i] + off - wd / 2
            if x0 > w or x0 + wd < 0:
                continue
            rect, txt = QtCore.QRectF(x0, 0, wd, h), ws[i]["w"]
            c = QtGui.QColor("#60a5fa" if i < idx else "#ececea")
            c.setAlpha(255 if d == 0 else max(20, int(190 * 0.55 ** (d - 1))))
            p.setFont(fnt); p.setPen(c)
            p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, txt)
            if i == idx:                                    # blue "fill" grows while the word is being sung
                s0 = ws[i]["start"]
                fill = min(1.0, max(0.0, (t - s0) / max(1e-3, ws[i]["end"] - s0)))
                p.save()
                p.setClipRect(QtCore.QRectF(x0, 0, wd * fill, h))
                p.setPen(QtGui.QColor("#60a5fa"))
                p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, txt)
                p.restore()
        p.setPen(QtGui.QPen(QtGui.QColor(96, 165, 250, 170), 2))
        p.drawLine(QtCore.QPointF(w / 2, 4), QtCore.QPointF(w / 2, 14))
        p.drawLine(QtCore.QPointF(w / 2, h - 14), QtCore.QPointF(w / 2, h - 4))


class Writer(io.TextIOBase):
    """File-like object that forwards print() text to a Qt signal, so messages from a background job reach the status bar."""
    def __init__(self, sig):
        self.sig = sig

    def write(self, s):
        if s.strip():
            self.sig.emit(s.strip()[:120])
        return len(s)


class PrepWorker(QtCore.QThread):
    """Runs canto.cmd_prep (separation, melody, alignment) in a background thread so the window never freezes."""
    msg = QtCore.Signal(str)
    done = QtCore.Signal(bool, str)

    def __init__(self, ns):
        super().__init__()
        self.ns = ns

    def run(self):
        try:
            with contextlib.redirect_stdout(Writer(self.msg)):
                C.cmd_prep(self.ns)
            self.done.emit(True, "")
        except Exception as e:
            self.done.emit(False, str(e))


def shift_audio(y, sr, n):
    """Transpose audio by n semitones WITHOUT changing its duration (y: samples x channels).

    Uses rubberband if available (best quality), otherwise librosa's phase vocoder.
    Large shifts (> 4 semitones) may sound a little metallic.
    """
    try:                                            # better quality if rubberband is installed
        import pyrubberband as pyrb
        out = pyrb.pitch_shift(y, sr, n)
    except Exception:
        import librosa
        out = librosa.effects.pitch_shift(np.ascontiguousarray(y.T), sr=sr, n_steps=n).T
    out = out[:len(y)]
    if len(out) < len(y):
        out = np.pad(out, ((0, len(y) - len(out)), (0, 0)))
    return out.astype(np.float32)


class ShiftWorker(QtCore.QThread):
    """Transposes backing and guide voice in a background thread (it takes a few seconds per song)."""
    done = QtCore.Signal(bool, object)

    def __init__(self, back, voc, sr, n):
        super().__init__()
        self.a = (back, voc, sr, n)

    def run(self):
        try:
            back, voc, sr, n = self.a
            b = shift_audio(back, sr, n) if np.any(back) else back
            o = shift_audio(voc, sr, n) if np.any(voc) else voc
            self.done.emit(True, (b, o))
        except Exception as e:
            self.done.emit(False, str(e))


class KeyBox(QtWidgets.QSpinBox):
    """Spin box that shows +2 / -3 / 0 instead of plain numbers."""
    def textFromValue(self, v):
        return f"{v:+d}" if v else "0"


class Win(QtWidgets.QMainWindow):
    """Main window.

    Data (loaded from canto_data/<song>/ by load()):
      rm0 / rm   reference melody, MIDI note per 10 ms frame (original / shifted by the chosen key)
      um         YOUR melody, same time grid, filled in while you sing
      words      [{'w', 'start', 'end', 'midi'}] lyrics (or note names) with timings
      back, voc  backing track and original voice as stereo arrays at the sound card's rate

    While you sing, three things run at the same time:
      1. cb()    audio callback (audio thread): plays the backing, stores the microphone samples
      2. tick()  every 40 ms: estimates your pitch from the latest samples and compares it
      3. draw()  every 33 ms: repaints the graph and the karaoke bar
    Slow jobs (preparation, transposition) run in QThreads so the window stays responsive.
    When the song ends, stop() computes the verdict and opens the navigable final trace.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Canto")
        self.resize(1280, 800)
        self.sp = 0                                    # playback position in samples (this is also our clock)
        self.song = self.lyr = self.rm = self.back = None
        self.words, self.starts, self.running, self.stream = [], np.array([]), False, None
        self.pts, self.wt, self.cur, self.last = [], {}, -2, []
        self.review, self.wst, self.dur, self.dev = False, [], 0.0, None
        self.rmode, self.listening = "song", False
        self.shift, self.cache, self.sug = 0, {}, 0
        self.rm0 = self.words0 = self.back0 = self.voc0 = None

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        v = QtWidgets.QVBoxLayout(root)

        r1 = QtWidgets.QHBoxLayout(); v.addLayout(r1)
        b1 = QtWidgets.QPushButton("Apri brano…"); b1.clicked.connect(self.pick_song)
        b2 = QtWidgets.QPushButton("Testo (.txt / .lrc)…"); b2.clicked.connect(self.pick_lyrics)
        self.lang = QtWidgets.QComboBox(); self.lang.addItems(["it", "en", "es", "fr", "de"])
        self.mode = QtWidgets.QComboBox()
        for lab, k in (("Canzone", "song"), ("Esercizio (voce + base)", "exercise"), ("Esercizio senza base", "plain")):
            self.mode.addItem(lab, k)
        self.bprep = QtWidgets.QPushButton("Prepara"); self.bprep.clicked.connect(self.prepare)
        for w in (b1, QtWidgets.QLabel("Tipo:"), self.mode, b2, QtWidgets.QLabel("Lingua:"), self.lang, self.bprep):
            r1.addWidget(w)
        self.busy = QtWidgets.QProgressBar(); self.busy.setRange(0, 0); self.busy.setMaximumWidth(120); self.busy.hide()
        r1.addWidget(self.busy); r1.addStretch()
        self.status = QtWidgets.QLabel("Scegli un brano dalla libreria (o aprine uno) e premi Prepara.")
        v.addWidget(self.status)

        self.kar = KaraokeBar()
        v.addWidget(self.kar)

        pg.setConfigOptions(antialias=True, background="#141413", foreground="#9a9a94")
        self.plot = pg.PlotWidget(axisItems={"left": NoteAxis(orientation="left")})
        self.vb = self.plot.getViewBox()
        self.vb.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons(); self.plot.showGrid(y=True, alpha=.2)
        self.plot.getAxis("left").setTickSpacing(2, 1)
        self.refc = self.plot.plot(pen=pg.mkPen("#60a5fa", width=5), connect="finite")
        self.usc = pg.ScatterPlotItem(size=8, pen=None); self.plot.addItem(self.usc)
        self.head = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ececea", width=1)); self.plot.addItem(self.head)
        self.vb.sigXRangeChanged.connect(self.on_range)
        v.addWidget(self.plot, 1)
        self.sb = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        self.sb.valueChanged.connect(self.on_scroll); self.sb.hide()
        v.addWidget(self.sb)

        r2 = QtWidgets.QHBoxLayout(); v.addLayout(r2)
        self.btn = QtWidgets.QPushButton("▶ Avvia"); self.btn.clicked.connect(self.toggle); self.btn.setEnabled(False)
        self.btn.setMinimumHeight(38)
        self.inp = QtWidgets.QComboBox()
        api = sd.query_devices(kind="output")["hostapi"]       # PortAudio needs input and output on the same host API (MME, WASAPI...)
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and d["hostapi"] == api:
                self.inp.addItem(f"{i}: {d['name']}"[:50], i)
        k = self.inp.findData(sd.query_hostapis(api)["default_input_device"])
        if k >= 0:
            self.inp.setCurrentIndex(k)
        self.lat = QtWidgets.QSpinBox(); self.lat.setRange(0, 600); self.lat.setValue(80); self.lat.setSuffix(" ms")
        self.guide = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.guide.setRange(0, 100); self.guide.setMaximumWidth(140)
        self.blis = QtWidgets.QPushButton("🔊 Ascolta"); self.blis.clicked.connect(self.listen); self.blis.setMinimumHeight(38)
        r2.addWidget(self.btn); r2.addWidget(self.blis)
        self.key = KeyBox(); self.key.setRange(-12, 12); self.key.setSuffix(" st")
        self.t_key = QtCore.QTimer(self, singleShot=True, interval=700, timeout=self.key_changed)
        self.key.valueChanged.connect(lambda _: self.t_key.start())      # debounce: wait until you stop clicking before transposing
        self.bsug = QtWidgets.QPushButton(); self.bsug.hide(); self.bsug.clicked.connect(self.apply_suggestion)
        for lab, w in (("Microfono:", self.inp), ("Latenza:", self.lat), ("Voce guida:", self.guide),
                       ("Tonalità:", self.key)):
            r2.addWidget(QtWidgets.QLabel(lab)); r2.addWidget(w)
        r2.addWidget(self.bsug)
        r2.addStretch()
        self.rep = QtWidgets.QTextEdit(); self.rep.setReadOnly(True); self.rep.setMaximumHeight(130)
        v.addWidget(self.rep)

        # library panel: recently practised songs, one click to load
        dock = QtWidgets.QDockWidget("Libreria", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.lib = QtWidgets.QListWidget(); self.lib.setMinimumWidth(250)
        self.lib.itemClicked.connect(self.lib_click)
        dock.setWidget(self.lib)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.refresh_lib()

        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.toggle)
        self.t_pitch = QtCore.QTimer(self, interval=40, timeout=self.tick)
        self.t_draw = QtCore.QTimer(self, interval=33, timeout=self.draw); self.t_draw.start()

    # ---------- library ----------
    def refresh_lib(self):
        """Fill the library panel: every song with a ref.json, most recently practised first."""
        rows = []
        for f in C.DATA.glob("*/ref.json"):
            meta = {}
            try:
                meta = json.loads((f.parent / "meta.json").read_text(encoding="utf-8"))
            except Exception:
                pass
            try:
                meta["mode"] = json.loads((f.parent / "info.json").read_text(encoding="utf-8")).get("mode", "song")
            except Exception:
                meta["mode"] = "song"
            rows.append((meta.get("last", 0), f.stat().st_mtime, f.parent.name, meta))
        rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
        self.lib.clear()
        for last, _, name, meta in rows:
            sub = (time.strftime("%d/%m %H:%M", time.localtime(last)) + f" · {meta.get('pct', '?')}% intonato"
                   if last else "mai cantata")
            it = QtWidgets.QListWidgetItem(("🎼 " if meta.get("mode", "song") != "song" else "") + f"{name}\n{sub}")
            it.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            self.lib.addItem(it)

    def lib_click(self, it):
        """One click on a library entry loads that song (no need to prepare it again)."""
        if self.running:
            self.stop()
        self.song = it.data(QtCore.Qt.ItemDataRole.UserRole) + ".mp3"   # only used to derive the cache folder name
        self.lyr = None
        self.load()

    # ---------- files and preparation ----------
    def pick_song(self):
        """Choose an audio file; if it was prepared before, load it right away."""
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Brano", "", "Audio (*.mp3 *.wav *.flac *.m4a *.ogg)")
        if f:
            self.song = f
            self.status.setText(f"Brano: {f}")
            if (C.cdir(f) / "ref.json").exists():
                self.load()

    def pick_lyrics(self):
        """Choose a .txt (plain lyrics) or .lrc (lyrics with [mm:ss] times) file."""
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Testo", "", "Testo (*.txt *.lrc)")
        if f:
            self.lyr = f
            self.status.setText(f"Testo: {f}. Premi Prepara per sincronizzarlo.")

    def prepare(self):
        """Start the preparation job (voice separation, melody, lyric alignment) in the background."""
        if not self.song or not Path(self.song).exists():
            self.status.setText("Il file originale non è disponibile: aprilo con «Apri brano…».")
            return
        ns = argparse.Namespace(song=self.song, lang=self.lang.currentText(), lyrics=None, lrc=None,
                                mode=self.mode.currentData())
        if self.lyr and ns.mode == "song":
            setattr(ns, "lrc" if self.lyr.lower().endswith(".lrc") else "lyrics", self.lyr)
        self.bprep.setEnabled(False); self.btn.setEnabled(False); self.busy.show()
        self.wk = PrepWorker(ns)
        self.wk.msg.connect(self.status.setText)
        self.wk.done.connect(self.prep_done)
        self.wk.start()

    def prep_done(self, ok, err):
        """Called when the background preparation ends: load the result or show the error."""
        self.busy.hide(); self.bprep.setEnabled(True)
        if ok:
            self.load()
            self.refresh_lib()
        else:
            self.status.setText("Errore: " + err)

    def rs(self, y, sr0):
        """Resample audio from rate sr0 to the sound card's rate (polyphase filter, done once at load time)."""
        if sr0 == self.sr:
            return y.astype(np.float32)
        g = gcd(self.sr, sr0)
        return resample_poly(y, self.sr // g, sr0 // g, axis=0).astype(np.float32)

    def load(self):
        """Load a prepared song from the cache folder and reset the whole interface state."""
        self.exit_review()
        d = C.cdir(self.song)
        try:
            ref = json.loads((d / "ref.json").read_text(encoding="utf-8"))
            self.rm0 = np.array([np.nan if x is None else x for x in ref["midi"]])
            self.words0 = ref["words"]
            # everything is resampled to the sound card's rate, so playback needs no conversion at runtime
            self.sr = int(sd.query_devices(kind="output")["default_samplerate"])
            ch = lambda y: np.stack((y, y), 1) if y.ndim == 1 else y[:, :2]
            o, s1 = sf.read(d / "vocals.wav", dtype="float32")
            nv = d / "no_vocals.wav"
            b, s0 = sf.read(nv, dtype="float32") if nv.exists() else (np.zeros_like(o), s1)
            self.rmode = ref.get("mode", "song")
        except Exception as e:
            self.status.setText(f"Impossibile caricare il brano: {e}")
            return
        self.back, self.voc = self.rs(ch(b), s0), self.rs(ch(o), s1)
        n = len(self.back)
        # make sure voice and backing have exactly the same length
        self.voc = np.pad(self.voc, ((0, max(0, n - len(self.voc))), (0, 0)))[:n]
        self.back0, self.voc0, self.cache, self.shift = self.back, self.voc, {}, 0
        self.key.blockSignals(True); self.key.setValue(0); self.key.blockSignals(False)
        self.bsug.hide()
        self.apply_shift()
        self.um = np.full(len(self.rm), np.nan)
        self.dur, self.sp, self.cur, self.pts = len(self.rm) * C.H, 0, -2, []
        self.clear_words(); self.usc.setData(x=[], y=[]); self.rep.clear()
        self.btn.setEnabled(True)
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.rmode)))
        if self.rmode == "plain":
            self.guide.setValue(100)                      # no backing track: the guide is the only audio you hear
        self.status.setText(f"Pronto: {Path(self.song).stem} ({len(self.words)} "
                            f"{'parole' if self.rmode == 'song' else 'note'}). Cuffie su, poi Avvia (o barra spaziatrice).")
        try:                                                  # restore the last key used for this song
            saved = int(json.loads((d / "meta.json").read_text(encoding="utf-8")).get("shift", 0))
        except Exception:
            saved = 0
        if saved:
            self.key.blockSignals(True); self.key.setValue(saved); self.key.blockSignals(False)
            self.begin_shift(saved)

    # ---------- audio and pitch ----------
    def cb(self, indata, outdata, frames, tinfo, status):
        """Audio callback, called by the sound library for every block of ~512 samples.

        It runs in a separate audio thread and must be FAST (no heavy work here).
        It plays the next block of backing (+ guide voice) and stores the microphone samples.
        """
        # p = where we are in the song, n = samples left to play in this block
        p, n = self.sp, max(0, min(frames, len(self.back) - self.sp))
        if n:
            # output = backing + guide voice (slider), clipped to avoid distortion
            outdata[:n] = np.clip(self.back[p:p + n] + (self.guide.value() / 100) * self.voc[p:p + n], -1, 1)
        outdata[n:] = 0
        # ring buffer: keep only the most recent microphone samples (tick() reads the tail)
        self.ring = np.concatenate((self.ring, indata[:, 0]))[-len(self.ring):]
        # the playback position IS our clock: song time = sp / sample rate
        self.sp += frames

    def infer(self, x):
        """Pitch of one 64 ms microphone frame (1024 samples at 16 kHz) with the small CREPE model.

        Returns a MIDI note number, or NaN when the model is not confident.
        """
        import torch, torchcrepe
        # shape (1, 1024): one batch containing one 64 ms frame
        a = torch.from_numpy(x.astype(np.float32))[None].to(self.dev)
        f0, per = torchcrepe.predict(a, 16000, hop_length=1024, fmin=60, fmax=1000, model="tiny",
                                     decoder=torchcrepe.decode.weighted_argmax, return_periodicity=True,
                                     batch_size=1, device=self.dev, pad=False)
        # f = frequency in Hz, p = confidence of the model; unsure frames are rejected
        f, p = float(f0[0, 0]), float(per[0, 0])
        # Hz -> MIDI note number (69 = A4 = 440 Hz); one semitone = 1.0
        return 69 + 12 * np.log2(f / 440) if (p > .3 and f > 0) else np.nan

    def toggle(self):
        """Start/Stop button (and the space bar)."""
        if self.rm is None:
            return
        self.stop() if self.running else self.start()

    def apply_shift(self):
        """Recompute melody, word notes and note names for the chosen key (+N semitones)."""
        n = self.shift
        self.rm = self.rm0 + n
        ws = []
        for w in self.words0:
            w = dict(w)
            if w.get("midi") is not None:
                w["midi"] = w["midi"] + n
                if self.rmode != "song":
                    w["w"] = C.note_it(w["midi"])
            ws.append(w)
        self.words = ws
        self.starts = np.array([w["start"] for w in ws])
        v = self.rm[~np.isnan(self.rm)]
        self.plot.setYRange(np.percentile(v, 2) - 3, np.percentile(v, 98) + 4, padding=0)

    def key_changed(self):
        """Called 0.7 s after the last click on the key box; refuses changes while you are singing."""
        n = self.key.value()
        if self.running or self.rm0 is None or n == self.shift:
            if n != self.shift:
                self.key.blockSignals(True); self.key.setValue(self.shift); self.key.blockSignals(False)
            return
        self.begin_shift(n)

    def begin_shift(self, n):
        """Transpose the audio to n semitones: instant if cached, otherwise in a background thread."""
        self.exit_review(); self.bsug.hide()
        if n == 0 or n in self.cache:
            self.finish_shift(n, self.cache.get(n))
            return
        self.status.setText(f"Cambio tonalità ({n:+d} semitoni)… può richiedere qualche decina di secondi.")
        self.busy.show()
        for w in (self.btn, self.blis, self.bprep, self.key):
            w.setEnabled(False)
        self.sw = ShiftWorker(self.back0, self.voc0, self.sr, n)
        self.sw.done.connect(lambda ok, res, n=n: self.shift_done(n, ok, res))
        self.sw.start()

    def shift_done(self, n, ok, res):
        """Background transposition finished: re-enable the controls and store the result in the cache."""
        self.busy.hide()
        for w in (self.btn, self.blis, self.bprep, self.key):
            w.setEnabled(True)
        if not ok:
            self.status.setText("Cambio tonalità non riuscito: " + str(res))
            self.key.blockSignals(True); self.key.setValue(self.shift); self.key.blockSignals(False)
            return
        self.cache[n] = res
        while len(self.cache) > 4:                            # keep only the most recently used transpositions in memory
            self.cache.pop(next(iter(self.cache)))
        self.finish_shift(n, res)

    def finish_shift(self, n, res):
        """Switch the player to the transposed audio and reset the data of the previous run."""
        self.back, self.voc = (self.back0, self.voc0) if n == 0 else res
        self.shift = n
        self.apply_shift()
        self.um = np.full(len(self.rm), np.nan)
        self.cur, self.pts = -2, []
        self.clear_words(); self.usc.setData(x=[], y=[])
        self.status.setText(f"Tonalità: {n:+d} semitoni rispetto all'originale. Pronto: premi Avvia.")

    def apply_suggestion(self):
        """Apply the key suggested after the last session (button shown by stop())."""
        self.bsug.hide()
        self.key.setValue(self.sug)

    def listen(self):
        """Play the reference (backing + original voice) without recording, to hear the song or exercise first."""
        if self.rm is None or self.running:
            return
        if self.listening:
            sd.stop(); self.listening = False; self.blis.setText("🔊 Ascolta")
            return
        sd.play(np.clip(self.back + self.voc, -1, 1), self.sr)
        self.listening = True; self.blis.setText("■ Ferma")
        QtCore.QTimer.singleShot(int(len(self.back) / self.sr * 1000) + 400, self.listen_done)

    def listen_done(self):
        """Reset the Listen button when the preview has finished."""
        if self.listening:
            self.listening = False; self.blis.setText("🔊 Ascolta")

    def start(self):
        """Start a session: warm up the model, open the full-duplex audio stream, start the pitch timer."""
        self.bsug.hide()
        if self.listening:
            self.listen()
        self.exit_review()
        # "cuda" if an NVIDIA GPU is usable, otherwise "cpu" (slower but works)
        self.dev = C.dev()
        self.status.setText("Carico il modello…"); QtWidgets.QApplication.processEvents()
        self.infer(np.random.randn(1024) * .1)                      # warm-up: the first inference loads the model, so do it before the audio starts
        self.sp, self.pts, self.last = 0, [], []
        self.um[:] = np.nan
        # 0.3 s ring buffer of microphone samples
        self.ring = np.zeros(int(self.sr * .3), np.float32)
        out = sd.query_devices(kind="output")["index"]
        if self.inp.currentData() is None:
            self.status.setText("Nessun microfono compatibile con l'uscita audio predefinita di Windows.")
            return
        try:
            # full-duplex stream: plays the backing and records the microphone with the same clock
            self.stream = sd.Stream(samplerate=self.sr, channels=(1, 2), dtype="float32", blocksize=512,
                                    device=(self.inp.currentData(), out), callback=self.cb)
            self.stream.start()
        except Exception as e:
            self.status.setText(f"Audio non avviato: {e}")
            return
        self.running = True; self.btn.setText("■ Stop"); self.rep.clear()
        self.status.setText("Canta! (verde = intonato, giallo = incerto, rosso = stonato)")
        self.t_pitch.start()

    def tick(self):
        """Every 40 ms: measure your pitch and compare it with the reference.

        1. song time = playback position minus the latency setting (your voice reacts to what you heard);
        2. read the latest 64 ms of microphone audio and ask the model for the pitch;
        3. take the median of the last 3 readings to remove glitches;
        4. compare with the reference notes within +/-120 ms, folding octaves;
        5. store the result: it is drawn immediately and used for the final report.
        """
        if not self.running:
            return
        if self.sp >= len(self.back):
            self.stop(); return
        # song time of what you are singing now = playback position minus audio latency
        t = self.sp / self.sr - self.lat.value() / 1000
        # last 64 ms of microphone audio, expressed at the device sample rate
        x = self.ring[-int(self.sr * 1024 / 16000):]
        # below a minimal volume (RMS) = silence; otherwise resample to 16 kHz and run the model
        m = self.infer(resample(x, 1024)) if np.sqrt(np.mean(x ** 2)) > .006 else np.nan
        # median of the last 3 readings (~120 ms) removes isolated glitches
        self.last = (self.last + [m])[-3:]
        if all(np.isnan(self.last)):
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ms = float(np.nanmedian(self.last))
        i = int(t / C.H)
        if i < 0 or i >= len(self.rm):
            return
        self.um[i] = ms
        w = self.rm[max(0, i - 12):i + 13]; w = w[~np.isnan(w)]
        if len(w):
            # error in semitones folded into [-6, +6): singing an octave away is not a mistake
            dd = ms - w; dd -= 12 * np.round(dd / 12)
            # compare with the closest reference note within +/-120 ms
            k = int(np.argmin(np.abs(dd)))
            self.pts.append((t, w[k] + dd[k], float(dd[k])))

    def stop(self):
        """End of the session: close audio, compute the verdict, save the report and open the final trace."""
        self.t_pitch.stop(); self.running = False; self.btn.setText("▶ Avvia")
        if self.stream:
            self.stream.stop(); self.stream.close(); self.stream = None
        # D = signed error per frame (semitones), AL = your note in the reference octave (for drawing)
        D, AL = C.frame_compare(self.rm, self.um)
        ok = D[~np.isnan(D)]
        if len(ok) < 30:
            self.status.setText("Troppa poca voce rilevata: controlla microfono e volume.")
            return
        med, pct = float(np.median(ok)), 100 * float(np.mean(np.abs(ok) <= .4))
        st = C.word_stats(D, self.words)
        worst = sorted([(s["bad"], i) for i, s in enumerate(st) if s["n"] >= 5], reverse=True)[:8]
        txt = (f"Tonalità del brano: {self.shift:+d} semitoni rispetto all'originale\n" if self.shift else "") + C.verdict(med, pct) + ("\n\nParole" if self.rmode == "song" else "\n\nNote") + " più difficili:\n" + "\n".join(
            f"  {self.words[i]['w']}  ({self.words[i]['start']:.1f}s)  {b * 100:.0f}% fuori, {st[i]['mean']:+.1f} st"
            for b, i in worst)
        self.rep.setPlainText(txt)
        C.plot(self.rm, self.um, AL, D, self.words, st, C.cdir(self.song))
        (C.cdir(self.song) / "meta.json").write_text(
            json.dumps({"last": time.time(), "pct": round(pct), "med": round(med, 2), "shift": self.shift}), encoding="utf-8")
        self.refresh_lib()
        new = max(-12, min(12, self.shift + int(round(med))))
        if abs(med) >= .5 and new != self.shift:               # you are consistently above/below the song: suggest the matching key
            self.sug = new
            self.bsug.setText(f"Applica tonalità suggerita ({new:+d})"); self.bsug.show()
        else:
            self.bsug.hide()
        self.enter_review(D, AL, st)
        self.status.setText("Fine. Scorri il tracciato (trascina o barra in basso), zoom con la rotellina. "
                            "Parole: verde ok, giallo incerte, rosso difficili. PNG in canto_data/…/report.png")

    # ---------- navigable final trace ----------
    def enter_review(self, D, AL, st):
        """Show the WHOLE song: pan with the mouse or the scroll bar, zoom with the wheel."""
        self.review, self.wst = True, st
        ok = np.nonzero(~np.isnan(AL))[0]
        brs = [pg.mkBrush(GOOD), pg.mkBrush(MID), pg.mkBrush(BAD)]
        kk = lambda e: 0 if e <= .4 else 1 if e <= 1 else 2
        self.usc.setData(x=ok * C.H, y=AL[ok], brush=[brs[kk(abs(D[i]))] for i in ok])
        self.refc.setData(np.arange(len(self.rm)) * C.H, self.rm)
        self.clear_words()
        self.head.setVisible(False)
        self.vb.setMouseEnabled(x=True, y=False)
        self.vb.setLimits(xMin=0, xMax=self.dur)
        span = min(REVIEW_SPAN, self.dur)
        self.plot.setXRange(0, span, padding=0)
        self.sb.show()

    def exit_review(self):
        """Go back to the live view (called when a new session or song starts)."""
        if not self.review:
            return
        self.review = False
        self.sb.hide()
        self.vb.setLimits(xMin=None, xMax=None)
        self.vb.setMouseEnabled(x=False, y=False)
        self.head.setVisible(True)
        self.clear_words()
        self.usc.setData(x=[], y=[])

    def on_range(self, vb, rng):
        """The view moved (mouse drag / zoom): keep the scroll bar in sync."""
        if not self.review:
            return
        a, b = rng
        self.sb.blockSignals(True)
        self.sb.setPageStep(max(1, int((b - a) * 10)))
        self.sb.setRange(0, max(0, int((self.dur - (b - a)) * 10)))
        self.sb.setValue(int(a * 10))
        self.sb.blockSignals(False)

    def on_scroll(self, val):
        """The scroll bar moved: move the graph view accordingly."""
        if not self.review:
            return
        a, b = self.vb.viewRange()[0]
        self.plot.setXRange(val / 10, val / 10 + (b - a), padding=0)

    # ---------- drawing ----------
    def show_words(self, a, b):
        """Show the words that fall inside the visible time window as labels on the graph."""
        if self.rmode != "song":
            return                                        # in exercises the note names are in the karaoke bar, not on the graph
        vis = set(np.nonzero((self.starts >= a) & (self.starts < b))[0].tolist())
        for k in list(self.wt):
            if k not in vis:
                self.plot.removeItem(self.wt.pop(k))
        for k in vis - set(self.wt):
            m = self.words[k].get("midi")
            if not m:
                continue
            c = "#ececea"
            if self.review and k < len(self.wst) and self.wst[k]["n"] >= 5:
                bd = self.wst[k]["bad"]
                c = BAD if bd > .5 else MID if bd > .25 else GOOD
            ti = pg.TextItem(self.words[k]["w"], color=c, anchor=(0, 1))
            ti.setPos(self.words[k]["start"], m + .8); self.plot.addItem(ti); self.wt[k] = ti

    def clear_words(self):
        """Remove all word labels from the graph."""
        for ti in self.wt.values():
            self.plot.removeItem(ti)
        self.wt = {}

    def draw(self):
        """Repaint, ~30 times per second.

        Two modes: REVIEW (the view is controlled by the user, the karaoke bar follows the centre of the view)
        and LIVE (the view follows the song position; your points appear as you sing).
        """
        if self.rm is None:
            return
        if self.review:
            a, b = self.vb.viewRange()[0]
            t = (a + b) / 2                                   # word at the centre of the view
            self.show_words(a, b) if b - a <= 40 else self.clear_words()
        else:
            t = (self.sp / self.sr if self.back is not None else 0.0) - self.lat.value() / 1000
            a, b = t - BEHIND, t + WIN - BEHIND
            self.plot.setXRange(a, b, padding=0); self.head.setPos(t)
            i0, i1 = max(0, int(a / C.H)), min(len(self.rm), int(b / C.H))
            if i1 > i0:
                self.refc.setData(np.arange(i0, i1) * C.H, self.rm[i0:i1])
            p = [q for q in self.pts[-300:] if q[0] >= a]
            self.usc.setData(x=[q[0] for q in p], y=[q[1] for q in p],
                             brush=[pg.mkBrush(col(abs(q[2]))) for q in p])
            self.show_words(a, b)
        if len(self.starts):
            self.kar.set_time(self.words, self.starts, t)

    def closeEvent(self, e):
        """Make sure audio is stopped when the window is closed."""
        if self.running:
            self.stop()
        e.accept()


def dark(app):
    """Dark colour palette for the whole application."""
    app.setStyle("Fusion")
    p, R = QtGui.QPalette(), QtGui.QPalette.ColorRole
    for r, c in ((R.Window, "#1e1e1c"), (R.WindowText, "#ececea"), (R.Base, "#141413"), (R.Text, "#ececea"),
                 (R.Button, "#2a2a27"), (R.ButtonText, "#ececea"), (R.Highlight, "#2563eb")):
        p.setColor(r, QtGui.QColor(c))
    app.setPalette(p)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dark(app)
    w = Win(); w.show()
    sys.exit(app.exec())
