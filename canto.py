#!/usr/bin/env python3
"""Canto - phase 1: file-based analysis (command line).

What this program does
----------------------
1. ``prep``    separates the voice from the music (Demucs), extracts the vocal melody
               (CREPE neural network) and aligns the lyrics to the audio (WhisperX or .lrc).
2. ``sing``    plays the backing track and records your microphone (use headphones!).
3. ``compare`` compares your voice with the original one, says where you are out of tune
               and whether you should sing higher or lower.

Ideas used everywhere in the code
---------------------------------
* Time is cut into *frames* of 10 ms. Frame number ``i`` means time ``i * H`` seconds.
* A melody is an array with one MIDI note number per frame (NaN = silence / no voice).
  MIDI numbers are a logarithmic pitch scale: 69 = A4 = 440 Hz, and +1 = one semitone (100 cents).
* Singing exactly one octave away from the original is NOT counted as an error:
  differences are folded into the range [-6, +6) semitones.
* Every file produced for a song is cached in ``canto_data/<song name>/``,
  so the slow steps (separation, neural networks) run only once per song.

Usage
-----
  python canto.py prep    song.mp3 --lyrics lyrics.txt [--lrc lyrics.lrc] [--lang it]
  python canto.py prep    vocalise.mp3 --mode exercise   # voice + backing: no lyrics, note names instead of words
  python canto.py prep    vocalise.mp3 --mode plain      # no backing (voice or piano only): no separation
  python canto.py sing    song.mp3 [--seconds 60] [--device N]
  python canto.py compare song.mp3 [--user mine.wav] [--latency 0]
"""
import argparse, json, re, subprocess, sys
from pathlib import Path
import numpy as np

# SR  = sample rate used for analysis (CREPE works at 16 kHz)
# HOP = samples between two analysis frames: 160 / 16000 = 10 ms
# ROW = seconds shown in each row of the PNG report
SR, HOP, ROW = 16000, 160, 30.0
H = HOP / SR                      # duration of one frame in seconds (0.01)
DATA = Path("canto_data")
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def cdir(song):
    """Return (and create) the cache folder of a song: canto_data/<song name>/."""
    d = DATA / Path(song).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def dev():
    """Pick the compute device: the NVIDIA GPU ("cuda") if PyTorch can see one, otherwise the CPU.

    The CPU is much slower but everything still works.
    """
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def note(m):
    """MIDI number -> English note name, e.g. 69 -> 'A4', 60 -> 'C4'."""
    m = int(round(m))
    return f"{NOTES[m % 12]}{m // 12 - 1}"


# ---------- melody extraction ----------
def pitch_track(path):
    """Extract the sung melody of an audio file: one MIDI note number per 10 ms frame.

    A frame is NaN when there is no reliable voice (silence, breath, consonants).
    The neural network (CREPE) looks at 64 ms of audio and outputs the pitch in Hz
    plus a confidence value; we keep only frames that are loud and confident enough.
    """
    import librosa, torch, torchcrepe
    # CREPE expects mono audio at 16 kHz
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    # add a batch dimension and move the audio to the GPU (if any)
    a = torch.tensor(y)[None].to(dev())
    # f0 = pitch in Hz for every 10 ms frame, per = confidence of the model (0..1)
    f0, per = torchcrepe.predict(a, SR, hop_length=HOP, fmin=60, fmax=1000, model="full",
                                 decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                 batch_size=512, device=dev())
    per = torchcrepe.filter.median(per, 5)       # smoothing filters: remove jumps and noise
    f0 = torchcrepe.filter.mean(f0, 5)
    f0, per = f0[0].cpu().numpy(), per[0].cpu().numpy()
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP, center=True)[0]
    n = min(len(f0), len(rms))
    f0, per, rms = f0[:n], per[:n], rms[:n]
    # keep a frame only if the model is confident, it is not silence (> -45 dB) and f0 is valid
    ok = (per > 0.3) & (rms > 10 ** (-45 / 20)) & (f0 > 0)
    # Hz -> MIDI note number (69 = A4 = 440 Hz); rejected frames become NaN
    return np.where(ok, 69 + 12 * np.log2(np.maximum(f0, 1) / 440), np.nan)


# ---------- text -> words with timings ----------
def words_whisperx(vocals, text, lang, dur):
    """Align known lyrics to the audio with WhisperX (*forced alignment*).

    We already know WHAT is sung, so the model only has to find WHEN each word is sung:
    much more reliable than transcribing singing from scratch.
    Returns [{'w': word, 'start': seconds, 'end': seconds}, ...].
    """
    import whisperx
    audio = whisperx.load_audio(str(vocals))
    model, meta = whisperx.load_align_model(language_code=lang, device=dev())
    seg = [{"text": " ".join(text.split()), "start": 0.0, "end": dur}]
    r = whisperx.align(seg, model, meta, audio, dev(), return_char_alignments=False)
    out = []
    for w in r["word_segments"]:
        if w.get("start") is not None and w.get("end") is not None:
            out.append({"w": w["word"], "start": float(w["start"]), "end": float(w["end"])})
    return out


def voiced_cum(midi, bridge=25):
    """Cumulative count of 'voiced' frames (frames where somebody is singing).

    Gaps shorter than ~250 ms (consonants, breaths) are bridged and count as singing.
    With this curve we can map 'time spent singing' back to real time, skipping pauses.
    """
    from scipy.ndimage import binary_closing
    return np.cumsum(binary_closing(~np.isnan(midi), structure=np.ones(bridge, bool)))


def words_lrc(lrc_text, dur, midi=None):
    """Build the word list from an .lrc file ([mm:ss] one line per verse).

    The .lrc gives the start of each verse only. Inside a verse the words are placed
    proportionally to their length, but only over the time where the voice is present.
    """
    rows = []
    for ln in lrc_text.splitlines():
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)", ln.strip())
        if m and m.group(3).strip():
            rows.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).split()))
    cum = voiced_cum(midi) if midi is not None else None
    out = []
    for i, (t, ws) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else min(dur, t + 5)
        tot = sum(len(w) + 1 for w in ws)
        f0 = int(t / H)
        f1 = min(int(end / H) - 1, len(cum) - 1) if cum is not None else 0
        base = cum[f0 - 1] if (cum is not None and 0 < f0 <= len(cum)) else 0
        vs = (cum[f1] - base) if (cum is not None and f0 < len(cum)) else 0
        a = 0
        for w in ws:
            wl = len(w) + 1
            if vs >= 15:                                   # at least 150 ms of voice: use the time actually sung
                u0, u1 = base + vs * a / tot, base + vs * (a + wl) / tot
                s_ = int(np.searchsorted(cum, u0 + 1e-9, "left")) * H
                e_ = (int(np.searchsorted(cum, u1, "left")) + 1) * H
            else:                                          # no voice detected: spread the words evenly
                s_, e_ = t + (end - t) * a / tot, t + (end - t) * (a + wl) / tot
            out.append({"w": w, "start": round(s_, 3), "end": round(max(e_, s_ + .05), 3)})
            a += wl
    return out


def snap_to_voice(words, midi, reach=40):
    """Move word starts/ends onto frames where the voice really is (within 400 ms).

    Alignment tools are often a few tenths of a second off; the vocal melody tells us
    where the sound actually begins and ends.
    """
    v, n, prev = ~np.isnan(midi), len(midi), 0.0
    for w in words:
        i0, i1 = int(w["start"] / H), min(int(w["end"] / H), n - 1)
        if 0 <= i0 < n and not v[i0]:
            k = np.nonzero(v[i0:min(n, i0 + reach)])[0]
            if len(k):
                w["start"] = round((i0 + int(k[0])) * H, 3)
        if 0 <= i1 < n and not v[i1]:
            lo_ = max(0, i1 - reach)
            k = np.nonzero(v[lo_:i1 + 1])[0]
            if len(k):
                w["end"] = round((lo_ + int(k[-1]) + 1) * H, 3)
        w["start"] = max(w["start"], prev); prev = w["start"]
        if w["end"] <= w["start"]:
            w["end"] = round(w["start"] + .05, 3)
    return words


NOMI_IT = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]


def note_it(m):
    """MIDI number -> Italian note name (Do, Re, Mi...), e.g. 60 -> 'Do4'."""
    m = int(round(m))
    return f"{NOMI_IT[m % 12]}{m // 12 - 1}"


def to_wav(src, dst):
    """Convert any audio file to WAV (used by --mode plain, where no separation is needed)."""
    import librosa, soundfile as sf
    y, sr = librosa.load(str(src), sr=None, mono=False)
    sf.write(str(dst), y.T if y.ndim > 1 else y, sr)


def notes_as_words(midi, min_len=0.12, hyst=0.8, gap=6):
    """For exercises: cut the melody into stable notes; the 'words' become note names.

    A new note starts when the pitch moves more than `hyst` semitones away from the
    running mean of the current note (so vibrato does not split a note). Notes shorter
    than `min_len` seconds are discarded; gaps up to `gap` frames do not end a note.
    """
    out, seg, s, last = [], [], 0, 0

    def close():
        if seg and (last - s + 1) * H >= min_len:
            md = float(np.median(seg))
            out.append({"w": note_it(md), "start": round(s * H, 3), "end": round((last + 1) * H, 3),
                        "midi": round(md, 2)})

    for j, m in enumerate(midi):
        if np.isnan(m):
            if seg and j - last > gap:
                close(); seg = []
            continue
        if not seg or abs(m - np.mean(seg)) > hyst:
            close(); seg, s = [m], j
        else:
            seg.append(m)
        last = j
    close()
    return out


def cmd_prep(a):
    """Command 'prep': the slow, one-time preparation of a song.

    mode = song      -> separate voice/backing (Demucs), lyrics aligned to the audio
    mode = exercise  -> separate voice/backing, no lyrics: note names are used instead
    mode = plain     -> no separation (a cappella voice or single-note piano)
    """
    d = cdir(a.song)
    mode = getattr(a, "mode", "song")             # song | exercise | plain
    voc, info = d / "vocals.wav", d / "info.json"
    old = (json.loads(info.read_text(encoding="utf-8")).get("mode") if info.exists()
           else ("song" if voc.exists() else None))
    # if the same file is prepared again with a different mode, the cached stems are stale: delete them
    if old is not None and (old == "plain") != (mode == "plain"):
        for s_ in ("vocals", "no_vocals"):
            (d / f"{s_}.wav").unlink(missing_ok=True)
    # heavy step, done only once per song
    if not voc.exists():
        if mode == "plain":
            print("Conversione dell'audio...")
            to_wav(a.song, voc)
        else:
            print("Separazione voce con Demucs (la prima volta scarica il modello)...")
            subprocess.check_call([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
                                   "-d", dev(), "-o", str(d / "sep"), "--filename", "{stem}.{ext}", a.song])
            for s_ in ("vocals", "no_vocals"):
                (d / "sep" / "htdemucs" / f"{s_}.wav").replace(d / f"{s_}.wav")
    print("Estrazione melodia...")
    midi = pitch_track(voc)
    dur = len(midi) * H
    words = []
    if mode != "song":
        words = notes_as_words(midi)
    elif getattr(a, "lrc", None):
        words = words_lrc(Path(a.lrc).read_text(encoding="utf-8"), dur, midi)
    elif getattr(a, "lyrics", None):
        print("Allineamento del testo (WhisperX)...")
        words = words_whisperx(voc, Path(a.lyrics).read_text(encoding="utf-8"), a.lang, dur)
    if mode == "song" and words:
        words = snap_to_voice(words, midi)
    for w in words:                               # median note of every word
        seg = midi[int(w["start"] / H): int(w["end"] / H) + 1]
        seg = seg[~np.isnan(seg)]
        w["midi"] = round(float(np.median(seg)), 2) if len(seg) >= 3 else None
    # everything the GUI needs later (melody + words with timings) goes in one JSON file
    ref = {"hop": H, "dur": dur, "mode": mode, "midi": [None if np.isnan(x) else round(float(x), 2) for x in midi],
           "words": words}
    (d / "ref.json").write_text(json.dumps(ref), encoding="utf-8")
    info.write_text(json.dumps({"mode": mode}), encoding="utf-8")
    print(f"Fatto: {len(words)} {'parole' if mode == 'song' else 'note'}, "
          f"{int(np.sum(~np.isnan(midi)))} frame vocali. Cartella: {d}")


def cmd_sing(a):
    """Command 'sing': play the backing (or the guide, if there is no backing) and record the mic.

    Playback and recording share the same clock, so they stay aligned
    (apart from the hardware latency, see --latency in 'compare').
    """
    import sounddevice as sd, soundfile as sf
    d = cdir(a.song)
    nv = d / "no_vocals.wav"
    back, sr = sf.read(nv if nv.exists() else d / "vocals.wav", dtype="float32")
    if a.seconds:
        back = back[: int(a.seconds * sr)]
    input("Metti le cuffie, premi Invio e canta insieme alla base... ")
    rec = sd.playrec(back, samplerate=sr, channels=1, device=a.device)
    sd.wait()
    sf.write(d / "user.wav", rec[:, 0], sr)
    print("Registrazione salvata. Ora: python canto.py compare", a.song)


def frame_compare(ref, usr, tol=12):
    """Compare two melodies frame by frame.

    Returns (D, AL):
      D  = signed error in semitones for every frame of `usr` (+ = too high, NaN = no data)
      AL = your note moved to the reference octave (used to draw the points)
    `tol` frames of timing tolerance (12 -> +/-120 ms) are allowed, because nobody sings
    exactly on the beat.
    """
    d = np.full(len(usr), np.nan)
    al = np.full(len(usr), np.nan)
    for i, u in enumerate(usr):
        if np.isnan(u):
            continue
        # reference notes within +/-120 ms of this frame (small timing differences are tolerated)
        w = ref[max(0, i - tol): i + tol + 1]
        w = w[~np.isnan(w)]
        if not len(w):
            continue
        dd = u - w
        dd -= 12 * np.round(dd / 12)              # fold into [-6, +6): ignore octave differences
        k = np.argmin(np.abs(dd))
        # pick the closest reference note; d = signed error, al = your note in the reference octave
        d[i], al[i] = dd[k], w[k] + dd[k]
    return d, al


def verdict(med, pct):
    """Turn the median error into a human-readable verdict: sing higher, lower, or fine.

    med = median error in semitones, pct = % of frames within 40 cents.
    """
    t = f"Intonazione: {pct:.0f}% dei frame entro ±40 cent | scostamento medio {med:+.2f} semitoni\n"
    if abs(med) < 0.35:
        return t + "Tonalità giusta: errori solo locali."
    if med > 0:
        return t + f"Sei ~{abs(med):.1f} semitoni SOPRA il brano: scendi, oppure trasponi il brano su di {round(abs(med))}."
    return t + f"Sei ~{abs(med):.1f} semitoni SOTTO il brano: sali, oppure trasponi il brano giù di {round(abs(med))}."


def word_stats(D, ws):
    """Per-word statistics of the error D: number of frames, share of 'out of tune' frames (> 0.5 semitone), mean error."""
    st = []
    for w in ws:
        x = D[int(w["start"] / H): int(w["end"] / H) + 1]
        x = x[~np.isnan(x)]
        st.append({"n": len(x), "bad": float(np.mean(np.abs(x) > 0.5)) if len(x) else 0,
                   "mean": float(np.mean(x)) if len(x) else 0})
    return st


def cmd_compare(a):
    """Command 'compare': compare a recording of your voice with the prepared song.

    Prints the verdict and the hardest words, and saves a PNG report.
    """
    d = cdir(a.song)
    ref = json.loads((d / "ref.json").read_text(encoding="utf-8"))
    rm = np.array([np.nan if x is None else x for x in ref["midi"]])
    upath = Path(a.user) if a.user else d / "user.wav"
    # drop the first frames to compensate for the latency of your audio chain
    um = pitch_track(upath)[int(a.latency / 1000 / H):]
    D, AL = frame_compare(rm, um)
    ok = D[~np.isnan(D)]
    if len(ok) < 50:
        sys.exit("Troppa poca voce rilevata: controlla microfono e volume.")
    med = float(np.median(ok))
    pct = 100 * np.mean(np.abs(ok) <= 0.4)
    print("\n" + verdict(med, pct))
    ws = ref["words"]
    st = word_stats(D, ws)
    worst = sorted([(s["bad"], i) for i, s in enumerate(st) if s["n"] >= 5], reverse=True)[:12]
    if worst:
        print(f"\n{'Parole' if ref.get('mode', 'song') == 'song' else 'Note'} più difficili:")
        print(f"{'parola':<18}{'tempo':>8}{'fuori':>8}{'scost.':>9}{'nota brano':>12}")
        for b, i in worst:
            w, s = ws[i], st[i]
            nn = note(w["midi"]) if w.get("midi") else "-"
            print(f"{w['w'][:17]:<18}{w['start']:>7.1f}s{b * 100:>7.0f}%{s['mean']:>+8.1f}{nn:>12}")
    plot(rm, um, AL, D, ws, st, d)


def plot(rm, um, AL, D, ws, st, d):
    """Save the full-song report as an image: blue = reference melody, dots = your voice
    (green/orange/red by error), words written at their note.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dur = len(rm) * H
    rows = int(np.ceil(dur / ROW))
    t, tu = np.arange(len(rm)) * H, np.arange(len(um)) * H
    v = rm[~np.isnan(rm)]
    lo, hi = np.percentile(v, 2) - 3, np.percentile(v, 98) + 4
    col = lambda e: "#16a34a" if e <= .4 else "#d97706" if e <= 1 else "#dc2626"
    fig, axs = plt.subplots(rows, 1, figsize=(15, 2.8 * rows), squeeze=False)
    for r in range(rows):
        ax = axs[r][0]
        a, b = r * ROW, (r + 1) * ROW
        ax.set_xlim(a, b); ax.set_ylim(lo, hi)
        ticks = list(range(int(lo), int(hi) + 1, 2))
        ax.set_yticks(ticks); ax.set_yticklabels([note(x) for x in ticks], fontsize=7)
        ax.grid(alpha=.2)
        m = (t >= a) & (t < b)
        ax.plot(t[m], rm[m], ".", ms=2, color="#2563eb")
        mu = (tu >= a) & (tu < b) & ~np.isnan(AL)
        ax.scatter(tu[mu], AL[mu], s=4, c=[col(abs(x)) for x in D[mu]])
        for w, s in zip(ws, st):
            if a <= w["start"] < b and w.get("midi"):
                ax.text(w["start"], w["midi"] + 1.3, w["w"], fontsize=7, rotation=40,
                        color=col(1.5 if s["bad"] > .5 else .8 if s["bad"] > .25 else 0))
    fig.tight_layout()
    out = d / "report.png"
    fig.savefig(out, dpi=110)
    print(f"\nGrafico salvato: {out}  (blu = brano; verde/arancio/rosso = tua voce)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("prep"); x.add_argument("song"); x.add_argument("--lyrics"); x.add_argument("--lrc")
    x.add_argument("--mode", choices=["song", "exercise", "plain"], default="song")
    x.add_argument("--lang", default="it"); x.set_defaults(f=cmd_prep)
    x = s.add_parser("sing"); x.add_argument("song"); x.add_argument("--seconds", type=float)
    x.add_argument("--device", type=int); x.set_defaults(f=cmd_sing)
    x = s.add_parser("compare"); x.add_argument("song"); x.add_argument("--user")
    x.add_argument("--latency", type=float, default=0.0, help="ms da scartare all'inizio della tua registrazione")
    x.set_defaults(f=cmd_compare)
    a = p.parse_args()
    a.f(a)
