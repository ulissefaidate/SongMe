# 🎤 Canto — a singing coach for your PC

Sing along with a song and see **in real time where you go out of tune**, with the lyrics shown karaoke-style.
When you finish, you get a verdict — **should you sing higher or lower?** — and the words that were hardest for you.

> 📘 **New to Python?** Follow the foolproof, step-by-step guide: **[INSTALL.md](INSTALL.md)**
> (NVIDIA GPU, CPU-only, macOS, Linux).
>
> ℹ️ The on-screen texts of the app are in **Italian**. The tables below give the English meaning of every control.

## What it does

| Feature | How |
|---|---|
| Separates voice and music from a song | Demucs |
| Extracts the sung melody (one note every 10 ms, filtered) | CREPE (torchcrepe) |
| Syncs the lyrics word by word | WhisperX (forced alignment) or an `.lrc` file |
| Compares your voice with the original | Difference in semitones, octaves ignored |
| Live karaoke with a scrolling graph | PySide6 + pyqtgraph |
| Transposes the song (−12 … +12 semitones) | rubberband or librosa |
| Navigable graph of the whole performance | Pan and zoom after the song ends |
| Library of recent songs, one click to reload | Cache folder `canto_data/` |
| Exercises and vocalises (no lyrics, with or without backing) | Note names instead of words |

## Requirements

- Python 3.11 recommended (3.10–3.12 work), **ffmpeg** in the PATH
- An NVIDIA GPU with CUDA is recommended; **without a GPU it works too**, only the one-time preparation of a song is slower (see [INSTALL.md](INSTALL.md), section 7)
- **Headphones** (otherwise the microphone also records the backing track)

## Installation (short version)

The full guide is in [INSTALL.md](INSTALL.md). The short version, on Windows with an NVIDIA GPU:

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate

# 1) PyTorch FIRST, pinned to 2.8.0 (cu128 = CUDA 12.8; for CPU use .../whl/cpu)
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 2) lock the versions, so no other package can replace PyTorch:
#    create a file named constraints.txt containing exactly these two lines
#        torch==2.8.0
#        torchaudio==2.8.0

# 3) the rest
pip install -r requirements.txt -c constraints.txt
```

Why pinned? WhisperX and other tools are not compatible with newer PyTorch versions.

## Quick start (graphical interface)

```bash
python canto_gui.py
```

1. **Apri brano…** *(Open song…)* — mp3, wav, flac, m4a, ogg
2. **Testo (.txt / .lrc)…** *(Lyrics file)* and choose the language
3. **Prepara** *(Prepare)* — the first time it separates the voice and analyses the song (a few minutes)
4. Put on headphones → **Avvia** *(Start, or the space bar)* and sing

What you see:

- **Blue line** = melody of the song · **dots** = your voice
- 🟢 within 40 cents · 🟡 up to 1 semitone · 🔴 out of tune
- **Karaoke bar** at the top: the current word is **fixed in the centre**, it stays until the next one starts and
  **fills with blue** while you sing it; neighbouring words fade with distance
- **At the end of the song:** the verdict, the hardest words, and a graph of the **whole performance** that you can
  pan (drag or scroll bar) and zoom (mouse wheel). `report.png` is saved in `canto_data/<song>/`

### Controls (Italian label → meaning)

| Control | Meaning |
|---|---|
| **Tipo** | Type: *Canzone* (Song), *Esercizio (voce + base)* (Exercise: voice + backing), *Esercizio senza base* (Exercise without backing) |
| **Lingua** | Language of the lyrics (it, en, es, fr, de) |
| **Ascolta** | Listen to the reference without recording |
| **Microfono** | Microphone |
| **Latenza** | Latency compensation in ms (default 80; raise it if your dots look late) |
| **Voce guida** | Guide voice: volume of the original singer |
| **Tonalità** | Key: transposes backing, guide voice and reference melody by −12 … +12 semitones without changing the speed |
| **Applica tonalità suggerita** | Apply suggested key: appears after a session if you are consistently above or below the song |
| **Libreria** (left panel) | Library of prepared songs, most recently practised first; one click loads a song |

The last key used is remembered for each song.

## Command line

```bash
python canto.py prep    song.mp3 --lyrics lyrics.txt --lang it   # or --lrc lyrics.lrc
python canto.py sing    song.mp3                                  # plays the backing and records you
python canto.py compare song.mp3 [--user mine.wav] [--latency 100]
```

`compare` prints the verdict and the table of the hardest words, and saves the graph.

## Exercises and vocalises

In the **Tipo** menu (or with `--mode` on the command line) you can also load audio without lyrics:

| Type | When to use it | What happens |
|---|---|---|
| **Canzone** (Song) | a song with voice and instruments | separation + synced lyrics |
| **Esercizio (voce + base)** (Exercise: voice + backing) | a vocalise with piano or backing | separation, but **note names** (Do4, Re4…) are shown instead of words |
| **Esercizio senza base** (Exercise without backing) | voice only (e.g. a teacher's demonstration) or single piano notes | no separation: the audio is used as it is, and **Voce guida** starts at 100% |

- **Ascolta** plays the reference without recording, so you can hear the exercise first.
- The comparison works by **singing along** with the audio. "Listen and repeat" (singing in the pauses) is not supported yet.
- If the audio is a piano playing **chords** (no melodic line), there is no melody to compare: single notes are needed.

## How it works

```
song ──Demucs──▶ voice + backing
voice ──CREPE──▶ melody (MIDI, 10 ms) ──┐
lyrics ──WhisperX / .lrc──▶ words + timings ─┴─▶ note per word ──▶ ref.json

microphone ──CREPE tiny──▶ your note ──▶ comparison (±120 ms, octaves ignored) ──▶ colours + verdict
```

- The **verdict** is the median of the differences: beyond ±0.35 semitones you are systematically above or below the song.
- **Word timings:** with an `.lrc` file the words of each verse are spread only over the time where the voice is actually
  present; with a `.txt` file WhisperX provides the timings, which are then snapped to the sung sound.
- Everything is cached in `canto_data/<song>/` (`vocals.wav`, `no_vocals.wav`, `ref.json`, `user.wav`, `report.png`):
  a song is prepared **only once**.
- Live, the small "tiny" CREPE model keeps latency low; the final graph and report use the full model.

The code is heavily commented on purpose, so it can be used to learn how the pieces fit together.

## Useful parameters

| Where | Parameter | Default |
|---|---|---|
| `canto.py` | confidence threshold of a note (`per`) | 0.3 |
| `canto.py` | silence threshold (`rms`) | −45 dB |
| `canto.py` | timing tolerance (`tol` in `frame_compare`) | ±120 ms |
| `canto.py` | colour / verdict thresholds | 0.4 · 1 · 0.35 semitones |
| `canto_gui.py` | latency (field in the interface) | 80 ms |
| `canto_gui.py` | graph window (`WIN`, `BEHIND`) | 8 s, 2.4 s of past |

## Limitations

- It judges **pitch**, not rhythm (the ±120 ms tolerance hides small timing errors).
- If Demucs leaves residues of instruments or backing vocals, that part of the melody will be wrong.
- One note per word is a simplification: long words change note.
- Large transpositions (beyond ±4 semitones) may sound slightly metallic.
- Capturing audio from streaming services is **not** supported.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Demucs cannot save the files | `pip install torchcodec -c constraints.txt` |
| WhisperX does not install or conflicts with PyTorch | use `--lrc` (lyrics with timings) |
| Dots always early or late | adjust **Latenza** (Bluetooth headphones add a lot of delay) |
| No audio / "Illegal combination of I/O devices" | pick another microphone from the list; set your headset as default in Windows |
| "Troppa poca voce rilevata" (too little voice detected) | move the microphone closer, raise the input volume |

More in [INSTALL.md](INSTALL.md), section 8.

## Licences and legal notes

- Add your project licence (e.g. MIT) in a `LICENSE` file.
- **Do not upload** songs, separated tracks or song lyrics to the repository: they are protected by copyright.
  Put `canto_data/`, `*.wav` and `*.mp3` in your `.gitignore`.
- Dependencies: **PySide6** is LGPL v3 (if you distribute an executable, include the licence and make the PySide source
  available). **Demucs**: the code is MIT, but there is uncertainty about the **model weights** for commercial use.
  The WhisperX alignment models have their own licences per language. Check before any commercial use.
- This is not legal advice.

## Project structure

```
canto.py          audio processing and command line
canto_gui.py      karaoke graphical interface
requirements.txt  dependencies (install PyTorch first)
INSTALL.md        foolproof step-by-step installation guide
```
