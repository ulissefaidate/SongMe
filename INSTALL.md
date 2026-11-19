# Canto — Foolproof Installation Guide

This guide assumes **zero experience**. Follow the steps **in order** and do the **✅ Check** after each one.
If a check fails, do not continue: go to [Troubleshooting](#8-troubleshooting).

- **Time:** 30–60 minutes, mostly downloads.
- **Disk space:** 5–15 GB (PyTorch and the AI models are big).
- **Tested on:** Windows 11 with an NVIDIA GPU. The CPU-only path uses the same steps with a different PyTorch download.
  macOS and Linux notes are provided but **have not been tested** by the author.

---

## 1. Choose your path

Press `Win + R`, type `dxdiag`, press Enter, open the **Display** tab: the *Name* field shows your graphics card.

| You have… | Take this path | Speed |
|---|---|---|
| Windows + **NVIDIA** card (GeForce, RTX, GTX, Quadro) | **Path A – GPU** | Fast (song prepared in minutes) |
| Windows + AMD / Intel graphics, or an old laptop | **Path B – CPU only** | Slower, but works (see [section 7](#7-extra-using-canto-without-a-gpu)) |
| macOS | **Path C** | CPU (Apple GPU is not used) |
| Linux | **Path D** | GPU or CPU |

> AMD and Intel graphics cards **cannot** be used with these instructions. Use Path B.

## 2. Mini glossary (so the commands make sense)

| Word | What it is |
|---|---|
| **Python** | The programming language Canto is written in. |
| **pip** | Python's app installer (`pip install something`). |
| **Virtual environment (`.venv`)** | A private folder with Canto's libraries, so nothing else on your PC is touched. |
| **PyTorch** | The library that runs the neural networks. |
| **CUDA** | NVIDIA's technology that lets PyTorch use the graphics card. |
| **Driver** | The software that lets Windows talk to your graphics card. |
| **Demucs** | Neural network that separates voice and music. |
| **CREPE** | Neural network that measures the pitch of a voice. |
| **WhisperX** | Optional: aligns your lyrics text to the audio (finds *when* each word is sung). |
| **ffmpeg** | Tool that reads mp3/m4a files. |

---

## 3. Common steps (everybody, Windows)

### Step 3.1 – Install Python 3.11

1. Go to <https://www.python.org/downloads/windows/> and download **Python 3.11** (64-bit installer).
   Do **not** use 3.13 or newer: some libraries are not ready for it yet.
2. Run the installer and **tick "Add python.exe to PATH"** (bottom of the first window). This is the step people forget.
3. Click **Install Now**.

✅ **Check** – open **Command Prompt**: press the Windows key, type `cmd`, press Enter. Then type:

```
python --version
```

It must print `Python 3.11.x`.

> **Always use Command Prompt (`cmd`), not PowerShell.** PowerShell blocks the activation script
> (error "running scripts is disabled"). If you are in PowerShell, just type `cmd` and press Enter.

### Step 3.2 – Install ffmpeg

In the Command Prompt:

```
winget install Gyan.FFmpeg
```

Accept the terms if asked (type `Y`). **Close the window and open a new Command Prompt.**

✅ **Check:** `ffmpeg -version` prints a version text without errors.

### Step 3.3 – Get the project files

Put `canto.py`, `canto_gui.py` and `requirements.txt` in a folder, for example `C:\canto`.
(From GitHub: green **Code** button → **Download ZIP** → unzip.)

Open that folder in File Explorer, click the **address bar**, type `cmd` and press Enter.
A black window opens **already inside the folder**. Use this window for everything below.

### Step 3.4 – Create the private environment

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

✅ **Check:** the line now starts with **(.venv)**. If you close the window you must run only the second line
(`.venv\Scripts\activate`) again.

---

## 4. Install PyTorch (this is where the paths differ)

PyTorch must be installed **first** and **pinned to version 2.8.0**. Newer versions break WhisperX and
other tools used by Canto.

### Path A – NVIDIA GPU

**4A.1 Update the driver.** Download it from <https://www.nvidia.com/drivers> (or use the NVIDIA App), install it and **restart the PC**.

**4A.2 Check what the driver supports.** In the Command Prompt:

```
nvidia-smi
```

Look at the top right: **CUDA Version: xx.x**. Choose the command in this table:

| CUDA Version shown | Use this download address (`INDEX`) |
|---|---|
| 12.8 or higher (including 13.x) | `https://download.pytorch.org/whl/cu128` |
| 12.6 or 12.7 | `https://download.pytorch.org/whl/cu126` |
| lower than 12.6 | update the driver (4A.1) and check again |

> This number is only the **maximum your driver can handle**. You do **not** need to install the CUDA Toolkit yourself.
> RTX 50-series cards need `cu128`.

**4A.3 Install PyTorch** (replace `INDEX` with the address from the table):

```
pip install torch==2.8.0 torchaudio==2.8.0 --index-url INDEX
```

For most people that is:

```
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

It downloads a few GB.

✅ **Check:**

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

It must print something like `2.8.0+cu128 True`. If it says `False`, see [Troubleshooting](#8-troubleshooting).

### Path B – CPU only (no NVIDIA card)

```
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

✅ **Check:**

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

It must print `2.8.0+cpu False`. **`False` is correct here** — you have no GPU.
Then read [section 7](#7-extra-using-canto-without-a-gpu).

### Path C – macOS (untested)

```
brew install ffmpeg                       # needs Homebrew: https://brew.sh
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 torchaudio==2.8.0
```

Canto runs on the CPU on a Mac (the code only uses NVIDIA GPUs). Read [section 7](#7-extra-using-canto-without-a-gpu).

### Path D – Linux (untested)

```
sudo apt install ffmpeg libportaudio2 python3-venv libxcb-cursor0
python3 -m venv .venv && source .venv/bin/activate
# NVIDIA GPU: use the cu128 / cu126 address from the table above; no GPU: use the cpu address
pip install torch==2.8.0 torchaudio==2.8.0 --index-url INDEX
```

---

## 5. Install the rest of the libraries

### Step 5.1 – Lock the PyTorch version

Create a text file named **`constraints.txt`** in the project folder (Notepad → *Save as*, file type *All files*)
with exactly these two lines:

```
torch==2.8.0
torchaudio==2.8.0
```

This stops any other library from silently replacing PyTorch with another version.

### Step 5.2 – Install in groups

Doing it in groups tells you which one fails. Every command ends with `-c constraints.txt`.

**Group A – basics:**

```
pip install numpy scipy soundfile sounddevice librosa matplotlib PySide6 pyqtgraph -c constraints.txt
```

**Group B – voice separation and pitch:**

```
pip install demucs torchcrepe -c constraints.txt
```

**Group C – lyric alignment (optional):**

```
pip install whisperx -c constraints.txt
```

> If Group C fails, **skip it**. You can still use lyrics files with times (`.lrc`) — see section 6.

### Step 5.3 – Final checks

```
python -c "import demucs, torchcrepe, librosa, sounddevice, soundfile, PySide6, pyqtgraph; print('OK')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip check
```

✅ The first line prints `OK`. The second still shows `2.8.0+cu128 True` (GPU) or `2.8.0+cpu False` (CPU).
`pip check` lists conflicts (a conflict about *whisperx* is the only one you may ignore if you skip Group C).

If the PyTorch line changed, repeat step 4 adding `--force-reinstall` at the end of the command.

---

## 6. First run, step by step

Start the program (with `(.venv)` active):

```
python canto_gui.py
```

The window is in Italian. Here is the translation of every control:

| On screen | Meaning |
|---|---|
| **Apri brano…** | Open song… (mp3, wav, flac, m4a, ogg) |
| **Tipo** → *Canzone* / *Esercizio (voce + base)* / *Esercizio senza base* | Type → Song / Exercise (voice + backing) / Exercise without backing |
| **Testo (.txt / .lrc)…** | Lyrics file (plain text or `.lrc` with times) |
| **Lingua** | Language of the lyrics (it, en, es, fr, de) |
| **Prepara** | Prepare (one-time analysis of the song) |
| **Avvia / Stop** | Start / Stop (the space bar does the same) |
| **🔊 Ascolta** | Listen to the reference without recording |
| **Microfono** | Microphone |
| **Latenza** | Latency compensation in ms (start with 80) |
| **Voce guida** | Guide voice: volume of the original singer |
| **Tonalità** | Key: transpose the song −12 … +12 semitones |
| **Libreria** (left panel) | Library of songs you already prepared — one click loads them |
| **Applica tonalità suggerita** | Apply suggested key (appears after a session) |

**Your first song:**

1. Pick a **short** song (1–2 minutes) with a clearly audible voice, and put on **headphones**
   (without them the microphone also records the music and the results are wrong).
2. **Apri brano…** → choose the file. Leave **Tipo** on *Canzone*.
3. **Testo (.txt / .lrc)…** → choose your lyrics file and set **Lingua**.
   - `.txt`: one verse per line (needs Group C from step 5.2).
   - `.lrc`: lines like `[0:23.5] verse text` (needs nothing extra).
   - No lyrics at all? Use **Tipo → Esercizio (voce + base)**: note names are shown instead of words.
4. **Prepara**. The first time it downloads AI models (a few GB) and analyses the song. A moving bar shows it is working.
   **Do not close the window.** Next time the same song loads instantly from the library.
5. Choose your **Microfono**, keep **Latenza** at 80, press **Avvia** and sing.
   Green = in tune, yellow = unsure, red = out of tune.
6. At the end you get a verdict (sing higher or lower?), the hardest words, and a graph you can scroll.
   If the app suggests a key, press **Applica tonalità suggerita** and try again.

> Please use songs and lyrics you are allowed to use. Do not upload songs, separated tracks or lyrics to a public repository.

### Next time (one double-click)

1. In the project folder create a new text file and rename it **`start_canto.bat`**
   (make sure it does not stay `start_canto.bat.txt`).
2. Open it with Notepad, paste, save:

```
@echo off
call .venv\Scripts\activate
python canto_gui.py
pause
```

3. From now on, double-click `start_canto.bat`.

---

## 7. Extra: using Canto without a GPU

Everything works on a CPU. Only the **heavy one-time preparation** is slower; singing live is light.
The figures below are rough estimates, not measurements: expect **several times the length of the song**
for separation on an ordinary laptop (for example 5–20 minutes for a 3-minute song).

### Where the time goes (and what to do)

| Step | Why it is slow on CPU | What you can do |
|---|---|---|
| Voice separation (Demucs) | Biggest cost | Do it **once**: results are cached in `canto_data/<song>/`. Or use one of the tricks below |
| Melody extraction (CREPE "full") | Large network | Optional: switch to the small model (see below) |
| Lyric alignment (WhisperX) | Neural network again | Use a `.lrc` file instead of `.txt`: no network needed |
| Live singing | Small model, every 40 ms | If the window stutters, slow the timer (see below) |

### Trick 1 – Do the separation for free in Google Colab (GPU)

1. Open <https://colab.research.google.com>, create a notebook, choose *Runtime → Change runtime type → GPU*.
2. Upload your song and run:

```
!pip install demucs
!python -m demucs --two-stems=vocals -n htdemucs "song.mp3"
```

3. Download `vocals.wav` and `no_vocals.wav` from the output folder (`separated/htdemucs/<song name>/`).
4. On your PC create the folder `canto_data\<song name>\` (same name as the mp3 file, without `.mp3`)
   and put both files inside.
5. Run **Prepara** in Canto. It sees that `vocals.wav` already exists and **skips the separation**.

### Trick 2 – Skip separation completely

Use audio that is **already voice-only** (a cappella, a teacher demonstrating an exercise) or **single-note piano**:
choose **Tipo → Esercizio senza base**. No separation happens and preparation takes seconds.

### Trick 3 – Make the analysis faster (small edit)

In `canto.py`, function `pitch_track`, change `model="full"` to `model="tiny"`.
Preparation becomes much faster; the pitch is a bit less precise.

### Trick 4 – Live window stutters

In `canto_gui.py`, find `interval=40` (the pitch timer) and change it to `interval=80`.
The coloured dots update less often but the app stays smooth.

### CPU-only checklist

- Prefer short songs and `.lrc` lyrics.
- Prepare songs in advance (they stay in the **Libreria** panel).
- Close heavy programs (browser with many tabs) while preparing.

---

## 8. Troubleshooting

| Problem | What to do |
|---|---|
| `python` is not recognised | You forgot *Add python.exe to PATH*. Uninstall Python, reinstall ticking it, open a **new** Command Prompt. |
| "Running scripts is disabled" when activating `.venv` | You are in PowerShell. Type `cmd` and use `.venv\Scripts\activate` there. |
| `nvidia-smi` not recognised | NVIDIA driver missing or PC not restarted (step 4A.1). |
| GPU check prints `False` on an NVIDIA PC | PyTorch was replaced by the CPU version. Repeat step 4A.3 adding `--force-reinstall`, then repeat the check. |
| `pip install -r …` / Group C errors | Check `(.venv)` is shown and Python is 3.11. If the error mentions **whisperx**, skip Group C and use `.lrc` lyrics. |
| Demucs cannot save files | `pip install torchcodec -c constraints.txt` and try again. |
| "Illegal combination of I/O devices" | Use the latest `canto_gui.py` and pick a microphone from the list (it only shows compatible ones). Set your headset as default input/output in Windows *Settings → System → Sound*. |
| No sound, or the microphone records nothing | Choose another microphone in the list; check Windows sound settings and microphone privacy permissions. |
| "Troppa poca voce rilevata" (too little voice) | Move the microphone closer, raise the input volume in Windows. |
| Dots always early or late | Change **Latenza**. Bluetooth headphones add a lot of delay and make results less precise: use wired ones. |
| Graph does not start after Avvia | The first start loads the model: wait a few seconds. |
| Big transposition sounds metallic | Normal beyond ±4 semitones. Optional: install `pyrubberband` and the `rubberband` program for better quality. |
| Nothing works and you are lost | Delete the `.venv` folder and redo steps 3.4 → 5.3 (your songs in `canto_data` stay safe). |

## 9. Updating and uninstalling

- **Update Canto:** replace `canto.py` and `canto_gui.py` with the new versions; keep `canto_data`.
- **Free disk space:** delete `canto_data\<song>\` folders you no longer need (they hold the separated audio).
- **Uninstall:** delete the project folder; optionally remove Python from *Settings → Apps*.
  The downloaded AI models live in your user folder (`C:\Users\<you>\.cache`).
