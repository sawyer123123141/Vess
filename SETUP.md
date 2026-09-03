# Setup

## One-time

1. **Ollama** — install from ollama.com, then:
   ```
   ollama pull qwen2.5:7b
   ollama pull llava:7b
   ```

2. **Keep the model warm.** Set a system environment variable:
   ```
   OLLAMA_KEEP_ALIVE = -1
   ```
   (Win+R -> `sysdm.cpl` -> Advanced -> Environment Variables -> System
   variables -> New.) Restart Ollama afterwards. Without this, the first
   request after idle takes 15s and the whole thing feels broken.

3. **Python deps:**
   ```
   pip install -r requirements.txt
   ```

4. **Fill in `config.json`** — the `apps` block needs real paths for anything
   you want it to be able to open.

## Running

```
python main.py
```

Or double-click `vess.bat`, which does the same and opens the browser.

## Optional Chatterbox Turbo evaluation

Kokoro remains the repository default while the RTX 3070 memory-placement plan
is still being decided. Chatterbox Turbo has now been physically validated on
the target PC, but the full Vess process can nearly fill the 8GB GPU when Qwen,
CUDA Whisper, Chatterbox, and the desktop are resident together.

### Known-good Windows / RTX 3070 install

Use a matched CUDA PyTorch trio first:

```powershell
python -m pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Then install the Vess Chatterbox extras. This second step restores the
Chatterbox-compatible NumPy and setuptools constraints after PyTorch's install
has resolved its own dependencies:

```powershell
python -m pip install -r requirements-chatterbox.txt
```

The currently validated extra stack is:

- `chatterbox-tts==0.1.7`
- `numpy==1.26.4`
- `setuptools<81`
- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`

`setuptools<81` is intentional. The current `resemble-perth` package imports
deprecated `pkg_resources`; with setuptools 84 on the target machine,
`PerthImplicitWatermarker` became `None` and Chatterbox failed during model
construction. Setuptools 80.9.0 was verified working. The resulting
`pkg_resources is deprecated` warning is expected until Perth updates its API.

Verify the CUDA and Chatterbox imports:

```powershell
python -c "import numpy, torch, torchvision, torchaudio; print('numpy:', numpy.__version__); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('torchaudio:', torchaudio.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; print('Chatterbox Turbo OK')"
```

To evaluate synthesis without changing the repository default config:

```powershell
python tools/voice_lab.py tts --engine chatterbox_turbo --runs 3 --expression neutral --intensity 0
python tools/voice_lab.py tts --engine chatterbox_turbo --runs 3 --expression playful --intensity 0.65
```

Voice Lab writes benchmark JSON and WAV files under `artifacts/voice-lab/`.
See `STATUS.md`, `CLAUDE.md`, and the September 2 voice-runtime planning
checkpoint for the measured target-PC latency and VRAM findings before changing
model placement or startup behavior.

If a reference voice is used later, set `voice.chatterbox.reference_audio` in
`config.json` to that local audio path. Leaving it empty uses Chatterbox Turbo's
built-in conditionals.

## Optional

RAM sticks in this machine are rated DDR4-3200 but running at 2400. Enabling
DOCP/XMP in BIOS is free performance. Unrelated to this project.
