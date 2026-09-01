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

Kokoro remains Vess's default TTS engine. Chatterbox Turbo is an optional
experimental engine until it is benchmarked on the target PC alongside the
resident Ollama model.

Install its extra dependencies separately:

```powershell
pip install -r requirements-chatterbox.txt
```

To evaluate it without changing the default config, run the benchmark harness:

```powershell
python tools/benchmark_tts.py --engine chatterbox_turbo --runs 3
```

Benchmark JSON and WAV files are written under `artifacts/tts-benchmark/`.
The real acceptance pass still needs the target PC to measure warm synthesis
latency, peak VRAM, Qwen slowdown, stability, and listening quality.

If a reference voice is used later, set `voice.chatterbox.reference_audio` in
`config.json` to that local audio path. Leaving it empty sends no reference
prompt to the model.

## Optional

RAM sticks in this machine are rated DDR4-3200 but running at 2400. Enabling
DOCP/XMP in BIOS is free performance. Unrelated to this project.
