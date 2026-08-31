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

## Optional

RAM sticks in this machine are rated DDR4-3200 but running at 2400. Enabling
DOCP/XMP in BIOS is free performance. Unrelated to this project.
