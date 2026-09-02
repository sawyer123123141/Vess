# Vess Voice Corpus

Voice Lab replays local WAV files so endpointing and Whisper changes are compared against identical speech instead of requiring a new live conversation for every setting.

## Audio contract

V1 intentionally accepts one simple format:

- WAV
- mono
- 16,000 Hz
- 16-bit PCM

The lab does not resample or download audio. Convert clips once before adding them here so format conversion cannot silently change between benchmark runs.

Files may come from:

- the owner's real microphone
- synthetic fixtures
- public speech datasets such as Common Voice or LibriSpeech, after conversion to the format above

Keep `source` accurate in the manifest. Owner speech and public speech should be reported separately when choosing Vess settings because general benchmark accuracy is not a substitute for working well for the person who actually uses Vess.

## Suggested first owner recordings

Start with roughly 15-25 clips, not hundreds. Add a new clip when a real failure exposes a missing case.

Useful coverage:

- very short replies: `yes`, `no`, `mhm`
- ordinary one-sentence requests
- longer requests
- natural hesitation before continuing
- a deliberate mid-sentence thinking pause
- filler words such as `uh` / `um`
- quiet speech
- faster speech
- wake phrase plus request
- follow-up without a wake phrase
- punctuation-heavy / comma-heavy sentence
- question intonation
- speech with normal room noise
- the exact wording of any future Whisper failure worth preserving

Do not perform the recordings like a benchmark announcer. The owner corpus is valuable specifically because it captures normal speech habits.

## Manifest

Each manifest item supports:

```json
{
  "id": "owner-hesitation-01",
  "audio": "owner-hesitation-01.wav",
  "transcript": "I was wondering actually tell me why the sky changes color",
  "expected_utterances": 1,
  "source": "owner",
  "tags": ["hesitation", "mid_sentence_pause"],
  "expression": "curious",
  "intensity": 0.55
}
```

`expression` and `intensity` do not change TTS behavior in Voice Lab v1. They are preserved so the next expressive-voice pass can render identical text/cue pairs for A/B listening tests without replacing the corpus format.

See `manifest.example.json` for a complete example.

## Commands

Endpoint sweep:

```powershell
python tools/voice_lab.py endpoint --manifest voice_corpus/manifest.json --silence 0.30 0.35 0.40 0.45
```

Whisper beam comparison:

```powershell
python tools/voice_lab.py whisper --manifest voice_corpus/manifest.json --beam-size 1 3 5
```

TTS benchmark:

```powershell
python tools/voice_lab.py tts --engine chatterbox_turbo --runs 3
```

Deterministic real-hardware cancellation benchmark:

```powershell
python tools/voice_lab.py cancel --engine chatterbox_turbo --text "This is a deliberately longer synthesis request so cancellation can happen during generation." --after-ms 50 100 250 500
```

Results are written under `artifacts/voice-lab/`. Human listening remains the authority for prosody and naturalness; Voice Lab exists to make the objective parts repeatable and to narrow subjective comparisons to a small set of identical WAVs.
